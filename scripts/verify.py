"""Check the saved data, models and scores, then write verification.json."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core import model, phantoms, physics, psnr


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_data(output, config, manifest, projector, checks):
    """Recreate each data split and compare it with the saved version."""
    splits = {}
    image_hashes = []
    for split, count in config['counts'].items():
        path = output / f'{split}.pt'
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        require(file_hash == manifest['splits'][split]['sha256'], f'{split}: file hash')

        data = torch.load(path, weights_only=True)
        require(data['config'] == config, f'{split}: config')
        targets = data['target']
        expected_shape = (count, 1, config['size'], config['size'])
        require(tuple(targets.shape) == expected_shape, f'{split}: target shape')
        regenerated = phantoms(count, config['size'], config['seeds'][split])
        require(torch.equal(targets, regenerated), f'{split}: regeneration')
        require(bool(((targets >= 0) & (targets <= 1)).all()), f'{split}: intensity range')
        for key in ['target', 'sinogram', 'fbp']:
            require(bool(torch.isfinite(data[key]).all()), f'{split}: finite {key}')

        # Recompute the measurements and FBP images using the same geometry.
        sinograms = []
        reconstructions = []
        for batch in targets.split(config['batch_size']):
            sinogram = projector.A(batch)
            sinograms.append(sinogram.detach())
            reconstructions.append(projector.A_dagger(sinogram, fbp=True).detach())
        torch.testing.assert_close(
            torch.cat(sinograms), data['sinogram'], rtol=1e-5, atol=1e-6,
        )
        torch.testing.assert_close(
            torch.cat(reconstructions), data['fbp'], rtol=1e-5, atol=1e-6,
        )
        for image in targets:
            image_hashes.append(hashlib.sha256(image.numpy().tobytes()).hexdigest())
        splits[split] = data
        checks.append(
            f'{split}: hash, config, shape, range, exact targets, forward operator and FBP'
        )

    # Check all splits together to catch overlap between training and test images.
    require(len(set(image_hashes)) == len(image_hashes), 'Duplicate target images')
    checks.append('No duplicate target images within or across splits')
    return splits


def check_physics(config, projector, test_data):
    """Check the adjoint, reconstruction orientation, and PSNR calculation."""
    torch.manual_seed(0)
    image = torch.randn_like(test_data['target'][:1])
    sinogram = torch.randn_like(test_data['sinogram'][:1])

    # The two inner products should agree: <Ax, y> = <x, A* y>.
    forward_product = (projector.A(image) * sinogram).sum()
    adjoint_product = (image * projector.A_adjoint(sinogram)).sum()
    scale = torch.maximum(forward_product.abs(), adjoint_product.abs()).clamp_min(1e-8)
    adjoint_error = ((forward_product - adjoint_product).abs() / scale).item()
    require(adjoint_error < 1e-4, 'Adjoint consistency')

    target = phantoms(1, config['size'], 5678)
    sparse_fbp = projector.A_dagger(projector.A(target), fbp=True)
    dense_projector = physics(config['size'], config['dense_views'])
    dense_fbp = dense_projector.A_dagger(dense_projector.A(target), fbp=True)
    require(psnr(dense_fbp, target).item() > psnr(sparse_fbp, target).item(),
            'Dense-view improvement')

    normal_error = (sparse_fbp - target).square().mean()
    flipped_error = (sparse_fbp - target.flip(-1)).square().mean()
    require(normal_error < flipped_error, 'Orientation')

    # A constant error of 0.1 should give 20 dB when the data range is 1.
    known_score = psnr(torch.full_like(target, 0.1), torch.zeros_like(target)).item()
    require(abs(known_score - 20) < 1e-5, 'Known PSNR value')
    return adjoint_error, sparse_fbp


def check_network(name, output, config, splits, summary, example_fbp):
    """Check the saved model and return its test predictions."""
    checkpoint = torch.load(output / f'{name}.pt', weights_only=True)
    require(checkpoint['config'] == config and checkpoint['name'] == name,
            f'{name}: checkpoint metadata')
    network = model(name).eval()
    with torch.no_grad():
        # Before training, the residual correction should be zero.
        require(torch.equal(network(example_fbp), example_fbp),
                f'{name}: identity initialization')
    network.load_state_dict(checkpoint['state_dict'])

    history = json.loads((output / f'{name}_history.json').read_text())
    epochs = [entry['epoch'] for entry in history]
    require(epochs == list(range(1, config['epochs'] + 1)), f'{name}: epoch history')
    best_epoch = min(history, key=lambda entry: entry['validation_mse'])
    require(
        checkpoint['epoch'] == best_epoch['epoch']
        and checkpoint['validation_mse'] == best_epoch['validation_mse'],
        f'{name}: best validation selection',
    )

    with torch.no_grad():
        predictions = torch.cat([
            network(batch) for batch in splits['test']['fbp'].split(config['batch_size'])
        ])
        validation_predictions = torch.cat([
            network(batch)
            for batch in splits['validation']['fbp'].split(config['batch_size'])
        ])
    validation_loss = (
        validation_predictions - splits['validation']['target']
    ).square().mean().item()
    require(abs(validation_loss - checkpoint['validation_mse']) < 1e-8,
            f'{name}: validation MSE')
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    require(parameter_count == summary['parameters'], f'{name}: parameters')
    return predictions


def check_scores(name, predictions, targets, summary, csv_rows):
    """Compare the recomputed PSNR with the JSON summary and every CSV row."""
    scores = psnr(predictions, targets)
    mean_score = scores.mean().item()
    score_sd = scores.std(unbiased=False).item()
    require(abs(mean_score - summary['psnr_mean']) < 1e-5, f'{name}: mean PSNR')
    require(abs(score_sd - summary['psnr_std']) < 1e-5, f'{name}: PSNR SD')

    records = [row for row in csv_rows if row['method'] == name]
    indices = sorted(int(row['test_index']) for row in records)
    require(indices == list(range(len(scores))), f'{name}: CSV indices')
    for row in records:
        image_index = int(row['test_index'])
        saved_score = float(row['psnr_db'])
        require(abs(saved_score - scores[image_index].item()) < 1e-5,
                f'{name}: CSV score')
    return {'psnr_mean': mean_score, 'psnr_std': score_sd}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/deepinv_cpu')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/cpu.json')
    args = parser.parse_args()
    output = args.output
    config = json.loads(args.config.read_text())
    manifest = json.loads((output / 'manifest.json').read_text())
    metrics = json.loads((output / 'metrics.json').read_text())
    require(manifest['config'] == config == metrics['config'], 'Configuration mismatch')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)

    # Start with the data and CT operator before checking the trained networks.
    checks = []
    projector = physics(config['size'], config['views'])
    splits = check_data(output, config, manifest, projector, checks)
    adjoint_error, example_fbp = check_physics(config, projector, splits['test'])
    checks.append('Adjoint consistency, dense-view improvement, orientation, known PSNR')

    saved_predictions = torch.load(output / 'predictions.pt', weights_only=True)
    with (output / 'per_image_metrics.csv').open() as file:
        csv_rows = list(csv.DictReader(file))
    require(len(csv_rows) == 3 * config['counts']['test'], 'CSV row count')

    scores_report = {}
    for name in ['fbp', 'multiscale', 'single_scale']:
        summary = metrics['methods'][name]
        if name == 'fbp':
            predictions = splits['test']['fbp']
        else:
            predictions = check_network(
                name, output, config, splits, summary, example_fbp,
            )
        torch.testing.assert_close(
            predictions, saved_predictions[name], rtol=1e-5, atol=1e-6,
        )
        scores_report[name] = check_scores(
            name, predictions, splits['test']['target'], summary, csv_rows,
        )
        description = f'{name}: predictions, summary and per-image metrics'
        if name != 'fbp':
            description += ', initialization, checkpoint, validation loss and parameters'
        checks.append(description)

    result = {
        'status': 'passed',
        'checks': checks,
        'adjoint_relative_error': adjoint_error,
        'methods': scores_report,
        'scope': (
            'Verifies saved experiment; no new training or hyperparameter selection. '
            'Timing measurements are not rerun.'
        ),
    }
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
