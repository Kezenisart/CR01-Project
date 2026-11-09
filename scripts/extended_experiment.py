"""Compare the two networks after 30 and 100 epochs, using three training seeds.

Run with --project PROJECT --output NEW_DIRECTORY. Existing runs are preserved.
Train every model before opening the test set.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

SEEDS = [4404, 5505, 6606]
BUDGETS = [30, 100]
MODEL_NAMES = ['multiscale', 'single_scale']


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_hashes(root):
    return {
        split: file_hash(root / f'results/deepinv_cpu/{split}.pt')
        for split in ['train', 'validation', 'test']
    }


@torch.no_grad()
def predict(network, inputs, batch_size):
    return torch.cat([network(batch) for batch in inputs.split(batch_size)])


def train_model(network, name, seed, config, training, validation, folder):
    """Train one model and save the best validation checkpoint for each budget."""
    folder.mkdir(parents=True)
    # The residual model should initially return the FBP image unchanged.
    assert torch.equal(network(training['fbp'][:1]), training['fbp'][:1])
    optimizer = torch.optim.Adam(network.parameters(), lr=config['learning_rate'])
    # Give both networks the same shuffled image order for a given seed.
    loader = DataLoader(
        TensorDataset(training['fbp'], training['target']),
        batch_size=config['batch_size'],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    history = []
    best_loss = float('inf')
    start = time.perf_counter()

    for epoch in range(1, max(BUDGETS) + 1):
        network.train()
        total_loss = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            loss = (network(inputs) - targets).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)

        # Validation chooses the saved model; test scores never enter this choice.
        network.eval()
        predictions = predict(network, validation['fbp'], config['batch_size'])
        validation_loss = (predictions - validation['target']).square().mean().item()
        history.append({
            'epoch': epoch,
            'train_mse': total_loss / len(training['target']),
            'validation_mse': validation_loss,
            'elapsed_seconds': time.perf_counter() - start,
        })

        if validation_loss < best_loss:
            best_loss = validation_loss
            checkpoint = {
                'state_dict': network.state_dict(),
                'epoch': epoch,
                'validation_mse': validation_loss,
                'seed': seed,
                'name': name,
            }
            # After epoch 30, only the 100-epoch checkpoint can change.
            for budget in BUDGETS:
                if epoch <= budget:
                    torch.save(checkpoint, folder / f'best_{budget}.pt')

        save_json(folder / 'history.json', history)
        if epoch % 10 == 0:
            print(
                f'{seed} {name} {epoch}/100 val={validation_loss:.8f} '
                f'seconds={time.perf_counter() - start:.1f}',
                flush=True,
            )


def evaluate_checkpoint(network, psnr, name, seed, budget, config,
                        validation, test, folder):
    """Load the best saved model and measure its error on the test set."""
    history = json.loads((folder / 'history.json').read_text())
    checkpoint = torch.load(folder / f'best_{budget}.pt', weights_only=True)
    network.eval()
    network.load_state_dict(checkpoint['state_dict'])
    assert checkpoint['validation_mse'] == min(
        epoch['validation_mse'] for epoch in history[:budget]
    )

    validation_predictions = predict(network, validation['fbp'], config['batch_size'])
    validation_loss = (
        validation_predictions - validation['target']
    ).square().mean().item()
    predictions = predict(network, test['fbp'], config['batch_size'])
    assert abs(validation_loss - checkpoint['validation_mse']) < 1e-9
    assert torch.isfinite(predictions).all()

    scores = psnr(predictions, test['target'])
    result = {
        'seed': seed,
        'name': name,
        'budget': budget,
        'selected_epoch': checkpoint['epoch'],
        'validation_mse': validation_loss,
        'psnr_mean': scores.mean().item(),
        'image_sd_population': scores.std(unbiased=False).item(),
        'per_image_psnr': scores.tolist(),
    }
    torch.save(predictions, folder / f'predictions_{budget}.pt')
    save_json(folder / f'metrics_{budget}.json', result)
    return result


def summarize(results, fbp_psnr):
    """Report the mean and spread of the three runs for each model."""
    summary = {'fbp_psnr_mean': fbp_psnr, 'results': results, 'aggregates': {}}
    scores_by_run = {
        (row['budget'], row['seed'], row['name']): row['psnr_mean']
        for row in results
    }
    # Each score is already an average over the test images.
    # Here the standard deviation measures differences between training seeds.
    for budget in BUDGETS:
        aggregates = {}
        for name in MODEL_NAMES:
            rows = [r for r in results if r['budget'] == budget and r['name'] == name]
            scores = torch.tensor([r['psnr_mean'] for r in rows], dtype=torch.float64)
            aggregates[name] = {
                'mean': scores.mean().item(),
                'seed_sd_sample': scores.std(unbiased=True).item(),
                'selected_epochs': [r['selected_epoch'] for r in rows],
            }
        aggregates['paired_gap_single_minus_multi'] = [
            scores_by_run[budget, seed, 'single_scale']
            - scores_by_run[budget, seed, 'multiscale']
            for seed in SEEDS
        ]
        summary['aggregates'][str(budget)] = aggregates
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    output = args.output.resolve()
    if (output / 'protocol.json').exists():
        raise RuntimeError('Choose a new output directory; existing run preserved.')
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    from src.core import model, psnr

    config = json.loads((root / 'configs/cpu.json').read_text())
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    # Record the settings and file hashes before training starts.
    protocol = {
        'training_seeds': SEEDS,
        'budgets': BUDGETS,
        'config': config,
        'epochs': 100,
        'selection': 'lowest validation MSE within each budget',
        'data': 'original fixed train/validation/test splits, unchanged',
        'test_policy': (
            'load test tensors only after all six training runs finish; '
            'original test set reused, no tuning on new test results'
        ),
        'sd': (
            'sample SD of per-seed mean test PSNR (ddof=1), '
            'not across-image SD'
        ),
        'data_hashes': data_hashes(root),
        'core_sha256': file_hash(root / 'src/core.py'),
        'environment': {
            'platform': platform.platform(),
            'python': platform.python_version(),
            'torch': torch.__version__,
            'deepinv': importlib.metadata.version('deepinv'),
        },
        'written_before_training': True,
    }
    save_json(output / 'protocol.json', protocol)
    training = torch.load(root / 'results/deepinv_cpu/train.pt', weights_only=True)
    validation = torch.load(root / 'results/deepinv_cpu/validation.pt', weights_only=True)
    assert training['config'] == validation['config'] == config, 'Data configuration mismatch'

    for seed in SEEDS:
        for name in MODEL_NAMES:
            folder = output / f'seed_{seed}' / name
            torch.manual_seed(seed)
            network = model(name)
            train_model(network, name, seed, config, training, validation, folder)

    # All six runs are finished. We can now compare them on the test set.
    print('Training finished. Checking the saved models on the test set.', flush=True)
    test = torch.load(root / 'results/deepinv_cpu/test.pt', weights_only=True)
    assert test['config'] == config, 'Test configuration mismatch'
    results = []
    checks = []
    for seed in SEEDS:
        for name in MODEL_NAMES:
            folder = output / f'seed_{seed}' / name
            for budget in BUDGETS:
                results.append(evaluate_checkpoint(
                    model(name), psnr, name, seed, budget, config, validation, test, folder,
                ))
                checks.append({
                    'seed': seed, 'name': name, 'budget': budget,
                    'selection_and_recomputed_validation_pass': True,
                })

    summary = summarize(results, psnr(test['fbp'], test['target']).mean().item())
    save_json(output / 'summary.json', summary)
    assert protocol['data_hashes'] == data_hashes(root)
    save_json(output / 'verification.json', {
        'checkpoint_checks': checks,
        'original_data_hashes_unchanged': True,
    })
    print(json.dumps(summary['aggregates'], indent=2), flush=True)


if __name__ == '__main__':
    main()
