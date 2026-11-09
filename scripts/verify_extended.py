"""Check that the saved predictions, checkpoints and reported scores agree."""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

MODEL_NAMES = ['multiscale', 'single_scale']


def verify_predictions(results_dir, rows, target, psnr):
    for row in rows:
        folder = results_dir / f"seed_{row['seed']}" / row['name']
        budget = row['budget']
        predictions = torch.load(folder / f'predictions_{budget}.pt', weights_only=True)
        scores = psnr(predictions, target)
        assert torch.equal(scores, torch.tensor(row['per_image_psnr']))
        assert abs(scores.mean().item() - row['psnr_mean']) < 1e-7

        # The saved checkpoint must have the lowest validation error in its budget.
        checkpoint = torch.load(folder / f'best_{budget}.pt', weights_only=True)
        history = json.loads((folder / 'history.json').read_text())
        assert len(history) == 100
        assert checkpoint['epoch'] == row['selected_epoch']
        assert checkpoint['validation_mse'] == min(
            epoch['validation_mse'] for epoch in history[:budget]
        )


def verify_aggregates(summary, budgets):
    for budget in budgets:
        for name in MODEL_NAMES:
            scores = [
                row['psnr_mean'] for row in summary['results']
                if row['budget'] == budget and row['name'] == name
            ]
            aggregate = summary['aggregates'][str(budget)][name]
            assert abs(statistics.mean(scores) - aggregate['mean']) < 1e-10
            assert abs(statistics.stdev(scores) - aggregate['seed_sd_sample']) < 1e-10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    results_dir = args.results.resolve()
    sys.path.insert(0, str(root))
    from src.core import psnr

    protocol = json.loads((results_dir / 'protocol.json').read_text())
    summary = json.loads((results_dir / 'summary.json').read_text())
    # A changed dataset or model file would make this a different experiment.
    for split, expected_hash in protocol['data_hashes'].items():
        data = (root / f'results/deepinv_cpu/{split}.pt').read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected_hash
    source = (root / 'src/core.py').read_bytes()
    assert hashlib.sha256(source).hexdigest() == protocol['core_sha256']

    target = torch.load(root / 'results/deepinv_cpu/test.pt', weights_only=True)['target']
    verify_predictions(results_dir, summary['results'], target, psnr)
    verify_aggregates(summary, protocol['budgets'])

    # Seed 4404 should reproduce the original 30-epoch experiment.
    original = json.loads((root / 'results/deepinv_cpu/metrics.json').read_text())
    for name in MODEL_NAMES:
        result = next(
            row for row in summary['results']
            if row['seed'] == 4404 and row['budget'] == 30 and row['name'] == name
        )
        assert abs(result['psnr_mean'] - original['methods'][name]['psnr_mean']) < 1e-7
    print(
        'PASS: 12 saved prediction arrays, per-image scores, summary means/SD, '
        'checkpoint selection, original-run reproduction, and unchanged source/data hashes.'
    )


if __name__ == '__main__':
    main()
