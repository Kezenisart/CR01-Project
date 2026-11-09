"""Plot how validation error changes during the 100 training epochs."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/cr01-extended-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MODELS = [
    ('multiscale', 'Multiscale', '#147d92'),
    ('single_scale', 'Single-scale', '#b54d35'),
]


def load_validation_losses(results_dir, seeds, name):
    losses = []
    for seed in seeds:
        path = results_dir / f'seed_{seed}' / name / 'history.json'
        history = json.loads(path.read_text())
        losses.append([epoch['validation_mse'] for epoch in history])
    losses = np.array(losses)
    assert losses.shape == (3, 100)
    return losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.results / 'protocol.json').read_text())
    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})
    figure, axis = plt.subplots(figsize=(4.5, 2.65), layout='constrained')
    epochs = np.arange(1, 101)

    for name, label, color in MODELS:
        losses = load_validation_losses(args.results, protocol['training_seeds'], name)
        # Shade the range of the three runs, then draw their average.
        axis.fill_between(
            epochs, losses.min(0), losses.max(0), color=color, alpha=.15,
        )
        axis.plot(epochs, losses.mean(0), label=label, color=color, lw=1.8)

    axis.axvline(30, color='#666666', linestyle='--', lw=.9)
    axis.set(xlabel='Epoch', ylabel='Validation MSE', yscale='log', xlim=(1, 100))
    axis.legend(frameon=False)
    axis.grid(alpha=.15)
    axis.spines[['top', 'right']].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix('.pdf'))
    figure.savefig(args.output.with_suffix('.png'), dpi=180)
    plt.close(figure)


if __name__ == '__main__':
    main()
