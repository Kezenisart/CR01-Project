"""Plot the validation curves and one reconstruction from the saved results."""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/cr01-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'report/figures'
MODELS = [
    ('multiscale', 'Multiscale', '#147d92'),
    ('single_scale', 'Single-scale', '#b54d35'),
]


def plot_validation():
    figure, axis = plt.subplots(figsize=(4.5, 2.65), layout='constrained')
    for name, label, color in MODELS:
        history = json.loads((ROOT / f'results/cpu/{name}_history.json').read_text())
        epochs = [entry['epoch'] for entry in history]
        losses = [entry['validation_mse'] for entry in history]
        axis.plot(epochs, losses, label=label, color=color, lw=1.8)
    axis.set(xlabel='Epoch', ylabel='Validation MSE', yscale='log')
    axis.legend(frameon=False)
    axis.grid(alpha=.2)
    axis.spines[['top', 'right']].set_visible(False)
    figure.savefig(OUTPUT / 'validation.pdf')
    figure.savefig(OUTPUT / 'validation.png', dpi=180)
    plt.close(figure)


def plot_comparison():
    test = torch.load(ROOT / 'results/cpu/test.pt', weights_only=True)
    predictions = torch.load(ROOT / 'results/cpu/predictions.pt', weights_only=True)
    # Always use the first test image so the example is not hand-picked.
    reference = test['target'][0, 0]
    images = [('Reference', reference), ('FBP', predictions['fbp'][0, 0])]
    images += [(label, predictions[name][0, 0]) for name, label, _ in MODELS]
    figure, axes = plt.subplots(2, 4, figsize=(7.1, 3.5), layout='constrained')
    for column, (label, image) in enumerate(images):
        axes[0, column].imshow(image, cmap='gray', vmin=0, vmax=1)
        axes[0, column].set_title(label)
        error_plot = axes[1, column].imshow(
            (image - reference).abs(), cmap='magma', vmin=0, vmax=.3,
        )
        axes[1, column].set_title('Absolute error', fontsize=11)
    for axis in axes.flat:
        axis.axis('off')
    figure.colorbar(
        error_plot, ax=list(axes[1]), shrink=.78,
        ticks=[0, .15, .3], fraction=.025, pad=.02,
    )
    figure.savefig(OUTPUT / 'comparison.pdf')
    figure.savefig(OUTPUT / 'comparison.png', dpi=200)
    plt.close(figure)


def main():
    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 12, 'font.family': 'DejaVu Sans',
    })
    plot_validation()
    plot_comparison()


if __name__ == '__main__':
    main()
