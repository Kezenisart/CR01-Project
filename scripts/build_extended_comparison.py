"""Show the first test image after 100 epochs, using seed 4404."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cr01-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report/figures'


def main():
    test = torch.load(ROOT/'results/cpu/test.pt', weights_only=True)
    reference = test['target'][0, 0]
    images = [('Reference', reference), ('FBP', test['fbp'][0, 0])]
    for name, label in [('multiscale', 'Multiscale'), ('single_scale', 'Single-scale')]:
        predictions = torch.load(ROOT/f'results/extended_100ep_3seed/seed_4404/{name}/predictions_100.pt', weights_only=True)
        images.append((label, predictions[0, 0]))
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 12, 'font.family': 'DejaVu Sans'})
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.5), layout='constrained')
    for column, (label, image) in enumerate(images):
        axes[0,column].imshow(image, cmap='gray', vmin=0, vmax=1)
        axes[0,column].set_title(label)
        if column:
            error_plot = axes[1,column].imshow((image-reference).abs(), cmap='magma', vmin=0, vmax=.3)
            axes[1,column].set_title('Absolute error')
    for ax in axes.flat:
        ax.axis('off')
    # The reference has no error panel: comparing it with itself adds no information.
    fig.colorbar(error_plot, ax=list(axes[1,1:]), shrink=.78, ticks=[0,.15,.3], fraction=.025, pad=.02)
    OUT.mkdir(exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(OUT/f'extended_comparison.{ext}', dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
