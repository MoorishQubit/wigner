import pylab as pl
from pathlib import Path
pl.style.use('seaborn-v0_8-white')
pl.rcParams.update({
    "text.usetex": True,
})

p = Path(".")

for fname in p.glob("*.npy"):
    d = pl.load(fname)
    fig, ax = pl.subplots(nrows=3, squeeze=True)
    ax[0].scatter(d[:, 0], d[:, 1], s=1)
    ax[1].scatter(d[:, 0], d[:, 2], s=1)
    ax[2].scatter(d[:, 0], d[:, 3], s=1)

    ax[0].set_xlabel("Wootters negativity")
    ax[0].set_ylabel("Negativity")

    ax[1].set_xlabel("Wootters negativity")
    ax[1].set_ylabel("Log negativity")

    ax[2].set_xlabel("Wootters negativity")
    ax[2].set_ylabel("Concurrence")

    #ax[0].tick_params(axis='both', which='major', labelsize=12)
    #ax[1].tick_params(axis='both', which='major', labelsize=12)

    fig.tight_layout()
    fig.savefig(f"figs/{fname.stem}.png")
    pl.close()