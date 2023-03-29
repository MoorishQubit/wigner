import pylab as pl
pl.rcParams.update({
    "text.usetex": True,
})
for r in [1, 2, 3, 4]:
    d = pl.load(f"samples_{r}.npy")
    fig, ax = pl.subplots(nrows=2, squeeze=True)
    ax[0].scatter(d[:, 0], d[:, 1], s=1)
    ax[1].scatter(d[:, 0], d[:, 2], s=1)

    ax[0].set_xlabel("Wigner negativity")
    ax[0].set_ylabel("Negativity")

    ax[1].set_xlabel("Wigner negativity")
    ax[1].set_ylabel("Log negativity")
    fig.savefig(f"figs/plot_{r}.png")
    pl.close()