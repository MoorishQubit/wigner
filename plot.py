import pylab as pl
pl.style.use('seaborn-v0_8-white')
pl.rcParams.update({
    "text.usetex": True,
})
for r in [1, 2, 3, 4]:
    d = pl.load(f"samples_{r}.npy")
    fig, ax = pl.subplots(nrows=2, squeeze=True)
    ax[0].scatter(d[:, 0], d[:, 1], s=1)
    ax[1].scatter(d[:, 0], d[:, 2], s=1)

    ax[0].set_xlabel("Wootters negativity")
    ax[0].set_ylabel("Negativity")

    ax[1].set_xlabel("Wootters negativity")
    ax[1].set_ylabel("Log negativity")

    #ax[0].tick_params(axis='both', which='major', labelsize=12)
    #ax[1].tick_params(axis='both', which='major', labelsize=12)

    fig.tight_layout()
    fig.savefig(f"figs/plot_{r}.png")
    pl.close()