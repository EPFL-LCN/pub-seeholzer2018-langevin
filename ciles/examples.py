import numpy as np
import pylab as pl
from .tools import runDistribution, runTrajectories
from matplotlib import mlab


def simple_diffusion():
    """Simulate diffusion from one initial position"""

    B = 0.05
    tmax = 2.
    drift = np.zeros(100)  # no drift field
    diff = np.ones(100) * B  # constant diffusion with 1 deg^2/s

    finals = runDistribution(
        drift,
        diff,
        dt=.01,
        tmax=tmax,
        reps=100000,
        initials=1)

    finals = finals.reshape(-1)
    finals[np.where(finals > np.pi)] = \
        finals[np.where(finals > np.pi)] - 2*np.pi

    # compare histogram to predicted distribution
    n, bins, patches = pl.hist(finals, bins=100, normed=True)
    y = mlab.normpdf(bins, 0., np.sqrt(B * tmax))
    pl.plot(bins, y, '--', label="Predicted", c="#ff0000")
    pl.title(r"Distribution after diffusion for $t=%g$s" % (tmax))
    pl.xlabel(r"Stochastic variable $x$")
    pl.ylabel(r"Probability density $p(x)$")

    pl.legend()
    pl.show()


def plot_trajectories():
    """Plot several trajectories under drift and diffusion with
    evenly spaced initial positions, with a drift-field A(x)
    that has two fixed points."""

    # create a drift field with 2 fixed points
    xs = np.arange(0, 2.*np.pi, 2.*np.pi/100.)
    drift = np.sin(xs * 2) * 0.5

    # create constant diffusion
    diff = np.ones(100) * 0.1

    # run trajectories
    reps = 100
    initials = 6
    tmax = 10.
    dt = .01
    out = runTrajectories(
        drift,
        diff,
        dt=dt,
        tmax=tmax,
        reps=reps,
        initials=initials)

    # plot data
    cm = pl.cm.rainbow
    for i in range(initials):
        tmp = out[i, :]
        xs = np.arange(0., tmax, dt)
        pos = np.where(np.abs(np.diff(tmp, 1)) >= np.pi)
        tmp[pos] = np.nan
        pl.plot(xs, tmp.T, c=cm(i/float(initials)))

    xlim = (0, tmax)
    xlabel = r"$t$ [s]"

    ylim = (0, np.pi * 2.)
    yticks = [0, np.pi, 2.*np.pi]
    ylabel = r"Stochastic variable $x$"

    ax = pl.subplot(111)
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.set_yticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
    ax.set_yticks(yticks)

    pl.title(r"Drift field with 2 fixed points")
    pl.show()


if __name__ == '__main__':
    simple_diffusion()
    plot_trajectories()
