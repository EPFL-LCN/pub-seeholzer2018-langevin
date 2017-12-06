import numpy as np
from .integrator import LangevinIntegrator as LI
import time


def get_initial_poss(initials):
    """Converts initials into an array of initial positions.

    Args:
        initials (int or list or np.array): If int, this is the number
            of (evenly spaced) initial positions. If list or np.array this
            corresponds to the actual initial positions.

    Returns:
        np.array: Array of inital positions
    """
    if type(initials) is int:
        return np.arange(0., 2. * np.pi, 2. * np.pi / float(initials))
    elif type(initials) in [list, np.ndarray]:
        return np.array(initials)
    else:
        raise Exception('Unknown data type for initials')


def runTrajectories(
        drift,
        diff,
        dt=.001,
        tmax=10.,
        reps=10,
        initials=20,
        silent=False):
    """Runs and returns several trajectories"""

    li = LI(drift, diff, dt=dt, tmax=tmax)
    initial_poss = get_initial_poss(initials)
    initials = len(initial_poss)

    # numpy array to store final result
    out = np.zeros((initials, reps, int(tmax/dt)))

    # simulate
    start = time.time()
    for i, init in enumerate(initial_poss):
        for r in range(reps):
            li.run(init)
            out[i, r, :] = li.out
    if not silent:
        print "took: %f" % (time.time() - start)

    return out


def runDistribution(
        drift,
        diff,
        dt=.001,
        tmax=10.,
        reps=10,
        initials=20,
        silent=False):
    """Runs and returns only final positions of trajectories"""

    li = LI(drift, diff, dt=dt, tmax=tmax)
    initial_poss = get_initial_poss(initials)
    initials = len(initial_poss)

    # numpy array to store final result
    out = np.zeros((initials, reps))

    # simulate
    start = time.time()
    for i in range(initials):
        for r in range(reps):
            li.run(initial_poss[i])
            out[i, r] = li.out[-1]
    if not silent:
        print "took: %f" % (time.time() - start)

    return out
