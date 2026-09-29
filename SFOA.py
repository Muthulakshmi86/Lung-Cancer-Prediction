import time

import numpy as np


# Starfish Optimization Algorithm (SFOA)
def SFOA(Xpos, fobj, lb, ub, Max_it):
    Npop, nD = Xpos.shape[0], Xpos.shape[1]
    GP = 0.5

    Curve = np.zeros(Max_it)
    Fitness = np.array([fobj(Xpos[i]) for i in range(Npop)])
    fvalbest = np.min(Fitness)
    xposbest = Xpos[np.argmin(Fitness)].copy()
    newX = np.zeros_like(Xpos)

    T = 0
    while T < Max_it:
        theta = np.pi / 2 * T / Max_it
        tEO = (Max_it - T) / Max_it * np.cos(theta)

        if np.random.rand() < GP:
            # Exploration
            for i in range(Npop):
                if nD > 5:
                    jp1 = np.random.permutation(nD)[:5]
                    for j in jp1:
                        pm = (2 * np.random.rand() - 1) * np.pi
                        if np.random.rand() < GP:
                            newX[i, j] = Xpos[i, j] + pm * (xposbest[j] - Xpos[i, j]) * np.cos(theta)
                        else:
                            newX[i, j] = Xpos[i, j] - pm * (xposbest[j] - Xpos[i, j]) * np.sin(theta)

                        if newX[i, j] > ub[j] or newX[i, j] < lb[j]:
                            newX[i, j] = Xpos[i, j]
                else:
                    jp2 = int(np.ceil(nD * np.random.rand())) - 1
                    im = np.random.permutation(Npop)
                    rand1 = 2 * np.random.rand() - 1
                    rand2 = 2 * np.random.rand() - 1
                    newX[i, jp2] = (tEO * Xpos[i, jp2] +
                                    rand1 * (Xpos[im[0], jp2] - Xpos[i, jp2]) +
                                    rand2 * (Xpos[im[1], jp2] - Xpos[i, jp2]))

                    if newX[i, jp2] > ub[jp2] or newX[i, jp2] < lb[jp2]:
                        newX[i, jp2] = Xpos[i, jp2]
                newX[i] = np.clip(newX[i], lb, ub)
        else:
            # Exploitation
            df = np.random.permutation(Npop)[:5]
            dm = np.array([xposbest - Xpos[df[i]] for i in range(5)])
            for i in range(Npop):
                r1, r2 = np.random.rand(2)
                kp = np.random.permutation(5)[:2]
                newX[i] = Xpos[i] + r1 * dm[kp[0]] + r2 * dm[kp[1]]

                if i == Npop - 1:
                    newX[i] = np.exp(-T * Npop / Max_it) * Xpos[i]
                newX[i] = np.clip(newX[i], lb, ub)

        # Evaluate new fitness
        for i in range(Npop):
            newFit = fobj(newX[i])
            if newFit < Fitness[i]:
                Fitness[i] = newFit
                Xpos[i] = newX[i].copy()
                if newFit < fvalbest:
                    fvalbest = newFit
                    xposbest = Xpos[i].copy()

        Curve[T] = fvalbest
        T += 1
    ct = time.time() - T

    return xposbest, Curve, fvalbest, ct
