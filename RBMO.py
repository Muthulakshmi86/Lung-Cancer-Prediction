import time

import numpy as np


def food_storage(fit, X, fit_old, X_old):
    # Boolean mask where previous fitness is better
    Inx = fit_old < fit
    # Update X: if fit_old is better, keep old X; else keep current X
    X = np.where(Inx[:, np.newaxis], X_old, X)
    # Update fitness: keep the better one
    fit = np.where(Inx, fit_old, fit)
    # Update historical records
    fit_old = fit.copy()
    X_old = X.copy()
    return fit, X, fit_old, X_old


# Red-billed Blue Magpie Optimizer (RBMO)
def RBMO(X, fobj, lb, ub, T):
    N, D = X.shape[0], X.shape[1]
    Xfood = np.zeros((D, 1))
    BestValue = float('inf')
    Conv = np.zeros(T)
    fitness = fobj(X[:])
    FES = 0
    Epsilon = 0.5
    X_old = X.copy()
    fitness_old = fobj(X[:])
    t = 0
    ct = time.time()
    while t < T:
        # Search for food (Exploration)
        for i in range(N):
            p = np.random.randint(2, 6)
            selected_index_p = np.random.choice(N, p, replace=False)
            Xp = X[selected_index_p]
            Xpmean = np.mean(Xp, axis=0)

            q = np.random.randint(10, N + 1)
            selected_index_q = np.random.choice(N, q, replace=False)
            Xq = X[selected_index_q]
            Xqmean = np.mean(Xq, axis=0)

            A = np.random.permutation(N)
            R1 = A[0]

            if np.random.rand() < Epsilon:
                X[i] += (Xpmean - X[R1]) * np.random.rand()
            else:
                X[i] += (Xqmean - X[R1]) * np.random.rand()

        for i in range(N):
            fitness[i] = fobj(X[:])
            if fitness[i] < BestValue:
                BestValue = fitness[i]
                Xfood = X[i].copy()
            FES += 1

        fitness, X, fitness_old, X_old = food_storage(fitness, X, fitness_old, X_old)

        CF = (1 - t / T) ** (2 * t / T)

        # Exploitation
        for i in range(N):
            p = np.random.randint(2, 6)
            selected_index_p = np.random.choice(N, p, replace=False)
            Xp = X[selected_index_p]
            Xpmean = np.mean(Xp, axis=0)

            q = np.random.randint(10, N + 1)
            selected_index_q = np.random.choice(N, q, replace=False)
            Xq = X[selected_index_q]
            Xqmean = np.mean(Xq, axis=0)

            if np.random.rand() < Epsilon:
                X[i] = Xfood + CF * (Xpmean - X[i]) * np.random.randn(D)
            else:
                X[i] = Xfood + CF * (Xqmean - X[i]) * np.random.randn(D)

        for i in range(N):
            fitness[i] = fobj(X[:])
            if fitness[i] < BestValue:
                BestValue = fitness[i]
                Xfood = X[i].copy()
            FES += 1

        fitness, X, fitness_old, X_old = food_storage(fitness, X, fitness_old, X_old)

        Conv[t] = BestValue
        t += 1
    ct = time.time() - ct
    return Xfood, Conv, BestValue, ct
