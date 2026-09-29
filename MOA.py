import time

import numpy as np


# Meerkat Optimization Algorithm
def MOA(meerkats, fobj, lb, ub, max_iterations):
    num_meerkats, num_dimensions = meerkats.shape[0], meerkats.shape[1]
    p_food = 0.8  # Probability of finding food
    p_prey = 0.7  # Probability of capturing prey
    p_escape = 0.1  # Probability of escaping from predators
    BestFit = float('inf')
    BestSol = np.zeros((num_dimensions, 1))
    fitness = fobj(meerkats[:])
    Convergence_curve = np.zeros((max_iterations, 1))

    t = 0
    ct = time.time()
    # Main loop (Meerkat Optimization Algorithm)
    for iteration in range(1, max_iterations + 1):
        # Evaluate fitness for each meerkat
        for i in range(num_meerkats):
            fitness[i] = fobj(meerkats[i])
        # Sort meerkats based on fitness
        sorted_indices = np.argsort(fitness)
        meerkats = meerkats[sorted_indices]
        fitness = fitness[sorted_indices]
        # Determine the leader (meerkat with the best fitness)
        leader = meerkats[0].copy()
        # Update meerkat positions
        for i in range(num_meerkats):
            if np.random.rand() < p_food:
                # Meerkat found food (exploitation)
                if np.random.rand() < p_prey:
                    # Meerkat caught the prey (exploitation)
                    meerkats[i] = leader + np.random.rand() * (meerkats[i] - leader)
                else:
                    # Meerkat didn't catch the prey (exploration)
                    meerkats[i] += np.random.randn(num_dimensions) * (ub - lb)
            else:
                # Meerkat didn't find food (exploration)
                if np.random.rand() < p_escape:
                    # Escaped from predators
                    meerkats[i] = leader + np.random.randn(num_dimensions) * (ub - lb)
                else:
                    # Moved to a new location
                    meerkats[i] += np.random.randn(num_dimensions) * (ub - lb)

            # Ensure positions stay within bounds [lb, ub]
            meerkats[i] = np.clip(meerkats[i], lb, ub)
            Convergence_curve[t] = BestFit
            t = t + 1
        BestFit = Convergence_curve[max_iterations - 1][0]
        ct = time.time() - ct

        return BestFit, Convergence_curve, BestSol, ct
