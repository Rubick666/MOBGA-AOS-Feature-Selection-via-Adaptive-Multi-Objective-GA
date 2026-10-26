import numpy as np

def shuffle_crossover(p1, p2):
    idx = np.random.permutation(len(p1))
    p1s, p2s = p1[idx], p2[idx]
    point = np.random.randint(1, len(p1))
    c1 = np.concatenate((p1s[:point], p2s[point:]))
    c2 = np.concatenate((p2s[:point], p1s[point:]))
    return c1, c2

def reduced_surrogate_crossover(p1, p2):
    diff = np.where(p1 != p2)[0]
    if len(diff) == 0:
        return p1.copy(), p2.copy()
    point = np.random.choice(diff)
    c1 = np.concatenate((p1[:point], p2[point:]))
    c2 = np.concatenate((p2[:point], p1[point:]))
    return c1, c2