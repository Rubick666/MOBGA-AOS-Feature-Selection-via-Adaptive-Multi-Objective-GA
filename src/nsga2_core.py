import numpy as np

def dominates(a, b):
    return all(a <= b) and any(a < b)

def fast_non_dominated_sort(objs):
    S = [[] for _ in range(len(objs))]
    front = [[]]
    n = [0] * len(objs)

    for p in range(len(objs)):
        for q in range(len(objs)):
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                n[p] += 1
        if n[p] == 0:
            front[0].append(p)

    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    Q.append(q)
        i += 1
        front.append(Q)

    return front[:-1]

def crowding_distance(front, objs):
    distance = np.zeros(len(front))
    for m in range(objs.shape[1]):
        idx = np.argsort(objs[front, m])
        distance[idx[0]] = distance[idx[-1]] = np.inf
        f_min = objs[front, m].min()
        f_max = objs[front, m].max()
        if f_max == f_min:
            continue
        for i in range(1, len(front) - 1):
            distance[idx[i]] += (
                objs[front[idx[i + 1]], m] -
                objs[front[idx[i - 1]], m]
            ) / (f_max - f_min)
    return distance