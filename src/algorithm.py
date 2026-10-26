import numpy as np
from data_utils import knn_error
from nsga2_core import fast_non_dominated_sort, crowding_distance
from operators import shuffle_crossover, reduced_surrogate_crossover

def mobga_aos(X, y, generations=50, pop_size=50, pm=0.01):
    n_features = X.shape[1]
    population = np.random.randint(0, 2, (pop_size, n_features))

    OSP = np.array([0.5, 0.5])
    credits = np.zeros(2)

    hv_history = []

    for gen in range(generations):
        offspring = []
        offspring_ops = []

        for _ in range(pop_size // 2):
            p1, p2 = population[np.random.choice(pop_size, 2, replace=False)]
            op = np.random.choice([0, 1], p=OSP)

            if op == 0:
                c1, c2 = shuffle_crossover(p1, p2)
            else:
                c1, c2 = reduced_surrogate_crossover(p1, p2)

            for c in (c1, c2):
                mut = np.random.rand(n_features) < pm
                c[mut] = 1 - c[mut]
                offspring.append(c)
                offspring_ops.append(op)

        offspring = np.array(offspring)

        pop_all = np.vstack((population, offspring))
        objs = np.array([[knn_error(X, y, ind), np.sum(ind)] for ind in pop_all])

        fronts = fast_non_dominated_sort(objs)
        new_pop = []

        for front in fronts:
            if len(new_pop) + len(front) > pop_size:
                cd = crowding_distance(front, objs)
                idx = np.argsort(-cd)
                new_pop.extend(np.array(front)[idx][:pop_size - len(new_pop)])
                break
            new_pop.extend(front)

        population = pop_all[new_pop]

        # credit assignment
        first_front = set(fronts[0])
        for i, op in enumerate(offspring_ops):
            if pop_size + i in first_front:
                credits[op] += 1

        if credits.sum() > 0:
            OSP = credits / credits.sum()

        hv_history.append(len(fronts[0]))

    return population, objs, hv_history