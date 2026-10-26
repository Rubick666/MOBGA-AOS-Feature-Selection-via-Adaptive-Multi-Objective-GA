import os
import time
import numpy as np
import matplotlib.pyplot as plt
from config import DATASETS, DATASET_DIR, NUM_RUNS, GENERATIONS, POP_SIZE, MUTATION_RATE
from data_utils import load_dataset
from nsga2_core import fast_non_dominated_sort
from metrics import compute_hv, compute_igd
from algorithm import mobga_aos

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    ensure_dir("results/plots")
    
    for ds in DATASETS:
        print(f"\n==============================")
        print(f"Processing Dataset {ds}")
        print(f"==============================")

        dataset_path = os.path.join(DATASET_DIR, ds + ".csv")
        if not os.path.exists(dataset_path):
            print(f"Dataset {ds} not found at {dataset_path}. Skipping...")
            continue

        X, y = load_dataset(dataset_path)

        pareto_runs = []
        hv_runs = []
        hv_curves = []

        for run in range(NUM_RUNS):
            print(f"  Run {run + 1}/{NUM_RUNS}")
            # Note: Passing config variables to algorithm
            _, objs, hv_proxy = mobga_aos(X, y, generations=GENERATIONS, pop_size=POP_SIZE, pm=MUTATION_RATE)

            idx = fast_non_dominated_sort(objs)[0]
            pareto = objs[idx]
            pareto_runs.append(pareto)

            ref = np.array([1.0, X.shape[1]])
            hv_runs.append(compute_hv(pareto, ref))
            hv_curves.append(hv_proxy)

        merged = np.vstack(pareto_runs)
        ref_idx = fast_non_dominated_sort(merged)[0]
        ref_pf = merged[ref_idx]

        igd_vals = [compute_igd(pf, ref_pf) for pf in pareto_runs]

        print(f"HV: {np.mean(hv_runs):.4f} ± {np.std(hv_runs):.4f}")
        print(f"IGD1: {np.mean(igd_vals):.4f} ± {np.std(igd_vals):.4f}")

        best = np.argmax(hv_runs)
        best_pf = pareto_runs[best]

        # Plot 1: Pareto Front
        plt.figure()
        plt.scatter(best_pf[:, 1], 1 - best_pf[:, 0])
        plt.xlabel("Number of selected features")
        plt.ylabel("Accuracy")
        plt.title(f"Pareto Front Dataset {ds}")
        plt.grid(True)
        plt.savefig(f"results/plots/pareto_{ds}.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Plot 2: HV over generations
        plt.figure()
        plt.plot(hv_curves[best])
        plt.xlabel("Generation")
        plt.ylabel("Hypervolume")
        plt.title(f"HV over generations Dataset {ds}")
        plt.grid(True)
        plt.savefig(f"results/plots/hv_{ds}.png", dpi=150, bbox_inches='tight')
        plt.close()

        time.sleep(1) # Reduced from 5s to 1s to speed up execution