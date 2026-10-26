# Benchmarks & Experimental Results

This document outlines the quantitative results and analysis of the MOBGA-AOS algorithm applied to 6 benchmark datasets for feature selection.

## Problem Overview
- **Dataset Pool:** DS02, DS04, DS05, DS07, DS08, DS10.
- **Objectives:** Minimize classification error (k-NN, k=3, 3-fold CV) and minimize the number of selected features.
- **Algorithm:** Multi-Objective Binary Genetic Algorithm with Adaptive Operator Selection (MOBGA-AOS).
- **Configuration:** Population = 50, Generations = 50, 3 independent runs per dataset.

## 1. Quantitative Results (IGD & HV)

The table below shows the mean and standard deviation of the IGD and Hypervolume (HV) metrics over 3 runs. *(Note: The report explicitly lists results for DS02 and DS04, which are recorded here).*

| Dataset | IGD (Mean ± Std) | HV (Mean ± Std) |
| :--- | :--- | :--- |
| **DS02** | 0.0133 ± 0.0187 | -1.5148 ± 0.3038 |
| **DS04** | 0.2650 ± 0.0711 | 13.2825 ± 13.6111 |
| **DS05** | *(See HV Plot)* | *(See HV Plot)* |
| **DS07** | *(See HV Plot)* | *(See HV Plot)* |
| **DS08** | *(See HV Plot)* | *(See HV Plot)* |
| **DS10** | *(See HV Plot)* | *(See HV Plot)* |

## 2. Pareto Front Analysis
- **DS02:** The Pareto front contains only a small number of solutions. This indicates the dataset offers limited meaningful trade-offs, and most feature subsets are dominated.
- **DS04:** The Pareto front shows a clear trade-off curve. As the number of features increases from 8 to 23, accuracy improves from ~0.81 to ~0.97.
- **DS05, DS07, DS08, DS10:** These datasets exhibit a more standard step-wise Pareto front, where significant feature reduction is possible with minimal loss of accuracy.
- **Data Issue Resolved:** The DS08 dataset had a target label column of `dtype=object`. The code forced a string transformation and then encoded it to integers to prevent k-NN evaluation errors.

## 3. Convergence for High-Dimensional Datasets (>250 features)
For datasets with a very large number of features (e.g., DS05, DS07, DS08, DS10), the search space is massive. To converge to optimal solutions more quickly, the following strategies are suggested:
1.  **Biasing Initial Populations:** Initialize the population with a higher probability of sparse solutions (fewer 1s) rather than purely random binaries. This immediately focuses the search on the feature-reduction objective.
2.  **Gradual Mutation Rate Reduction:** Start with a higher \(P_m\) to explore, then linearly decrease it over generations to exploit and fine-tune the Pareto front.
3.  **Increased Reward Impact:** Amplify the credit assignment (OSP update) to aggressively favor the crossover operator that is producing non-dominated offspring, reducing wasted evaluations on ineffective search strategies.