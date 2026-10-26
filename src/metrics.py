import numpy as np

def compute_hv(pf, ref_point):
    hv = 0.0
    pf = pf[np.argsort(pf[:, 1])]
    prev_f2 = ref_point[1]
    for f1, f2 in pf:
        hv += (ref_point[0] - f1) * (prev_f2 - f2)
        prev_f2 = f2
    return hv

def compute_igd(pf, ref_pf):
    dists = []
    for r in ref_pf:
        d = np.linalg.norm(pf - r, axis=1)
        dists.append(np.min(d))
    return np.mean(dists)