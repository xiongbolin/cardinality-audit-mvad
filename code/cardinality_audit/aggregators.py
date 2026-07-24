from itertools import combinations
import numpy as np


def _topk(values, k):
    arr=np.asarray(values,dtype=float)
    if arr.size<=k:
        return float(arr.mean())
    return float(np.sort(arr)[-k:].mean())


def fixed_subsets(values, order=4, cap=128, seed=42):
    arr=np.asarray(values,dtype=float)
    if len(arr)<order:
        return [arr]
    combos=list(combinations(range(len(arr)), order))
    if len(combos)>cap:
        rng=np.random.default_rng(seed)
        combos=rng.choice(len(combos),size=cap,replace=False)
        combos=[list(combinations(range(len(arr)),order))[i] for i in combos]
    return [arr[list(c)] for c in combos]


def aggregate(values, method, tau=1.0, seed=42):
    arr=np.asarray(values,dtype=float)
    if method=='mean': return float(arr.mean())
    if method=='max': return float(arr.max())
    if method=='top2': return _topk(arr,2)
    if method=='top3': return _topk(arr,3)
    if method=='q75': return float(np.quantile(arr,.75))
    if method=='q90': return float(np.quantile(arr,.90))
    if method=='lse':
        m=arr.max(); return float(m+tau*np.log(np.exp((arr-m)/tau).sum()))
    if method=='logmeanexp':
        m=arr.max(); return float(m+tau*np.log(np.exp((arr-m)/tau).mean()))
    if method=='u2_max': return float(np.mean([x.max() for x in fixed_subsets(arr,2,128,seed)]))
    if method=='u3_top2': return float(np.mean([_topk(x,2) for x in fixed_subsets(arr,3,128,seed)]))
    if method=='u4_top3': return float(np.mean([_topk(x,3) for x in fixed_subsets(arr,4,128,seed)]))
    raise ValueError(method)


def registered_aggregators():
    return ['mean','max','top2','top3','q75','q90','lse','logmeanexp','u2_max','u3_top2','u4_top3']
