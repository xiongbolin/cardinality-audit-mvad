import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def binary_metrics(y_true, scores):
    return {'auroc':float(roc_auc_score(y_true,scores)), 'auprc':float(average_precision_score(y_true,scores))}

def normal_size_dependence(sizes, scores):
    from scipy.stats import spearmanr
    return {'spearman_rho':float(spearmanr(sizes,scores).statistic)}

def ranking_reversal(a,b):
    ra=np.argsort(np.argsort(-np.asarray(a)))
    rb=np.argsort(np.argsort(-np.asarray(b)))
    return {'changed':int(np.sum(ra!=rb)), 'total':int(len(ra))}

def category_bootstrap(values, n=2000, seed=42):
    rng=np.random.default_rng(seed)
    arr=np.asarray(values)
    out=[]
    for _ in range(n):
        out.append(float(arr[rng.integers(0,len(arr),len(arr))].mean()))
    return np.asarray(out)
