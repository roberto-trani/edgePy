import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calcNormFactors(counts, lib_size=None, method="none", refColumn=None, logratioTrim=0.3, sumTrim=0.05, doWeighting=True, Acutoff=-1e10, p=0.75):
    """
    Scale normalization of RNA-Seq data, for count matrices.
    Original version in R by Mark Robinson, Gordon Smyth and edgeR team (2010).
    Porting from R to python by Roberto Trani.
    """
    # check counts
    if len(counts.shape) != 2:
        
    if np.any(np.isnan(counts)):
        raise ValueError("NA counts not permitted")
    nsamples = counts.shape[1]
    if not isinstance(counts, pd.DataFrame):
        counts = pd.DataFrame(counts)

    # check lib.size
    if lib_size is None:
        lib_size = np.sum(counts, axis=0)
    else:
        if np.any(np.isnan(lib_size)):
            raise ValueError("NA lib_sizes not permitted")
        if isinstance(lib_size, int):
            lib_size = np.full(nsamples, lib_size)
        elif len(lib_size) != nsamples:
            raise ValueError("calcNormFactors: len(lib_size) != nsamples")

    # check method
    method = method.lower()
    assert method in ["rle", "upperquartile", "tmm", "tmmwsp", "none"]
    
    # remove all zero rows
    allzero = np.sum(counts > 0, axis=1) == 0
    if np.any(allzero):
        counts = counts.iloc[~allzero,:]
    
    # degenerate cases
    if counts.shape[0] == 0 or nsamples == 1:
        method = "none"
    
    # calculate factors
    if method == "rle":
        f = _calcFactorRLE(counts) / lib_size
    elif method == "upperquartile":
        f = _calcFactorQuantile(counts, lib_size, p=p)
    elif method in ("tmm", "tmmwsp"):
        method_fun = _calcFactorTMM if method == "tmm" else _calcFactorTMMwsp
        f75 = _calcFactorQuantile(counts, lib_size, p=0.75)
        if refColumn is None:
            refColumn = np.argmin(np.abs(f75 - np.mean(f75)))
        elif isinstance(refColumn, int) and (refColumn < 0 or refColumn > nsamples):
            refColumn = 0
        f = np.full(nsamples, 0.0)
        for i in range(nsamples):
            f[i] = method_fun(
                counts.iloc[:,i], counts.iloc[:,refColumn], lib_size[i], lib_size[refColumn],
                logratioTrim=logratioTrim, sumTrim=sumTrim, doWeighting=doWeighting, Acutoff=Acutoff
            )
    elif method == "none":
        f = np.full(nsamples, 1.0)
    else:
        raise ValueError("method '{}' not implemented yet".format(method))
    
    # factors should multiple to one
    f = f / np.exp(np.mean(np.log(f)))
    
    # output
    return pd.DataFrame(
        f,
        index=counts.columns if isinstance(counts, pd.DataFrame) else None,
        columns=["factor"]
    )

def _calcFactorRLE(counts):
    """
    Scale factors as in Anders et al (2010).
    Porting from R to python by Roberto Trani.
    """
    gm = np.exp(np.mean(np.log(counts), axis=1))

    return np.apply_along_axis(
        (lambda col: np.median([col[i]/gm[i] for i in range(len(col)) if gm[i] > 0])),
        axis=0,
        arr=counts
    )

def _calcFactorQuantile(counts, lib_size, p=0.075):
    """
    Generalized version of upper-quartile normalization.
    Original version in R by Mark Robinson (2010).
    Porting from R to python by Roberto Trani.
    """
    y = counts / lib_size
    return np.percentile(y, q=p*100, axis=0)

def _calcFactorTMM(obs, ref, libsize_obs=None, libsize_ref=None, logratioTrim=0.3, sumTrim=0.05, doWeighting=True, Acutoff=-1e10):
    """
    TMM between two libraries.
    Original version in R by Mark Robinson (2010).
    Porting from R to python by Roberto Trani.
    """
    nO = obs.sum() if libsize_obs is None else libsize_obs
    nR = ref.sum() if libsize_ref is None else libsize_ref

    logR = np.log2((obs/nO)/(ref/nR))  # log ratio of expression, accounting for library size
    absE = np.log2(obs/nO) + np.log2(ref/nR) / 2  # absolute expression
    v = (nO-obs)/nO/obs + (nR-ref)/nR/ref   # estimated asymptotic variance
    
    # remove infinite values, cutoff based on A
    fin = np.isfinite(logR) & np.isfinite(absE) & (absE > Acutoff)

    logR = logR[fin]
    absE = absE[fin]
    v = v[fin]
    
    if np.max(np.abs(logR)) < 1e-6:
        return 1.0
    
    # taken from the original mean() function
    n = len(logR)
    loL = np.floor(n * logratioTrim)
    hiL = n - loL
    loS = np.floor(n * sumTrim)
    hiS = n - loS

    # keep = (rank(logR) %in% loL:hiL) & (rank(absE) %in% loS:hiS)
    # a fix from leonardo ivan almonacid cardenas, since rank() can return
    # non-integer values when there are a lot of ties
    rank_logR = scipy.stats.rankdata(logR, method='ordinal')
    rank_absE = scipy.stats.rankdata(absE, method='ordinal')
    keep = (rank_logR >= loL) & (rank_logR <= hiL) & (rank_absE >= loS) & (rank_absE <= hiS)

    # fix by Roberto for python
    keep = keep & (~np.isnan(logR)) & (~np.isnan(v))

    if doWeighting:
        f = np.sum(logR[keep] / v[keep]) / np.sum(1.0/v[keep])
    else:
        f = np.mean(logR[keep])

    if np.isnan(f):
        f = 0

    return np.exp2(f)

def _calcFactorTMMwsp(obs, ref, libsize_obs=None, libsize_ref=None, logratioTrim=0.3, sumTrim=0.05, doWeighting=True, Acutoff=-1e10):
    """
    TMM with pairing of singleton positive counts between the obs and ref libraries.
    Original version in by Gordon Smyth (2018).
    Porting from R to python by Roberto Trani.
    """
    
    # epsilon serves as floating-point zero
    eps = 1e-14
    
    # identify zero counts
    pos_obs = (obs > eps)
    pos_ref = (ref > eps)
    npos = 2 * pos_obs + pos_ref

    # remove double zeros and NAs
    mask = ~((npos == 0) | np.isnan(npos))
    if np.any(~mask):
        obs = obs[mask]
        ref = ref[mask]
        npos = npos[mask]

        # fix by Roberto
        libsize_obs = obs.sum()
        libsize_ref = ref.sum()

    # check library sizes
    libsize_obs = obs.sum() if libsize_obs is None else libsize_obs
    libsize_ref = ref.sum() if libsize_ref is None else libsize_ref

    # pair up as many singleton positives as possible
    # the unpaired singleton positives are discarded so that no zeros remain
    zero_obs = (npos == 1)
    zero_ref = (npos == 2)
    k = (zero_obs | zero_ref)
    n_eligible_singles = min(np.sum(zero_obs), np.sum(zero_ref))
    if n_eligible_singles > 0:
        obsk = -np.sort(-obs[k])[:n_eligible_singles]
        refk = -np.sort(-ref[k])[:n_eligible_singles]
        obs = np.concatenate([obs[~k], obsk])
        ref = np.concatenate([ref[~k], refk])
    else:
        obs = obs[~k]
        ref = ref[~k]

    # any left?
    n = len(obs)
    if n == 0:
        return 1.0

    # compute M and A values
    obs_p = obs / libsize_obs
    ref_p = ref / libsize_ref
    M = np.log2(obs_p / ref_p)
    A = 0.5 * np.log2(obs_p * ref_p)
    
    # if M all zero, return 1
    if np.max(np.abs(M)) < 1e-6:
        return 1.0
    
    # M order, breaking ties by shrunk M
    obs_p_shrunk = (obs + 0.5) / (libsize_obs + 0.5)
    ref_p_shrunk = (ref + 0.5) / (libsize_ref + 0.5)
    M_shrunk = np.log2(obs_p_shrunk / ref_p_shrunk)

    order_support = np.array(zip(M, M_shrunk), dtype=[("x", M.dtype), ("y", M_shrunk.dtype)])
    o_M = np.argsort(order_support, order=["x", "y"])
    
    # A order
    o_A = np.argsort(A)
    
    # trim
    loM = int(1.0 * n * logratioTrim)
    hiM = n - loM
    keep_M = np.full(n, False)
    keep_M[o_M[loM:hiM]] = True
    loA = int(n * sumTrim)
    hiA = n - loA
    keep_A = np.full(n, False)
    keep_A[o_A[loA:hiA]] = True
    keep = keep_M & keep_A
    M = M[keep]
    
    # average the M values
    if doWeighting:
        obs_p = obs_p[keep]
        ref_p = ref_p[keep]
        v = (1 - obs_p) / obs_p / libsize_obs + (1 - ref_p) / ref_p / libsize_ref
        w = (1 + 1e-6) / (v + 1e-6)
        TMM = np.sum(w * M) / np.sum(w)
    else:
        TMM = np.mean(M)
    
    return np.exp2(TMM)
