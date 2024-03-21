import numpy as np
from numba import njit


@njit(nogil=True)
def _calc_mean_shrunken_residuals(
    data_chrom,
    indices_chrom,
    indptr_chrom,
    start,
    end,
    smoothed_vals,
    n_cells,
    chrom_len,
    shrinkage_factor=1,
):
    shrunken_resid = np.full(n_cells, np.nan)
    if start > chrom_len:
        return shrunken_resid
    end += 1
    if end > chrom_len:
        end = chrom_len
    # slice the methylation values so that we only keep the values in the window
    data = data_chrom[indptr_chrom[start] : indptr_chrom[end]]
    if data.size == 0:
        # return NaN for regions without coverage or regions without CpGs
        return shrunken_resid
    # slice indices
    indices = indices_chrom[indptr_chrom[start] : indptr_chrom[end]]
    # slice index pointer
    indptr = indptr_chrom[start : end + 1] - indptr_chrom[start]
    indptr_diff = np.diff(indptr)

    n_obs = np.zeros(n_cells, dtype=np.int64)
    n_obs_start = np.bincount(indices)
    n_obs[0 : n_obs_start.shape[0]] = n_obs_start

    meth_sums = np.zeros(n_cells, dtype=np.int64)
    smooth_sums = np.zeros(n_cells, dtype=np.float64)
    cpg_idx = 0
    nobs_cpg = indptr_diff[cpg_idx]
    # nobs_cpg: how many of the next values correspond to the same CpG
    # e.g. a value of 3 means that the next 3 values are of the same CpG
    for i in range(data.shape[0]):
        while nobs_cpg == 0:
            cpg_idx += 1
            nobs_cpg = indptr_diff[cpg_idx]
        nobs_cpg -= 1
        cell_idx = indices[i]
        smooth_sums[cell_idx] += smoothed_vals[start + cpg_idx]
        meth_value = data[i]
        if meth_value == -1:
            continue  # skip 0 meth values when summing
        meth_sums[cell_idx] += 1

    for i in range(n_cells):
        if n_obs[i] > 0:
            shrunken_resid[i] = (meth_sums[i] - smooth_sums[i]) / (
                n_obs[i] + shrinkage_factor
            )
    return shrunken_resid


@njit(nogil=True)
def _calc_mean_shrunken_residuals_and_mfracs(
    data_chrom,
    indices_chrom,
    indptr_chrom,
    start,
    end,
    smoothed_vals,
    n_cells,
    chrom_len,
    shrinkage_factor=1,
):
    """
    A copy of _calc_mean_shrunken_residuals() that also returns the average methylation
    of both cell groups. The function exists twice for performance reasons.
    """
    shrunken_resid = np.full(n_cells, np.nan)
    if start > chrom_len:
        return shrunken_resid, np.full(n_cells, np.nan)
    end += 1
    if end > chrom_len:
        end = chrom_len
    # slice the methylation values so that we only keep the values in the window
    data = data_chrom[indptr_chrom[start] : indptr_chrom[end]]
    if data.size == 0:
        # return NaN for regions without coverage or regions without CpGs
        return shrunken_resid, np.full(n_cells, np.nan)
    # slice indices
    indices = indices_chrom[indptr_chrom[start] : indptr_chrom[end]]
    # slice index pointer
    indptr = indptr_chrom[start : end + 1] - indptr_chrom[start]
    indptr_diff = np.diff(indptr)

    n_obs = np.zeros(n_cells, dtype=np.int64)
    n_obs_start = np.bincount(indices)
    n_obs[0 : n_obs_start.shape[0]] = n_obs_start

    meth_sums = np.zeros(n_cells, dtype=np.int64)
    n_total = np.zeros(n_cells, dtype=np.int64)
    smooth_sums = np.zeros(n_cells, dtype=np.float64)
    cpg_idx = 0
    nobs_cpg = indptr_diff[cpg_idx]
    # nobs_cpg: how many of the next values correspond to the same CpG
    # e.g. a value of 3 means that the next 3 values are of the same CpG
    for i in range(data.shape[0]):
        while nobs_cpg == 0:
            cpg_idx += 1
            nobs_cpg = indptr_diff[cpg_idx]
        nobs_cpg -= 1
        cell_idx = indices[i]
        smooth_sums[cell_idx] += smoothed_vals[start + cpg_idx]
        meth_value = data[i]
        n_total[cell_idx] += 1
        if meth_value == -1:
            continue  # skip 0 meth values when summing
        meth_sums[cell_idx] += 1

    for i in range(n_cells):
        if n_obs[i] > 0:
            shrunken_resid[i] = (meth_sums[i] - smooth_sums[i]) / (
                n_obs[i] + shrinkage_factor
            )
    return shrunken_resid, np.divide(meth_sums, n_total)


@njit
def _calc_region_stats(
    data_chrom, indices_chrom, indptr_chrom, start, end, n_cells, chrom_len
):
    n_meth = np.zeros(n_cells, dtype=np.int64)
    n_total = np.zeros(n_cells, dtype=np.int64)
    if start > chrom_len:
        n_obs_cpg = 0
    else:
        end += 1
        if end > chrom_len:
            end = chrom_len
        # slice the methylation values so that we only keep the values in the window
        data = data_chrom[indptr_chrom[start] : indptr_chrom[end]]
        if data.size > 0:
            # slice indices
            indices = indices_chrom[indptr_chrom[start] : indptr_chrom[end]]
            # slice index pointer
            indptr = indptr_chrom[start : end + 1] - indptr_chrom[start]
            n_obs_cpg = _count_n_cpg(indptr)  # total number of CpGs in the region
            for i in range(data.shape[0]):
                cell_i = indices[i]
                meth_value = data[i]
                n_total[cell_i] += 1
                if meth_value == -1:
                    continue
                n_meth[cell_i] += 1
    return n_meth, n_total, np.divide(n_meth, n_total), n_obs_cpg


@njit
def _count_n_cpg(region_indptr):
    """
    Count the total number of CpGs in a region, based on CSR matrix index pointers.
    """
    prev_val = 0
    n_cpg = 0
    for val in region_indptr:
        if val != prev_val:
            n_cpg += 1
            prev_val = val
    return n_cpg


@njit
def _count_n_cells(region_indices):
    """
    Count the total number of cells that have sequencing coverage in a region,
    based on CSR matrix indices.
    """
    return np.unique(region_indices).size
