import math
import os
from glob import glob

import numba
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.stats import t

from .numerics import (
    _calc_mean_shrunken_residuals,
    _calc_mean_shrunken_residuals_and_mfracs,
    _count_n_cpg,
)
from .scan import _find_peaks
from .smooth import _load_smoothed_chrom
from .utils import _check_data_dir, _get_filepath, _load_chrom_mat, echo, secho

# ignore division by 0 and division by NaN error
np.seterr(divide="ignore", invalid="ignore")

np.random.seed(5)


def permuted_indices(idx_celltypes, celltype_1, celltype_2, total_cells):
    permutation = np.random.permutation(idx_celltypes)
    while (permutation == idx_celltypes).all():
        permutation = np.random.permutation(idx_celltypes)

    index_g1 = np.zeros(total_cells, dtype=bool)
    index_g2 = np.zeros(total_cells, dtype=bool)

    for i in permutation[celltype_1]:
        index_g1[i] = True
    for i in permutation[celltype_2]:
        index_g2[i] = True
    return index_g1, index_g2


@njit
def calc_fdr(bools):
    """
    Calculates an adjusted p-value for each DMR using the Benjamini-Hochberg method.
    Input is an array of boolean values, where each value represents a DMR.
    The array is sorted by absolute t-statistic and the boolean indicates whether the
    the DMR was found in a permutation (False) or in the real comparison (True).
    """
    n_t = bools.sum()  # number of DMRs in the real comparison
    n_t_null = len(bools) - n_t  # number of DMRs in the permutation
    fdisc = 0  # number of false discoveries up until a certain threshold
    tdisc = 0
    adj_p_vals = np.empty(bools.shape, dtype=np.float64)
    for i in range(len(bools)):
        if bools[i]:  # it's from the real comparison
            tdisc += 1
        else:  # it's from the permutation
            fdisc += 1
        if tdisc:
            adj_p_val = (fdisc / n_t_null) / (tdisc / n_t)
        else:
            # prevent div by 0 in the rare case that the best hit is from a permutation
            adj_p_val = 1
        adj_p_vals[i] = adj_p_val
    # iterate through the list from the back to make sure that there are no values
    # greater than one, and to make sure it's continuously increasing
    current_min = 1.0
    for i_rev in np.arange(1, len(bools))[::-1]:
        p = adj_p_vals[i_rev]
        if p > current_min:
            adj_p_vals[i_rev] = current_min
        elif p < current_min:
            current_min = p
    return adj_p_vals


@njit(nogil=True)
def calc_welch_tstat_df(group1, group2, min_cells):
    """
    Calculates the t-statistic and the degrees of freedom (df) according to Welch's
    t-test for unequal variances.
    Returns the t-statistic, df, and the two group sizes.
    """
    len_g1 = len(group1)
    len_g2 = len(group2)
    if len_g1 < min_cells:
        return np.nan, np.nan, len_g1, len_g2
    if len_g2 < min_cells:
        return np.nan, np.nan, len_g1, len_g2

    mean_g1 = np.mean(group1)
    mean_g2 = np.mean(group2)

    sum1 = 0.0
    sum2 = 0.0

    for value in group1:
        sqdif1 = (value - mean_g1) ** 2
        sum1 += sqdif1

    for value in group2:
        sqdif2 = (value - mean_g2) ** 2
        sum2 += sqdif2

    if sum1 == 0.0 and sum2 == 0.0:
        return np.nan, np.nan, len_g1, len_g2

    var_g1 = sum1 / (len_g1 - 1)
    var_g2 = sum2 / (len_g2 - 1)

    s_delta = math.sqrt(var_g1 / len_g1 + var_g2 / len_g2)
    t = (mean_g1 - mean_g2) / s_delta

    # calculate degrees of freedom
    numerator = ((var_g1 / len_g1) + (var_g2 / len_g2)) ** 2
    denominator = ((var_g1 / len_g1) ** 2 / (len_g1 - 1)) + (
        (var_g2 / len_g2) ** 2 / (len_g2 - 1)
    )
    df = numerator / denominator
    return t, df, len_g1, len_g2


@njit(nogil=True)
def calc_welch_tstat(group1, group2, min_cells):
    """
    Calculates the t-statistic according to Welch's t-test for unequal variances.
    Mostly copy-paste from calc_welch_tstat_df, this is ugly but gives a slight edge
    in performance.
    Returns the t-statistic only.
    """
    len_g1 = len(group1)
    if len_g1 < min_cells:
        return np.nan
    len_g2 = len(group2)
    if len_g2 < min_cells:
        return np.nan

    mean_g1 = np.mean(group1)
    mean_g2 = np.mean(group2)

    sum1 = 0.0
    sum2 = 0.0

    for value in group1:
        sqdif1 = (value - mean_g1) ** 2
        sum1 += sqdif1

    for value in group2:
        sqdif2 = (value - mean_g2) ** 2
        sum2 += sqdif2

    if sum1 == 0.0 and sum2 == 0.0:
        return np.nan

    var_g1 = sum1 / (len_g1 - 1)
    var_g2 = sum2 / (len_g2 - 1)

    s_delta = math.sqrt(var_g1 / len_g1 + var_g2 / len_g2)
    t = (mean_g1 - mean_g2) / s_delta
    return t


@njit(parallel=True)
def _move_windows(
    start,
    end,
    stepsize,
    half_bw,
    data_chrom,
    indices_chrom,
    indptr_chrom,
    smoothed_vals,
    n_cells,
    chrom_len,
    index,
    min_cells,
    datatype,
):
    """
    Move the sliding window along the whole chromosome.
    For each window, calculate the mean shrunken residuals,
    i.e. 1 methylation value per cell for that window. Then
    calculate the t-statistic of shrunken residuals. This is our
    measure of differential methylation for that window.
    """
    windows = np.arange(start, end, stepsize)
    t_array = np.empty(windows.shape, dtype=np.float64)

    for i in prange(windows.shape[0]):
        pos = windows[i]
        mean_shrunk_resid = _calc_mean_shrunken_residuals(
            data_chrom,
            indices_chrom,
            indptr_chrom,
            pos - half_bw,
            pos + half_bw,
            smoothed_vals,
            n_cells,
            chrom_len,
        )

        # the real indices are stored in the first "column".
        if datatype == "real":
            group1 = mean_shrunk_resid[index[0, 0]]
            group1 = group1[~np.isnan(group1)]

            group2 = mean_shrunk_resid[index[1, 0]]
            group2 = group2[~np.isnan(group2)]

        # A new permutation is used every 2Mbp.
        # Therefore, the next "column" of the array has to be accessed.
        # The column index (chrom_bin)
        # equals the window position on the chromosome separated by 2Mbp.
        else:
            chrom_bin = pos // 2000000 + 1
            group1 = mean_shrunk_resid[index[0, chrom_bin]]
            group1 = group1[~np.isnan(group1)]

            group2 = mean_shrunk_resid[index[1, chrom_bin]]
            group2 = group2[~np.isnan(group2)]

        t = calc_welch_tstat(group1, group2, min_cells)
        t_array[i] = t

    return windows, t_array


def calc_tstat_peaks(
    chrom,
    datatype,
    group_names,
    half_bw,
    data_chrom,
    indices_chrom,
    indptr_chrom,
    smoothed_vals,
    n_cells,
    chrom_len,
    index,
    min_cells,
    bridge_gaps,
    threshold_datatype,
    window_tstat_groups,
    genomic_positions,
):
    output = []
    for _ in range(13):  # number of columns in the final output table
        output.append(np.empty(0))

    for tstat_windows, group_name, threshold_value in zip(
        window_tstat_groups, group_names, threshold_datatype
    ):
        # merge overlapping windows with lowest and highest t-statistic,
        # to get bigger regions of variable size
        peak_starts, peak_ends = _find_peaks(
            tstat_windows, genomic_positions, threshold_value, half_bw, bridge_gaps
        )

        # for each big merged peak, re-calculate the t-statistic
        for ps, pe in zip(peak_starts, peak_ends):
            # how many CpGs does the VMR contain?
            region_indptr = indptr_chrom[ps : pe + 2] - indptr_chrom[ps]
            n_cpg = _count_n_cpg(region_indptr)

            mean_shrunk_resid, mfracs = _calc_mean_shrunken_residuals_and_mfracs(
                data_chrom,
                indices_chrom,
                indptr_chrom,
                ps,
                pe,
                smoothed_vals,
                n_cells,
                chrom_len,
            )
            # the real indices are stored in the first "column".
            if datatype == "real":
                group1 = mean_shrunk_resid[index[0, 0]]
                group1 = group1[~np.isnan(group1)]

                group2 = mean_shrunk_resid[index[1, 0]]
                group2 = group2[~np.isnan(group2)]

                g1_mfrac = np.nanmean(mfracs[index[0, 0]])
                g2_mfrac = np.nanmean(mfracs[index[1, 0]])

            # A new permutation is used every 2Mbp.
            # Therefore, the next "column" of the array has to be accessed.
            # The column index (chrom_bin)
            # equals the middle of the merged peak separated by 2Mbp.
            else:
                chrom_bin = int(((pe - ps) / 2 + ps) // 2000000 + 1)
                group1 = mean_shrunk_resid[index[0, chrom_bin]]
                group1 = group1[~np.isnan(group1)]

                group2 = mean_shrunk_resid[index[1, chrom_bin]]
                group2 = group2[~np.isnan(group2)]

                g1_mfrac = np.nanmean(mfracs[index[0, chrom_bin]])
                g2_mfrac = np.nanmean(mfracs[index[1, chrom_bin]])

            t_stat, df, n_cells_g1, n_cells_g2 = calc_welch_tstat_df(
                group1, group2, min_cells
            )
            p = 2 * (1 - t.cdf(x=abs(t_stat), df=df))  # raw two-tailed p-value

            # output columns (adjusted p-value appended later:)
            datapoints = [
                chrom,  # chromosome
                ps,  # start position
                pe,  # end position
                t_stat,  # t-statistic
                n_cpg,  # total number of methylation sites in region
                n_cells_g1,  # number of group1-cells with coverage
                n_cells_g2,  # number of group2-cells with coverage
                g1_mfrac,  # methylation fraction of cells in group 1
                g2_mfrac,  # methylation fraction of cells in group 2
                group_name,  # label of the group with lower methylation
                df,  # degrees of freedom according to Welch-Satterthwaite equation
                datatype,  # was the DMR found in a permutation or not?
                p,  # raw p-value
            ]

            for datapoint in range(len(datapoints)):
                output[datapoint] = np.append(output[datapoint], datapoints[datapoint])
    return output


def parse_cell_groups(csv_path, data_dir):
    """
    Parses the user-specified csv file that denotes the two groups of cells that
    should be compared with methscan diff. Also checks that this file is valid.
    Returns an array with the group labels and a list of the two group names.
    """
    cellname_path = os.path.join(data_dir, "column_header.txt")
    cell_order = pd.read_csv(cellname_path, dtype="str", header=None, names=["cell"])
    n_cells_total = len(cell_order)
    group_df = pd.read_csv(
        csv_path,
        dtype=str,
        delimiter=",",
        header=None,
        names=["cell", "group"],
        index_col=0,
    )
    if len(group_df) > n_cells_total:
        raise Exception(
            f"The data set stored in {data_dir} comprises only {n_cells_total} "
            f"cells, but {csv_path} contains group labels for {len(group_df)} cells."
        )
    extra_cells = set(group_df.index) - set(cell_order.cell)
    if extra_cells:
        raise Exception(
            f"One or more cells that you specified in {_get_filepath(csv_path)} are "
            f"not present in {cellname_path}. These are the cells that were not "
            f"found: '{', '.join(extra_cells)}'."
        )
    group_df = group_df.reindex(cell_order["cell"])
    group_df["group"] = group_df["group"].fillna("-")
    groups = set(group_df["group"])
    groups -= {"-", "‒", "–", "—", "―"}
    groups = sorted(list(groups))
    group_arr = np.array(group_df["group"])
    if len(groups) != 2:
        raise Exception(
            f"{csv_path} specifies {len(groups)} cell groups "
            f"({', '.join(groups)}), but you need to specify exactly 2!"
        )
    n_g1 = (group_arr == groups[0]).sum()
    n_g2 = (group_arr == groups[1]).sum()
    secho(
        f"Scanning the genome for differentially methylated regions between {n_g1} "
        f"cells of group '{groups[0]}' and {n_g2} cells of group '{groups[1]}'.",
        fg="green",
    )
    n_nogroup = n_cells_total - n_g1 - n_g2
    if n_nogroup:
        secho(
            f"{n_nogroup} cells were not assigned to any group and will be ignored.",
            fg="green",
        )
    echo()
    return group_arr, groups


def diff(
    data_dir,
    cell_groups,
    output,
    bandwidth,
    stepsize,
    threshold,
    min_cells,
    bridge_gaps,
    threads=-1,
    write_header=False,
    debug=False,
):
    _check_data_dir(data_dir, assert_smoothed=True)
    if threads != -1:
        numba.set_num_threads(threads)
    n_threads = numba.get_num_threads()
    half_bw = bandwidth // 2

    celltypes, group_names = parse_cell_groups(cell_groups, data_dir)

    index_realg1 = (celltypes == group_names[0]).flatten()
    index_realg2 = (celltypes == group_names[1]).flatten()

    # needs to be calculated for the permutation
    labeled_cells = (celltypes == group_names[0]) | (celltypes == group_names[1])
    idx_celltypes = np.asarray(np.where(labeled_cells)).flatten()
    celltype_1 = celltypes[labeled_cells] == group_names[0]
    celltype_2 = celltypes[labeled_cells] == group_names[1]
    total_cells = len(celltypes)

    # sort chroms by file size. We start with largest chrom to find the threshold
    chrom_paths = sorted(
        glob(os.path.join(data_dir, "*.npz")),
        key=lambda x: os.path.getsize(x),
        reverse=True,
    )

    threshold_values = np.zeros([2, 2])
    output_final = []

    for mat_path in chrom_paths:
        chrom = os.path.basename(os.path.splitext(mat_path)[0])
        mat = _load_chrom_mat(data_dir, chrom)
        smoothed_cpg_vals = _load_smoothed_chrom(data_dir, chrom)
        chrom_len, n_cells = mat.shape
        cpg_pos_chrom = np.nonzero(mat.getnnz(axis=1))[0]

        if n_threads > 1:
            echo(f"Scanning chromosome {chrom} using {n_threads} parallel threads ...")
        else:
            echo(f"Scanning chromosome {chrom} ...")

        # Permute data every 2Mbp.
        # Store necessary number of permutations for
        # the according chromosome length in an array.
        # While scanning the chromosome every 2 Mbp another permutation
        # ("column" of the array) can be used.
        n_perm = chrom_len // 2000000 + 1
        perm_idx = np.empty([2, n_perm + 1, len(celltypes)], dtype=bool)
        # first "column" holds the real perm_idx of both cell types
        perm_idx[0][0] = index_realg1
        perm_idx[1][0] = index_realg2
        for i in range(n_perm):
            perm_idx[0][i + 1], perm_idx[1][i + 1] = permuted_indices(
                idx_celltypes, celltype_1, celltype_2, total_cells
            )

        output_chrom = []
        datatypes = ["real", "permuted"]
        for datatype in datatypes:
            start = cpg_pos_chrom[0] + half_bw + 1
            end = cpg_pos_chrom[-1] - half_bw - 1
            genomic_positions, window_tstat = _move_windows(
                start,
                end,
                stepsize,
                half_bw,
                mat.data,
                mat.indices,
                mat.indptr,
                smoothed_cpg_vals,
                n_cells,
                chrom_len,
                perm_idx,
                min_cells,
                datatype,
            )

            # calculate t-statistic for group1 (negative t-values)
            # and group 2 (positive t-values)
            # to make this possible the t-stat is multiplied by -1 for group1
            window_tstat_groups = [window_tstat * -1, window_tstat]

            # This is the first=biggest chrom,
            # so let's find our t-statistic threshold here.
            # The t-statistic threshold will be determined
            # based on the largest chromosome.
            # By default, we take the 98th percentile of all window t-statistics.
            # Thresholds for real indices are stored in the first
            # and for permuted indices in the second array.
            iteration = 0
            if threshold_values[1, 1] == 0:
                for tstat_windows, group_name in zip(window_tstat_groups, group_names):
                    if datatype == "real":
                        threshold_values[0, iteration] = np.nanquantile(
                            tstat_windows, 1 - threshold
                        )
                        echo(
                            f"Determined t-value threshold of "
                            f"{threshold_values[0,iteration]:.3f} for the "
                            f"'{group_name}' cell group."
                        )
                    if datatype == "permuted":
                        threshold_values[1, iteration] = np.nanquantile(
                            tstat_windows, 1 - threshold
                        )

                        if debug:
                            echo(
                                f"Determined t-value threshold of "
                                f"{threshold_values[1,iteration]:.3f} "
                                f"for permutation {iteration} ({group_name})."
                            )
                    iteration += 1

            if datatype == "real":
                threshold_datatype = threshold_values[0]

            if datatype == "permuted":
                threshold_datatype = threshold_values[1]

            # calculate t-statistic for merged peaks
            output_iteration = calc_tstat_peaks(
                chrom,
                datatype,
                group_names,
                half_bw,
                mat.data,
                mat.indices,
                mat.indptr,
                smoothed_cpg_vals,
                n_cells,
                chrom_len,
                perm_idx,
                min_cells,
                bridge_gaps,
                threshold_datatype,
                window_tstat_groups,
                genomic_positions,
            )

            # join outputs of all loops
            if len(output_chrom) == 0:
                output_chrom = output_iteration
            else:
                for i in range(len(output_chrom)):
                    output_chrom[i] = np.append(output_chrom[i], output_iteration[i])

        # join outputs of individual chromosomes
        if len(output_final) == 0:
            output_final = output_chrom
        else:
            for column in range(len(output_final)):
                output_final[column] = np.append(
                    output_final[column], output_chrom[column]
                )

    # sort descending by absolute t-statistic
    idx = np.argsort(-np.absolute(output_final[3]))
    for column in range(len(output_final)):
        output_final[column] = output_final[column][idx]

    for int_col in (1, 2, 4, 5, 6):
        output_final[int_col] = output_final[int_col].astype("int")

    # calculate FDR / adjusted p-values
    adj_p_val = calc_fdr(output_final[11] == "real")
    output_final.append(adj_p_val)

    if not debug:
        # remove DMRs from permutations, and remove "df" and "datatype" column
        is_from_permutation = output_final[11] == "real"
        for column in range(len(output_final)):
            output_final[column] = output_final[column][is_from_permutation]
        del output_final[11]  # datatype (labels DMRs from permutation)
        del output_final[10]  # degrees of freedom

        n_sig = np.count_nonzero(output_final[-1] < 0.05)
        secho(
            f"\nFound {n_sig} differentially methylated regions "
            "with an adjusted p-value below 0.05.\n",
            fg="green" if n_sig else "red",
        )
    echo(f"Writing DMRs to {_get_filepath(output)}")

    header = ""
    if write_header:
        header = (
            "chromosome\tDMR_start\tDMR_end\tt_statistic\tn_sites\t"
            "n_cells_group1\tn_cells_group2\tmeth_frac_group1\tmeth_frac_group2\t"
            "low_group_label\tdegrees_of_freedom\tsource\tp\tadjusted_p"
        )
        if not debug:
            header = header.replace("degrees_of_freedom\tsource\t", "")

    np.savetxt(
        output, np.transpose(output_final), delimiter="\t", fmt="%s", header=header
    )
