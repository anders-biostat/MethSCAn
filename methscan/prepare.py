import gzip
import os
import sys
from datetime import datetime, timedelta
from glob import glob

import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
from numba import njit

from . import __version__
from .utils import echo, secho


def prepare(input_files, data_dir, input_format, round_sites, chunksize):
    input_files = sorted(input_files)  # ensure output mtx is ordered consistently
    begin_time = datetime.now()  # to log runtime
    cell_names = _get_cell_names(input_files)
    n_cells = len(cell_names)
    os.makedirs(data_dir, exist_ok=True)
    # we use this opportunity to count some basic summary stats
    n_obs_cell = np.zeros(n_cells, dtype=np.int64)
    n_meth_cell = np.zeros(n_cells, dtype=np.int64)

    # For each chromosome, we first make a sparse matrix in COO (coordinate)
    # format, because COO can be constructed value by value, without knowing the
    # dimensions beforehand. This means we can construct it cell by cell.
    # We dump the COO to hard disk to save RAM and then later convert each COO to a
    # more efficient format (CSR).
    echo(f"Processing {n_cells} methylation files...")
    chrom_sizes = _dump_coo_files(
        input_files, input_format, n_cells, data_dir, round_sites, chunksize
    )
    echo(
        "\nStoring methylation data in 'compressed "
        "sparse row' (CSR) matrix format for future use."
    )

    # read each COO file and convert the matrix to CSR format.
    for chrom, chrom_size in chrom_sizes.items():
        # create empty matrix
        echo(f"Populating {chrom_size} x {n_cells} matrix for chromosome {chrom}...")
        # populate with values from temporary COO file
        mat_path = os.path.join(data_dir, f"{chrom}.npz")
        mat = _load_csr_from_coo(data_dir, chrom, chrom_size, n_cells)
        n_obs_cell += mat.getnnz(axis=0)
        n_meth_cell += np.ravel(np.sum(mat > 0, axis=0))

        echo(f"Writing to {mat_path} ...")
        sp_sparse.save_npz(mat_path, mat)
        _delete_coo_chunks(data_dir, chrom)  # delete temporary .coo file

    colname_path = _write_column_names(data_dir, cell_names)
    echo(f"\nWrote cell names to {colname_path}")
    stats_path = _write_summary_stats(data_dir, cell_names, n_obs_cell, n_meth_cell)
    echo(f"Wrote summary stats for each cell to {stats_path}")
    _write_run_info(
        os.path.join(data_dir, "run_info.txt"),
        begin_time,
        data_dir=data_dir,
        input_format=input_format,
        round_sites=round_sites,
        chunksize=chunksize,
        input_files=input_files,
    )
    secho(
        f"\nSuccessfully stored methylation data for {n_cells} cells "
        f"with {len(chrom_sizes.keys())} chromosomes.",
        fg="green",
    )


def _write_run_info(fpath, begin_time, **kwargs):
    """On prepare, write methscan version, run date and parameters to a logfile"""
    now = datetime.now()
    runtime = timedelta(seconds=(now - begin_time).seconds)
    with open(fpath, "w") as log:
        log.write(
            "This directory was generated "
            f"on {now.strftime('%a %b %d %H:%M:%S %Y')}\n"
            f"with methscan prepare version {__version__}.\n"
            f"The total runtime was {runtime} (hour:min:s).\n"
            "\nThe following parameters were used:"
        )
        for arg, value in kwargs.items():
            log.write(f"\n\n{arg}:\n")
            if isinstance(value, (list, tuple, set)):
                log.write("\n".join(str(v) for v in value))
            else:
                log.write(str(value))


def _get_cell_names(cov_files):
    """Use the file base names (without extension) as cell names."""
    names = []
    for file_path in cov_files:
        name = os.path.basename(file_path)
        if name.lower().endswith(".gz"):
            # remove .gz
            name = name[:-3]
        # remove .xxx
        names.append(os.path.splitext(name)[0])
    if len(set(names)) < len(names):
        err_msg = (
            "\n".join(names) + "\nThese sample names are not unique, "
            "check your file names again!"
        )
        raise Exception(err_msg)
    return names


def _dump_coo_files(fpaths, input_format, n_cells, output_dir, round_sites, chunksize):
    try:
        c_col, p_col, m_col, u_col, coverage, onlyrel, sep, header = _human_to_computer(
            input_format
        )
    except Exception as exc:
        raise type(exc)(
            f"{exc}\n\nUnknown input file format '{input_format}'.\nValid options "
            "include 'bismark', 'allc', 'methylpy' or a custom ':'-separated format "
            "(check 'methscan prepare --help' for details)."
        ).with_traceback(sys.exc_info()[2])

    coo_files = {}
    chrom_sizes = {}
    for cell_n, cov_file in enumerate(fpaths):
        if cell_n % 50 == 0:
            echo("{0:.2f}% done...".format(100 * cell_n / n_cells))
        for line_vals in _iterate_covfile(
            cov_file, c_col, p_col, m_col, u_col, coverage, onlyrel, sep, header
        ):
            chrom, genomic_pos, n_meth, n_unmeth = line_vals
            # to make the individual coo files smaller, we split each chromosome
            # into chunks (of size 10 Mbp by default)
            chrom_chunk = int(genomic_pos // chunksize)
            coo_tuple = (chrom, chrom_chunk)  # one coo file per chromosome chunk

            # deal with unexpected CpG sites that are not 1 or 0. These could be
            # caused by mapping artifacts, strand/allele-specific methylation, etc:
            if n_meth != 0 and n_unmeth != 0:
                if round_sites:
                    if n_meth == n_unmeth:
                        continue  # if it's 50:50, ignore this CpG
                    meth_value = 1 if n_meth > n_unmeth else -1
                else:
                    continue  # ignore all CpGs that are not "clear"!
            else:
                meth_value = 1 if n_meth > 0 else -1

            if coo_tuple not in coo_files:
                coo_path = os.path.join(
                    output_dir, f"{chrom}_chunk{chrom_chunk:07}.coo"
                )
                coo_files[coo_tuple] = open(coo_path, "w")
                if chrom not in chrom_sizes:
                    chrom_sizes[chrom] = 0
            if genomic_pos > chrom_sizes[chrom]:
                chrom_sizes[chrom] = genomic_pos
            coo_files[coo_tuple].write(f"{genomic_pos},{cell_n},{meth_value}\n")
    for fhandle in coo_files.values():
        # maybe somehow use try/finally or "with" to make sure
        # they're closed even when crashing
        fhandle.close()
    echo("100% done.")
    return chrom_sizes


def _iter_chunks(data_dir, chrom):
    chunk_paths = glob(os.path.join(data_dir, f"{chrom}_chunk*.coo"))
    for chunk_path in sorted(chunk_paths):
        echo(f"processing chunk {os.path.basename(chunk_path)} ...")
        chunk = pd.read_csv(chunk_path, delimiter=",", header=None).values
        yield chunk


def _delete_coo_chunks(data_dir, chrom):
    chunk_paths = glob(os.path.join(data_dir, f"{chrom}_chunk*.coo"))
    for chunk_path in chunk_paths:
        os.remove(chunk_path)  # delete temporary .coo file


@njit
def _process_chunk(positions, indptr, last_pos, indptr_counter, indptr_i):
    for pos in positions:
        if pos > last_pos:
            for __ in range(pos - last_pos):
                indptr[indptr_i] = indptr_counter
                indptr_i += 1
            last_pos = pos
        indptr_counter += 1
    return last_pos, indptr_counter, indptr_i


def _load_csr_from_coo(data_dir, chrom, chrom_size, n_cells):
    data_chunks = []
    indices_chunks = []
    # indptr size is n_rows + 1
    indptr = np.empty(chrom_size + 2, dtype=np.int64)

    last_pos = -1
    indptr_counter = 0
    indptr_i = 0
    for chunk in _iter_chunks(data_dir, chrom):
        # sort the chunk to facilitate COO to CSR conversion
        sorting_idx = np.lexsort((chunk[:, 1], chunk[:, 0]))
        last_pos, indptr_counter, indptr_i = _process_chunk(
            chunk[sorting_idx, 0], indptr, last_pos, indptr_counter, indptr_i
        )
        # in a sorted COO file, two of the columns are identical to the
        # data in indices vectors of the CSR format
        data_chunks.append(chunk[sorting_idx, 2].astype(np.int8))
        indices_chunks.append(chunk[sorting_idx, 1].astype(np.int64))
    indptr[indptr_i] = indptr_counter  # last missing index pointer value
    data = np.concatenate(data_chunks)
    indices = np.concatenate(indices_chunks)
    mat = sp_sparse.csr_matrix((data, indices, indptr), shape=(chrom_size + 1, n_cells))
    return mat


def _write_column_names(output_dir, cell_names, fname="column_header.txt"):
    """
    The column names (usually cell names) will be
    written to a separate text file
    """
    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w") as col_head:
        for cell_name in cell_names:
            col_head.write(cell_name + "\n")
    return out_path


def _human_to_computer(file_format):
    """Convert the human-readable input file format to a tuple."""
    if ":" in file_format:
        return create_custom_format(file_format).totuple()
    return _create_standard_format(file_format).totuple()


def _write_summary_stats(data_dir, cell_names, n_obs, n_meth):
    stats_df = pd.DataFrame(
        {
            "cell_name": cell_names,
            "n_obs": n_obs,
            "n_meth": n_meth,
            "global_meth_frac": np.divide(n_meth, n_obs),
        }
    )
    out_path = os.path.join(data_dir, "cell_stats.csv")
    with open(out_path, "w") as outfile:
        outfile.write(stats_df.to_csv(index=False))
    return out_path


class CoverageFormat:
    """Describes the columns in the coverage file."""

    def __init__(self, chrom, pos, meth, umeth, coverage, onlyrel, sep, header):
        self.chrom = chrom
        self.pos = pos
        self.meth = meth
        self.umeth = umeth
        self.cov = coverage
        self.onlyrel = onlyrel
        self.sep = sep
        self.header = header

    def totuple(self):
        """Transform to use it in non-refactored code for now."""
        return (
            self.chrom,
            self.pos,
            self.meth,
            self.umeth,
            self.cov,
            self.onlyrel,
            self.sep,
            self.header,
        )


def create_custom_format(format_string):
    """Create from user specified string."""
    format_string = format_string.lower().split(":")
    if len(format_string) != 6:
        raise Exception("Invalid number of ':'-separated values in custom input format")
    chrom = int(format_string[0]) - 1
    pos = int(format_string[1]) - 1
    meth = int(format_string[2]) - 1
    info = format_string[3][-1]
    if info == "c":
        coverage = True
    elif info == "u":
        coverage = False
    else:
        raise Exception(
            "The 4th column of a custom input format must contain an integer and "
            "either 'c' for coverage or 'u' for unmethylated counts (e.g. '4c'), "
            f"but you provided '{format_string[3]}'."
        )
    umeth = int(format_string[3][0:-1]) - 1
    sep = str(format_string[4])
    if sep in ("\\t", "TAB", "tab", "T", "t"):
        sep = "\t"
    header = bool(int(format_string[5]))
    return CoverageFormat(chrom, pos, meth, umeth, coverage, False, sep, header)


def _create_standard_format(format_name):
    """Create a format object on the basis of the format name."""
    format_name = format_name.lower()  # ignore case
    if format_name in ("bismarck", "bismark"):
        return CoverageFormat(
            0,
            1,
            4,
            5,
            False,
            False,
            "\t",
            False,
        )
    elif format_name in ("allc", "methylpy"):
        return CoverageFormat(
            0,
            1,
            4,
            5,
            True,
            False,
            "\t",
            True,
        )
    elif format_name in ("biscuit"):
        return CoverageFormat(
            0,
            1,
            7,  # Methylation share
            7,  # Dummy
            8,  # Total reads
            True,  # is relative
            "\t",
            True,
        )
    elif format_name in ("biscuit_short"):
        return CoverageFormat(
            0,
            1,
            3,  # Methylation share
            3,  # Dummy
            4,  # Total reads
            True,  # is relative
            "\t",
            True,
        )
    else:
        raise Exception(f"{format_name} is not a known format")


def _iterate_covfile(
    cov_file, c_col, p_col, m_col, u_col, coverage, onlyrel, sep, header
):
    try:
        if cov_file.lower().endswith(".gz"):
            # handle gzip-compressed file
            with gzip.open(cov_file, "rb") as lines:
                if header:
                    lines.readline()
                for line in lines:
                    yield _line_to_values(
                        line.decode().strip().split(sep),
                        c_col,
                        p_col,
                        m_col,
                        u_col,
                        coverage,
                        onlyrel,
                    )
        else:
            # handle uncompressed file
            with open(cov_file, "r") as lines:
                if header:
                    lines.readline()
                for line in lines:
                    yield _line_to_values(
                        line.strip().split(sep),
                        c_col,
                        p_col,
                        m_col,
                        u_col,
                        coverage,
                        onlyrel,
                    )
    # we add the name of the file which caused the crash so that the user can fix it
    except Exception as exc:
        raise type(exc)(f"{exc} (in file: {cov_file})").with_traceback(
            sys.exc_info()[2]
        )


def _line_to_values(line, c_col, p_col, m_col, u_col, coverage, onlyrel):
    chrom = line[c_col]
    pos = int(line[p_col])
    if onlyrel:
        n_meth = round(float(line[m_col]) * int(line[coverage]))
        n_unmeth = round((1 - float(line[m_col])) * int(line[coverage]))
    else:
        n_meth = int(line[m_col])
        if coverage:
            n_unmeth = int(line[u_col]) - n_meth
        else:
            n_unmeth = int(line[u_col])
    return chrom, pos, n_meth, n_unmeth
