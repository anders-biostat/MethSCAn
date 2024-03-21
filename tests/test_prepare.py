import os

import numpy as np
import scipy.sparse as sp_sparse
from click.testing import CliRunner

from methscan.cli import cli
from methscan.prepare import (
    _get_cell_names,
    _human_to_computer,
    _iter_chunks,
    _load_csr_from_coo,
)


def test_prepare_cli(tmp_path):
    runner = CliRunner()
    p = os.path.join(tmp_path, "data_dir")
    result = runner.invoke(
        cli,
        [
            "prepare",
            "--chunksize",
            "42",
            "--input-format",
            "biSmArcK",
            "tests/data/tiny/a.cov",
            "tests/data/tiny/b.cov.gz",
            p,
        ],
    )
    assert result.exit_code == 0, result.output
    mat = sp_sparse.load_npz(os.path.join(p, "1.npz"))
    assert mat.shape == (53, 2)
    assert mat.data.shape == (5,)
    mat = sp_sparse.load_npz(os.path.join(p, "2.npz"))
    assert mat.shape == (1236, 2)
    assert mat.data.shape == (5,)
    assert mat[1000, 0] == -1
    assert mat[1234, 0] == 1
    assert mat[42, 0] == 0
    with open(os.path.join(p, "cell_stats.csv")) as stats:
        assert stats.read() == (
            "cell_name,n_obs,n_meth,global_meth_frac\n" "a,5,2,0.4\n" "b,5,3,0.6\n"
        )


def test_prepare_rounded_cli(tmp_path):
    runner = CliRunner()
    p = os.path.join(tmp_path, "data_dir_rounded")
    result = runner.invoke(
        cli,
        [
            "prepare",
            "--chunksize",
            "8",
            "--round-sites",
            "tests/data/tiny/a.cov",
            "tests/data/tiny/b.cov.gz",
            p,
        ],
    )
    assert result.exit_code == 0, result.output
    mat = sp_sparse.load_npz(os.path.join(p, "1.npz"))
    assert mat.shape == (53, 2)
    assert mat.data.shape == (7,)
    assert mat[1, 0] == -1  # 25% methylated
    assert mat[2, 0] == 1  # 75% methylated
    assert mat[3, 0] == 0  # 50% methylated


def test_prepare_glob(tmp_path):
    runner = CliRunner()
    p = os.path.join(tmp_path, "data_dir_glob")
    result = runner.invoke(
        cli,
        [
            "prepare",
            "--chunksize",
            "8",
            "--round-sites",
            "tests/data/tiny/*.cov*",
            p,
        ],
    )
    assert result.exit_code == 0, result.output
    mat = sp_sparse.load_npz(os.path.join(p, "1.npz"))
    assert mat.shape == (53, 2)
    assert mat.data.shape == (7,)
    assert mat[1, 0] == -1  # 25% methylated
    assert mat[2, 0] == 1  # 75% methylated
    assert mat[3, 0] == 0  # 50% methylated


def test_prepare_custom_format_cli(tmp_path):
    runner = CliRunner()
    p = os.path.join(tmp_path, "data_dir_custom_format")
    result = runner.invoke(
        cli,
        [
            "prepare",
            "--chunksize",
            "3",
            "--input-format",
            "2:1:4:3c:;:1",
            "tests/data/tiny_custom/a.cov",
            "tests/data/tiny_custom/b.cov.gz",
            p,
        ],
    )
    assert result.exit_code == 0, result.output
    mat = sp_sparse.load_npz(os.path.join(p, "1.npz"))
    assert mat.shape == (53, 2)
    assert mat.data.shape == (5,)
    mat = sp_sparse.load_npz(os.path.join(p, "2.npz"))
    assert mat.shape == (1236, 2)
    assert mat.data.shape == (5,)
    assert mat[1000, 0] == -1
    assert mat[1234, 0] == 1
    assert mat[42, 0] == 0
    with open(os.path.join(p, "cell_stats.csv")) as stats:
        assert stats.read() == (
            "cell_name,n_obs,n_meth,global_meth_frac\n" "a,5,2,0.4\n" "b,5,3,0.6\n"
        )


def test_load_csr_from_coo():
    chrom_size = 17
    n_cells = 6
    # read CSR matrix from coo chunks, the manual way, using little RAM
    mat1 = _load_csr_from_coo(
        "tests/data/coo_chunks/", "1", chrom_size, n_cells
    ).todense()

    # read CSR matrix the simple way: just load everything into RAM and convert
    coo_chunks = []
    for chunk in _iter_chunks("tests/data/coo_chunks/", "1"):
        coo_chunks.append(chunk)
    coo = np.concatenate(coo_chunks, axis=0)
    mat2 = sp_sparse.coo_matrix(
        (coo[:, 2], (coo[:, 0], coo[:, 1])),
        shape=(chrom_size + 1, n_cells),
        dtype=np.int8,
    )
    mat2 = mat2.todense()
    assert np.array_equal(mat1, mat2)


def test_get_cell_name():
    f = ["dir/a.csv", "/dir/dir2/b.csv.gz", "c.CsV.Gz", "/x/d.e.cSV"]
    assert _get_cell_names(f) == ["a", "b", "c", "d.e"]


def test_coverage_format_creation():
    assert _human_to_computer("Bismarck") == (
        0,
        1,
        4,
        5,
        False,
        False,
        "\t",
        False,
    )
    assert _human_to_computer("allC") == (
        0,
        1,
        4,
        5,
        True,
        False,
        "\t",
        True,
    )
    assert _human_to_computer("1:2:3:4c:\\t:1") == (
        0,
        1,
        2,
        3,
        True,
        False,
        "\t",
        True,
    )
    assert _human_to_computer("1:2:3:4u:\\t:1") == (
        0,
        1,
        2,
        3,
        False,
        False,
        "\t",
        True,
    )
    assert _human_to_computer("1:2:3:4u:\\t:0") == (
        0,
        1,
        2,
        3,
        False,
        False,
        "\t",
        False,
    )
