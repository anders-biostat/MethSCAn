import os
import shutil
import gzip

from click.testing import CliRunner
from pandas import read_csv

from methscan.cli import cli


def scan_gzip_contents(path_gz):
    contents = []
    with gzip.open(path_gz, "rt") as infile:
        for line in infile:
            contents.append(line.strip())
    return contents


def test_smooth_cli():
    runner = CliRunner()
    result = runner.invoke(
        cli, ["smooth", "--bandwidth", "2", "tests/data/tiny/data_dir/"]
    )
    with open("tests/data/tiny/data_dir/smoothed/1.csv") as smoothed:
        assert smoothed.read() == "42,0.5\n50,1.0\n52,0.0\n"
    with open("tests/data/tiny/data_dir/smoothed/2.csv") as smoothed:
        assert smoothed.read() == "1000,0.0\n1234,1.0\n1235,1.0\n"
    shutil.rmtree("tests/data/tiny/data_dir/smoothed/")
    assert result.exit_code == 0, result.output


def test_matrix_cli(tmp_path):
    bed = "1\t50\t52\tx\n2\t1000\t1234\ty\n"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "matrix",
            "-",
            "tests/data/tiny/data_dir_smooth/",
            os.path.join(tmp_path, "mtx"),
        ],
        input=bed,
    )
    assert result.exit_code == 0, result.output
    mtx = read_csv(os.path.join(tmp_path, "mtx", "methylation_fractions.csv.gz"))
    assert mtx.values.tolist() == [["a", 0.5, 0.5], ["b", 0.0, 0.5]]
    assert os.path.isfile(os.path.join(tmp_path, "mtx", "total_sites.csv.gz"))
    assert os.path.isfile(os.path.join(tmp_path, "mtx", "methylated_sites.csv.gz"))
    assert os.path.isfile(
        os.path.join(tmp_path, "mtx", "mean_shrunken_residuals.csv.gz")
    )


def test_matrix_sparse_cli(tmp_path):
    bed = "1\t50\t52\tx\n2\t1000\t1234\ty\n"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "matrix",
            "--sparse",
            "-",
            "tests/data/tiny/data_dir_smooth/",
            os.path.join(tmp_path, "s_mtx"),
        ],
        input=bed,
    )
    assert result.exit_code == 0, result.output
    coo_values = set()
    for line in scan_gzip_contents(os.path.join(tmp_path, "s_mtx", "matrix.mtx.gz")):
        values = line.split(" ")
        coo_values.add((int(values[0]), int(values[1]), float(values[3])))
    assert coo_values == {(1, 1, 0.5), (2, 1, 0.0), (1, 2, 0.5), (2, 2, 0.5)}
    feature_names = scan_gzip_contents(
        os.path.join(tmp_path, "s_mtx", "features.tsv.gz")
    )
    assert len(feature_names) == 2
    assert feature_names[0] == "1:50-53"
    assert feature_names[1] == "2:1000-1235"
    cell_names = scan_gzip_contents(os.path.join(tmp_path, "s_mtx", "barcodes.tsv.gz"))
    assert len(cell_names) == 2
    assert cell_names[0] == "a"
    assert cell_names[1] == "b"


def test_profile_cli():
    bed = "1\t51\t51\t+\tx\n2\t1234\t1234\t-\ty\n"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "profile",
            "--strand-column",
            "4",
            "--width",
            "2",
            "-",
            "tests/data/tiny/data_dir/",
            "-",
        ],
        input=bed,
    )
    profile = (
        "position,cell,n_meth,cell_name,n_total,meth_frac,ci_lower,ci_upper\n"
        "-1,1,2,a,2,1.0,0.2902272522159686,1.0\n"
        "-1,2,1,b,1,1.0,0.167499485479413,1.0\n"
    )
    assert result.exit_code == 0, result.output
    assert profile in result.output


def test_filter_cli_threshold(tmp_path):
    p = os.path.join(tmp_path, "filtered_data_dir")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["filter", "--min-meth", "50", "tests/data/tiny/data_dir/", p]
    )
    assert result.exit_code == 0, result.output
    with open(os.path.join(p, "column_header.txt")) as colnames:
        assert colnames.read().strip() == "b"
    with open(os.path.join(p, "cell_stats.csv")) as csv:
        assert csv.readline().startswith("cell_name,")
        assert csv.readline().startswith("b,")
    with open(os.path.join(p, "run_info.txt")) as run_info:
        assert "methscan prepare version" in run_info.read()


def test_filter_cli_toostrict(tmp_path):
    p = os.path.join(tmp_path, "filtered_data_dir")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["filter", "--min-meth", "100", "tests/data/tiny/data_dir/", p]
    )
    assert result.exit_code == 1, result.output


def test_filter_cli_keep(tmp_path):
    p = os.path.join(tmp_path, "filtered_data_dir")
    keep_txt = os.path.join(tmp_path, "cells_to_keep.txt")
    with open(keep_txt, "w") as f:
        f.write("a\n\n")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["filter", "--cell-names", keep_txt, "tests/data/tiny/data_dir/", p]
    )
    assert result.exit_code == 0, result.output
    with open(os.path.join(p, "column_header.txt")) as colnames:
        assert colnames.read().strip() == "a"
    with open(os.path.join(p, "cell_stats.csv")) as csv:
        assert csv.readline().startswith("cell_name,")
        assert csv.readline().startswith("a,")


def test_filter_cli_discard(tmp_path):
    p = os.path.join(tmp_path, "filtered_data_dir")
    discard_txt = os.path.join(tmp_path, "cells_to_discard.txt")
    with open(discard_txt, "w") as f:
        f.write("\na\n\na\n\n")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "filter",
            "--discard",
            "--cell-names",
            discard_txt,
            "tests/data/tiny/data_dir/",
            p,
        ],
    )
    assert result.exit_code == 0, result.output
    with open(os.path.join(p, "column_header.txt")) as colnames:
        assert colnames.read().strip() == "b"
    with open(os.path.join(p, "cell_stats.csv")) as csv:
        assert csv.readline().startswith("cell_name,")
        assert csv.readline().startswith("b,")
