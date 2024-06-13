import pytest
import numpy as np
from click.testing import CliRunner

from methscan.cli import cli
from methscan.scan import _find_peaks


TEST_VARS1 = np.array(
    [
        0.6,
        0.1,
        0.2,
        0.1,
        0.5,
        0.4,
        0.6,
        0.4,
        0.5,
        0.4,
        0.5,
        0.4,
        0.7,
        0.8,
        0.9,
        1,
        0.1,
        0.6,
        0.4,
        0.2,
        0.1,
        0.1,
        0.2,
        0.1,
        0.7,
        0.1,
        0.1,
        0.2,
        0.1,
        0.2,
        0.1,
        0.2,
        0.1,
        0.2,
        0.6,
    ]
)
TEST_VARS2 = np.array(
    [
        0,
        np.nan,
        0,
        0.5,
        0.2,
        0.2,
        0.2,
        0.2,
        0.5,
        0.2,
        0.2,
        0.2,
        0.5,
        0.2,
        0.2,
        0.5,
        0.2,
        0.5,
        0.2,
        0.5,
        0.2,
        0.5,
        0,
        0,
        0,
        0,
        0.5,
        0.5,
        0.2,
        0.2,
        0.2,
        0.2,
        0.5,
        0.5,
        0.2,
        0.2,
        0.2,
        0.5,
        0.5,
        0.2,
        0.2,
        0.5,
        0.5,
        0.2,
        0.5,
        0.5,
        0.2,
        0.5,
        0.5,
        0.2,
        0.5,
        0.5,
        0,
        0,
        0,
        0,
    ]
)

FIND_PEAKS_TEST_CASES = [
    {
        "input": {
            "smoothed_vars": TEST_VARS1,
            "bandwidth": 2000,
            "stepsize": 100,
            "var_cutoff": 0.55,
        },
        "output": {
            "peak_starts": [0],
            "peak_ends": [5400],
        },
    },
    {
        "input": {
            "smoothed_vars": TEST_VARS1,
            "bandwidth": 200,
            "stepsize": 100,
            "var_cutoff": 0.55,
        },
        "output": {
            "peak_starts": [0, 600, 1200, 2400, 3400],
            "peak_ends": [200, 800, 1900, 2600, 3600],
        },
    },
    {
        "input": {
            "smoothed_vars": TEST_VARS1,
            "bandwidth": 2000,
            "stepsize": 2000,
            "var_cutoff": 0.55,
        },
        "output": {
            "peak_starts": [0, 12000, 24000, 34000, 48000, 68000],
            "peak_ends": [2000, 14000, 32000, 36000, 50000, 70000],
        },
    },
    {
        "input": {
            "smoothed_vars": TEST_VARS1,
            "bandwidth": 100,
            "stepsize": 200,
            "var_cutoff": 0.55,
        },
        "output": {
            "peak_starts": [0, 1200, 2400, 3400, 4800, 6800],
            "peak_ends": [100, 1300, 3100, 3500, 4900, 6900],
        },
    },
    {
        "input": {
            "smoothed_vars": TEST_VARS1,
            "bandwidth": 398,
            "stepsize": 200,
            "var_cutoff": 0.55,
        },
        "output": {
            "peak_starts": [0, 1200, 2400, 3400, 4800, 6800],
            "peak_ends": [398, 1598, 3398, 3798, 5198, 7198],
        },
    },
    {
        "input": {
            "smoothed_vars": TEST_VARS1,
            "bandwidth": 450,
            "stepsize": 200,
            "var_cutoff": 0.25,
        },
        "output": {
            "peak_starts": [0, 800, 4800, 6800],
            "peak_ends": [450, 4050, 5250, 7250],
        },
    },
    {
        "input": {
            "smoothed_vars": TEST_VARS1,
            "bandwidth": 250,
            "stepsize": 100,
            "var_cutoff": 0.85,
        },
        "output": {
            "peak_starts": [1400],
            "peak_ends": [1750],
        },
    },
    {
        "input": {
            "smoothed_vars": TEST_VARS2,
            "bandwidth": 300,
            "stepsize": 100,
            "var_cutoff": 0.3,
        },
        "output": {
            "peak_starts": [300, 800, 1200, 2600, 3200, 3700],
            "peak_ends": [600, 1100, 2400, 3000, 3600, 5400],
        },
    },
]


def test_find_peaks():
    """
    test the variance peak calling function on a bunch
    of made-up test data sets
    """
    for test_id, test_dict in enumerate(FIND_PEAKS_TEST_CASES):
        run_find_peaks(test_dict)


def run_find_peaks(test_dict):
    stepsize = test_dict["input"]["stepsize"]
    half_bw = test_dict["input"]["bandwidth"] // 2
    var_cutoff = test_dict["input"]["var_cutoff"]
    smoothed_vars = test_dict["input"]["smoothed_vars"]
    swindow_centers = (np.arange(0, smoothed_vars.size) * stepsize) + half_bw
    starts, ends = _find_peaks(smoothed_vars, swindow_centers, var_cutoff, half_bw)
    assert np.array_equal(starts, np.array(test_dict["output"]["peak_starts"]))
    assert np.array_equal(ends, np.array(test_dict["output"]["peak_ends"]))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scan_cli():
    """
    pretty poor test currently, since the test data set is not useful.
    it just makes sure that the CLI doesn't crash.
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "scan",
            "--min-cells",
            "1",
            "--threads",
            "1",
            "--bandwidth",
            "2",
            "tests/data/tiny_diff/",
            "-",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Found no variably methylated regions" in result.output
