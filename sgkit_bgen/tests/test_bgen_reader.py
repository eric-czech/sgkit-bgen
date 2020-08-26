import numpy as np
import numpy.testing as npt
import pytest
from sgkit_bgen import read_bgen

CHUNKS = [
    (100, 200, 3),
    (100, 200, 1),
    (100, 500, 3),
    (199, 500, 3),
    ((100, 99), 500, 2),
    "auto",
]
INDEXES = [0, 10, 20, 100, -1]

# Expectations below generated using bgen-reader directly, ex:
# > from bgen_reader import open_bgen
# > bgen = open_bgen('sgkit_bgen/tests/data/example.bgen', verbose=False)
# > bgen.read(-1)[0] # Probabilities for last variant, first sample
# array([[0.0133972 , 0.98135378, 0.00524902]]
# > bgen.allele_expectation(-1)[0, 0, -1] # Dosage for last variant, first sample
# 0.9918518217727197
EXPECTED_PROBABILITIES = np.array(
    [  # Generated using bgen-reader directly
        [np.nan, np.nan, np.nan],
        [0.007, 0.966, 0.0259],
        [0.993, 0.002, 0.003],
        [0.916, 0.007, 0.0765],
        [0.013, 0.981, 0.0052],
    ]
)
EXPECTED_DOSAGES = np.array(
    [np.nan, 1.018, 0.010, 0.160, 0.991]  # Generated using bgen-reader directly
)


@pytest.mark.parametrize("chunks", CHUNKS)
def test_read_bgen(shared_datadir, chunks):
    path = shared_datadir / "example.bgen"
    ds = read_bgen(path, chunks=chunks)

    # check some of the data (in different chunks)
    assert ds["call_dosage"].shape == (199, 500)
    npt.assert_almost_equal(ds["call_dosage"].values[1][0], 1.987, decimal=3)
    npt.assert_almost_equal(ds["call_dosage"].values[100][0], 0.160, decimal=3)
    npt.assert_array_equal(ds["call_dosage_mask"].values[0, 0], [True])
    npt.assert_array_equal(ds["call_dosage_mask"].values[0, 1], [False])
    assert ds["call_genotype_probability"].shape == (199, 500, 3)
    npt.assert_almost_equal(
        ds["call_genotype_probability"].values[1][0], [0.005, 0.002, 0.992], decimal=3
    )
    npt.assert_almost_equal(
        ds["call_genotype_probability"].values[100][0], [0.916, 0.007, 0.076], decimal=3
    )
    npt.assert_array_equal(
        ds["call_genotype_probability_mask"].values[0, 0], [True] * 3
    )
    npt.assert_array_equal(
        ds["call_genotype_probability_mask"].values[0, 1], [False] * 3
    )


def test_read_bgen_with_sample_file(shared_datadir):
    # The example file was generated using
    # qctool -g sgkit_bgen/tests/data/example.bgen -og sgkit_bgen/tests/data/example-separate-samples.bgen -os sgkit_bgen/tests/data/example-separate-samples.sample -incl-samples sgkit_bgen/tests/data/samples
    # Then editing example-separate-samples.sample to change the sample IDs
    path = shared_datadir / "example-separate-samples.bgen"
    ds = read_bgen(path)
    # Check the sample IDs are the ones from the .sample file
    assert ds["sample_id"].values.tolist() == ["s1", "s2", "s3", "s4", "s5"]


def test_read_bgen_with_no_samples(shared_datadir):
    # The example file was generated using
    # qctool -g sgkit_bgen/tests/data/example.bgen -og sgkit_bgen/tests/data/example-no-samples.bgen -os sgkit_bgen/tests/data/example-no-samples.sample -bgen-omit-sample-identifier-block -incl-samples sgkit_bgen/tests/data/samples
    # Then deleting example-no-samples.sample
    path = shared_datadir / "example-no-samples.bgen"
    ds = read_bgen(path)
    # Check the sample IDs are generated
    assert ds["sample_id"].values.tolist() == [
        "sample_0",
        "sample_1",
        "sample_2",
        "sample_3",
        "sample_4",
    ]


@pytest.mark.parametrize("chunks", CHUNKS)
def test_read_bgen_fancy_index(shared_datadir, chunks):
    path = shared_datadir / "example.bgen"
    ds = read_bgen(path, chunks=chunks)
    npt.assert_almost_equal(
        ds["call_genotype_probability"][INDEXES, 0], EXPECTED_PROBABILITIES, decimal=3
    )
    npt.assert_almost_equal(ds["call_dosage"][INDEXES, 0], EXPECTED_DOSAGES, decimal=3)


@pytest.mark.parametrize("chunks", CHUNKS)
def test_read_bgen_scalar_index(shared_datadir, chunks):
    path = shared_datadir / "example.bgen"
    ds = read_bgen(path, chunks=chunks)
    for i, ix in enumerate(INDEXES):
        npt.assert_almost_equal(
            ds["call_genotype_probability"][ix, 0], EXPECTED_PROBABILITIES[i], decimal=3
        )
        npt.assert_almost_equal(
            ds["call_dosage"][ix, 0], EXPECTED_DOSAGES[i], decimal=3
        )
        for j in range(3):
            npt.assert_almost_equal(
                ds["call_genotype_probability"][ix, 0, j],
                EXPECTED_PROBABILITIES[i, j],
                decimal=3,
            )


# TODO:
# - test that encoded GP works
# - test that str max len works
