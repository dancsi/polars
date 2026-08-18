from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError, ShapeError
from polars.testing import assert_frame_equal, assert_series_equal


def test_bin_ranks() -> None:
    s = pl.Series("a", [10, 20, 30, 40])

    result = s.bin_ranks([0.5], labels=["low", "high"])

    expected = pl.Series(
        "a", ["low", "low", "high", "high"], dtype=pl.Enum(["low", "high"])
    )
    assert_series_equal(result, expected)


def test_bin_ranks_lazy_schema() -> None:
    lf = pl.LazyFrame({"a": [10, 20, 30, 40]})

    result = lf.select(pl.col("a").bin_ranks([0.5], labels=["low", "high"]))

    expected = pl.LazyFrame(
        {"a": ["low", "low", "high", "high"]},
        schema={"a": pl.Enum(["low", "high"])},
    )
    assert_frame_equal(result, expected)


def test_bin_ranks_labels_false() -> None:
    s = pl.Series("a", [10, 20, 30, 40])

    assert s.bin_ranks([0.5], labels=False).to_list() == [0, 0, 1, 1]


def test_bin_ranks_fractions_give_requested_shares() -> None:
    s = pl.Series("a", list(range(10)))

    result = s.bin_ranks([0.2, 0.5, 0.8], labels=False)

    # Bin `i` holds `ranks[i + 1] - ranks[i]` of the values: 20%, 30%, 30%, 20%.
    assert result.to_list() == [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]


def test_bin_ranks_uniform_sizes() -> None:
    s = pl.Series("a", list(range(14)))

    result = s.bin_ranks(4, labels=False)

    # 14 over 4 bins is 4 + 4 + 3 + 3: the earlier bins take the remainder.
    assert result.value_counts().sort("a")["count"].to_list() == [4, 4, 3, 3]


@pytest.mark.parametrize(
    ("length", "n_bins", "expected"),
    [
        (14, 4, [4, 4, 3, 3]),
        (10, 3, [4, 3, 3]),
        (12, 4, [3, 3, 3, 3]),
        (3, 2, [2, 1]),
    ],
)
def test_bin_ranks_uniform_sizes_parametrized(
    length: int, n_bins: int, expected: list[int]
) -> None:
    s = pl.Series("a", list(range(length)))

    result = s.bin_ranks(n_bins, labels=False)

    assert result.value_counts().sort("a")["count"].to_list() == expected


def test_bin_ranks_splits_ties() -> None:
    # Every value is identical, so no value-based binning could separate them at all.
    s = pl.Series("a", [7] * 14)

    result = s.bin_ranks(4, labels=False)

    assert result.value_counts().sort("a")["count"].to_list() == [4, 4, 3, 3]


def test_bin_ranks_ties_broken_by_input_order() -> None:
    s = pl.Series("a", [5, 5])

    # Position, not value, decides: the first occurrence goes to the lower bin.
    assert s.bin_ranks(2, labels=False).to_list() == [0, 1]


def test_bin_ranks_boundaries_are_values_not_ranks() -> None:
    s = pl.Series("a", [10, 20, 30, 40])

    result = s.bin_ranks(2, labels=False, include_intervals=True)

    expected = pl.Series(
        "a",
        [
            {"bin": 0, "left": None, "right": 30},
            {"bin": 0, "left": None, "right": 30},
            {"bin": 1, "left": 30, "right": None},
            {"bin": 1, "left": 30, "right": None},
        ],
        dtype=pl.Struct({"bin": pl.UInt32, "left": pl.Int64, "right": pl.Int64}),
    )
    assert_series_equal(result, expected)


def test_bin_ranks_include_intervals_lazy_schema() -> None:
    lf = pl.LazyFrame({"a": [10, 20, 30, 40]})

    q = lf.select(pl.col("a").bin_ranks(2, labels=["l", "h"], include_intervals=True))

    assert q.collect_schema() == q.collect().schema


def test_bin_ranks_boundaries_keep_input_dtype() -> None:
    s = pl.Series("a", [1, 2, 3, 4], dtype=pl.Int16)

    result = s.bin_ranks(2, labels=False, include_intervals=True)

    assert result.struct["left"].dtype == pl.Int16
    assert result.struct["right"].dtype == pl.Int16


def test_bin_ranks_on_non_numeric() -> None:
    s = pl.Series("a", ["a", "b", "c", "d"])

    result = s.bin_ranks(2, labels=False, include_intervals=True)

    assert result.struct["bin"].to_list() == [0, 0, 1, 1]
    assert result.struct["left"].dtype == pl.String


def test_bin_ranks_has_no_right_closed() -> None:
    s = pl.Series("a", [1, 2])

    # Bins are delimited by position, so there is no value boundary to close on.
    with pytest.raises(TypeError, match="unexpected keyword argument 'right_closed'"):
        s.bin_ranks(2, labels=False, right_closed=True)  # type: ignore[call-arg]


def test_bin_ranks_trailing_fraction_of_one_gives_an_empty_bin() -> None:
    s = pl.Series("a", [1, 2, 3, 4])

    result = s.bin_ranks([0.5, 1.0], labels=False, include_intervals=True)

    # `1.0` puts the second cut past the last element, so the third bin is empty and
    # the boundary value for it is null.
    assert result.struct["bin"].to_list() == [0, 0, 1, 1]
    assert result.struct["right"].to_list() == [3, 3, None, None]


def test_bin_ranks_more_bins_than_rows() -> None:
    s = pl.Series("a", [1, 2, 3])

    assert s.bin_ranks(5, labels=False).to_list() == [0, 1, 2]


def test_bin_ranks_label_count_mismatch_raises() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    with pytest.raises(ShapeError, match="produces 2 bins but got 1 labels"):
        lf.select(pl.col("a").bin_ranks([0.5], labels=["x"])).collect_schema()


@pytest.mark.parametrize("ranks", [[0.6, 0.3], [0.5, 0.5]])
def test_bin_ranks_not_ascending_raises(ranks: list[float]) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    with pytest.raises(ComputeError, match="strictly ascending"):
        lf.select(pl.col("a").bin_ranks(ranks, labels=False)).collect_schema()


@pytest.mark.parametrize("ranks", [[-0.1], [1.5]])
def test_bin_ranks_out_of_range_raises(ranks: list[float]) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    with pytest.raises(ComputeError, match=r"between 0\.0 and 1\.0"):
        lf.select(pl.col("a").bin_ranks(ranks, labels=False)).collect_schema()


def test_bin_ranks_zero_bins_raises() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    with pytest.raises(ComputeError, match="at least one bin"):
        lf.select(pl.col("a").bin_ranks(0, labels=False)).collect_schema()


def test_bin_ranks_null_values() -> None:
    s = pl.Series("a", [1, None, 3, 5])

    # Nulls are excluded from the ranking and stay null in the output.
    assert s.bin_ranks([0.5], labels=False).to_list() == [0, None, 0, 1]


@pytest.mark.parametrize("include_intervals", [False, True])
def test_bin_ranks_empty(include_intervals: bool) -> None:
    s = pl.Series("a", [], dtype=pl.Int64)

    result = s.bin_ranks(2, labels=False, include_intervals=include_intervals)

    assert result.len() == 0
    if include_intervals:
        assert result.dtype == pl.Struct(
            {"bin": pl.UInt32, "left": pl.Int64, "right": pl.Int64}
        )


@pytest.mark.parametrize("include_intervals", [False, True])
def test_bin_ranks_all_null(include_intervals: bool) -> None:
    s = pl.Series("a", [None, None], dtype=pl.Int64)

    result = s.bin_ranks(2, labels=False, include_intervals=include_intervals)

    assert result.len() == 2
    if include_intervals:
        assert result.dtype == pl.Struct(
            {"bin": pl.UInt32, "left": pl.Int64, "right": pl.Int64}
        )


def test_bin_ranks_categorical_raises() -> None:
    s = pl.Series("a", ["a", "b"], dtype=pl.Categorical)

    with pytest.raises(InvalidOperationError, match="not supported"):
        s.bin_ranks(2, labels=False)


def test_bin_ranks_over_is_per_group() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 100, 200], "g": ["x"] * 3 + ["y"] * 3})

    expr = pl.col("a").bin_ranks(2, labels=False)

    # Bins are positional within each group.
    assert df.select(expr.over("g")).to_series().to_list() == [0, 0, 1, 0, 0, 1]


def test_bin_ranks_streaming() -> None:
    lf = pl.LazyFrame({"a": [10, 20, 30, 40]})

    q = lf.select(pl.col("a").bin_ranks(2, labels=["l", "h"]))

    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_bin_ranks_serde() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3, 4]})

    q = lf.select(pl.col("a").bin_ranks(2, labels=["l", "h"]))

    assert_frame_equal(pl.LazyFrame.deserialize(q.serialize()).collect(), q.collect())
