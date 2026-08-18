from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import ComputeError, ShapeError
from polars.testing import assert_frame_equal, assert_series_equal


def test_bin_quantiles() -> None:
    s = pl.Series("a", [-2, -1, 0, 1, 2])

    result = s.bin_quantiles([0.25, 0.75], labels=["a", "b", "c"])

    expected = pl.Series("a", ["a", "b", "b", "c", "c"], dtype=pl.Enum(["a", "b", "c"]))
    assert_series_equal(result, expected)


def test_bin_quantiles_lazy_schema() -> None:
    lf = pl.LazyFrame({"a": [-2, -1, 0, 1, 2]})

    result = lf.select(pl.col("a").bin_quantiles([0.25, 0.75], labels=["a", "b", "c"]))

    expected = pl.LazyFrame(
        {"a": ["a", "b", "b", "c", "c"]}, schema={"a": pl.Enum(["a", "b", "c"])}
    )
    assert_frame_equal(result, expected)


def test_bin_quantiles_labels_false() -> None:
    s = pl.Series("a", [-2, -1, 0, 1, 2])

    assert s.bin_quantiles([0.25, 0.75], labels=False).to_list() == [0, 1, 1, 2, 2]


def test_bin_quantiles_uniform() -> None:
    s = pl.Series("a", [-2, -1, 0, 1, 2])

    result = s.bin_quantiles(2, labels=["low", "high"])

    assert result.to_list() == ["low", "low", "high", "high", "high"]


def test_bin_quantiles_uniform_is_computed_exactly() -> None:
    s = pl.Series("a", list(range(91)))

    result = s.bin_quantiles(10, labels=False, include_intervals=True)
    breaks = sorted(set(result.struct["right"].drop_nulls().to_list()))

    # Positions are `((i + 1) * (len - 1)) / n_bins` in integer arithmetic. Expanding to
    # `(i + 1) / n_bins` probabilities instead would put breakpoint 7 at 62, because
    # `7 / 10` is 0.69999999999999996 and `0.7 * 90` floors to 62.
    assert breaks == [9 * (i + 1) for i in range(9)]
    assert breaks[6] == 63


def test_bin_quantiles_uniform_differs_from_expanded_probabilities() -> None:
    s = pl.Series("a", list(range(91)))

    uniform = s.bin_quantiles(10, labels=False)
    expanded = s.bin_quantiles([i / 10 for i in range(1, 10)], labels=False)

    # Deliberate: the two forms are not required to agree, because only the integer
    # form can be evaluated without going through inexact probabilities.
    assert uniform.to_list() != expanded.to_list()


def test_bin_quantiles_include_intervals() -> None:
    s = pl.Series("a", [1, 2, 3, 4, 5])

    result = s.bin_quantiles([0.25, 0.75], labels=False, include_intervals=True)

    # floor(0.25 * 4) == 1 and floor(0.75 * 4) == 3, so the breakpoints are the values
    # at those sorted positions: 2 and 4.
    expected = pl.Series(
        "a",
        [
            {"bin": 0, "left": None, "right": 2},
            {"bin": 1, "left": 2, "right": 4},
            {"bin": 1, "left": 2, "right": 4},
            {"bin": 2, "left": 4, "right": None},
            {"bin": 2, "left": 4, "right": None},
        ],
        dtype=pl.Struct({"bin": pl.UInt32, "left": pl.Int64, "right": pl.Int64}),
    )
    assert_series_equal(result, expected)


def test_bin_quantiles_include_intervals_lazy_schema() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})

    q = lf.select(
        pl.col("a").bin_quantiles([0.5], labels=["l", "h"], include_intervals=True)
    )

    assert q.collect_schema() == q.collect().schema


def test_bin_quantiles_boundaries_keep_input_dtype() -> None:
    s = pl.Series("a", [1, 2, 3, 4, 5], dtype=pl.Int16)

    result = s.bin_quantiles([0.5], labels=False, include_intervals=True)

    # Breakpoints are chosen positionally, so they are values from the input rather
    # than interpolated floats.
    assert result.struct["left"].dtype == pl.Int16
    assert result.struct["right"].dtype == pl.Int16


def test_bin_quantiles_enum_uses_declaration_order() -> None:
    dtype = pl.Enum(["zebra", "apple", "mango"])
    s = pl.Series("a", ["mango", "zebra", "apple"], dtype=dtype)

    result = s.bin_quantiles([0.5], labels=False, include_intervals=True)

    # Sorted by declaration order the values are zebra, apple, mango, so the median
    # breakpoint is apple.
    assert result.struct["bin"].to_list() == [1, 0, 1]
    assert result.struct["left"].dtype == dtype


def test_bin_quantiles_categorical_uses_lexical_order() -> None:
    s = pl.Series("a", ["mango", "zebra", "apple"], dtype=pl.Categorical)

    result = s.bin_quantiles([0.5], labels=False)

    # Sorted lexically the values are apple, mango, zebra, so the breakpoint is mango.
    assert result.to_list() == [1, 1, 0]


def test_bin_quantiles_on_non_numeric() -> None:
    s = pl.Series("a", ["a", "b", "c", "d", "e"])

    result = s.bin_quantiles([0.5], labels=False, include_intervals=True)

    assert result.struct["left"].dtype == pl.String
    assert result.struct["bin"].to_list() == [0, 0, 1, 1, 1]


def test_bin_quantiles_label_count_mismatch_raises() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    with pytest.raises(ShapeError, match="produces 3 bins but got 2 labels"):
        lf.select(
            pl.col("a").bin_quantiles([0.3, 0.6], labels=["x", "y"])
        ).collect_schema()


@pytest.mark.parametrize("quantiles", [[0.6, 0.3], [0.5, 0.5]])
def test_bin_quantiles_not_ascending_raises(quantiles: list[float]) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    with pytest.raises(ComputeError, match="strictly ascending"):
        lf.select(pl.col("a").bin_quantiles(quantiles, labels=False)).collect_schema()


@pytest.mark.parametrize("quantiles", [[-0.1], [1.5]])
def test_bin_quantiles_out_of_range_raises(quantiles: list[float]) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    with pytest.raises(ComputeError, match=r"between 0\.0 and 1\.0"):
        lf.select(pl.col("a").bin_quantiles(quantiles, labels=False)).collect_schema()


def test_bin_quantiles_zero_bins_raises() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    with pytest.raises(ComputeError, match="at least one bin"):
        lf.select(pl.col("a").bin_quantiles(0, labels=False)).collect_schema()


def test_bin_quantiles_null_values() -> None:
    s = pl.Series("a", [1, None, 3, 5])

    # Nulls are excluded from the quantile population and stay null in the output.
    assert s.bin_quantiles([0.5], labels=False).to_list() == [0, None, 1, 1]


@pytest.mark.parametrize("include_intervals", [False, True])
def test_bin_quantiles_empty(include_intervals: bool) -> None:
    s = pl.Series("a", [], dtype=pl.Int64)

    result = s.bin_quantiles([0.5], labels=False, include_intervals=include_intervals)

    assert result.len() == 0
    if include_intervals:
        assert result.dtype == pl.Struct(
            {"bin": pl.UInt32, "left": pl.Int64, "right": pl.Int64}
        )


@pytest.mark.parametrize("include_intervals", [False, True])
def test_bin_quantiles_all_null(include_intervals: bool) -> None:
    s = pl.Series("a", [None, None], dtype=pl.Int64)

    result = s.bin_quantiles([0.5], labels=False, include_intervals=include_intervals)

    assert result.len() == 2
    if include_intervals:
        assert result.dtype == pl.Struct(
            {"bin": pl.UInt32, "left": pl.Int64, "right": pl.Int64}
        )


def test_bin_quantiles_over_is_per_group() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 100, 200, 300], "g": ["x"] * 3 + ["y"] * 3})

    expr = pl.col("a").bin_quantiles([0.5], labels=False)

    # The breakpoints are derived from the data, so each group gets its own.
    assert df.select(expr.over("g")).to_series().to_list() == [0, 1, 1, 0, 1, 1]


def test_bin_quantiles_streaming() -> None:
    lf = pl.LazyFrame({"a": [-2, -1, 0, 1, 2]})

    q = lf.select(pl.col("a").bin_quantiles([0.5], labels=["l", "h"]))

    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_bin_quantiles_serde() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    q = lf.select(pl.col("a").bin_quantiles(2, labels=["l", "h"]))

    assert_frame_equal(pl.LazyFrame.deserialize(q.serialize()).collect(), q.collect())
