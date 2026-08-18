from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any

import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError, ShapeError
from polars.testing import assert_frame_equal, assert_series_equal


def test_bin_intervals() -> None:
    s = pl.Series("a", [-2, -1, 0, 1, 2])

    result = s.bin_intervals([-1, 1], labels=["a", "b", "c"])

    expected = pl.Series("a", ["a", "b", "b", "c", "c"], dtype=pl.Enum(["a", "b", "c"]))
    assert_series_equal(result, expected)


def test_bin_intervals_lazy_schema() -> None:
    lf = pl.LazyFrame({"a": [-2, -1, 0, 1, 2]})

    result = lf.select(pl.col("a").bin_intervals([-1, 1], labels=["a", "b", "c"]))

    expected = pl.LazyFrame(
        {"a": ["a", "b", "b", "c", "c"]},
        schema={"a": pl.Enum(["a", "b", "c"])},
    )
    assert_frame_equal(result, expected)


def test_bin_intervals_labels_false() -> None:
    s = pl.Series("a", [-2, -1, 0, 1, 2])

    result = s.bin_intervals([-1, 1], labels=False)

    assert_series_equal(result, pl.Series("a", [0, 1, 1, 2, 2], dtype=pl.UInt32))


def test_bin_intervals_right_closed() -> None:
    s = pl.Series("a", [0, 1, 2, 3, 4])

    left = s.bin_intervals([1, 2, 3], labels=False)
    right = s.bin_intervals([1, 2, 3], labels=False, right_closed=True)

    # A value sitting exactly on a breakpoint belongs to the upper bin when
    # left-closed and to the lower bin when right-closed.
    assert left.to_list() == [0, 1, 2, 3, 3]
    assert right.to_list() == [0, 0, 1, 2, 3]


def test_bin_intervals_single_bin() -> None:
    s = pl.Series("a", [1, 2, 3])

    result = s.bin_intervals([], labels=["only"])

    assert result.to_list() == ["only"] * 3


def test_bin_intervals_include_intervals() -> None:
    s = pl.Series("a", [-2, 0, 2])

    result = s.bin_intervals([-1, 1], labels=False, include_intervals=True)

    expected = pl.Series(
        "a",
        [
            {"bin": 0, "left": None, "right": -1},
            {"bin": 1, "left": -1, "right": 1},
            {"bin": 2, "left": 1, "right": None},
        ],
        dtype=pl.Struct({"bin": pl.UInt32, "left": pl.Int64, "right": pl.Int64}),
    )
    assert_series_equal(result, expected)


def test_bin_intervals_include_intervals_lazy_schema() -> None:
    lf = pl.LazyFrame({"a": [-2, 0, 2]})

    q = lf.select(
        pl.col("a").bin_intervals(
            [-1, 1], labels=["x", "y", "z"], include_intervals=True
        )
    )

    assert q.collect_schema() == q.collect().schema


def test_bin_intervals_label_count_mismatch_raises() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    # Resolved from the schema, so this fails before any data is touched.
    with pytest.raises(ShapeError, match="produces 3 bins but got 1 labels"):
        lf.select(pl.col("a").bin_intervals([1, 2], labels=["x"])).collect_schema()


@pytest.mark.parametrize("intervals", [[2, 1], [1, 1]])
def test_bin_intervals_not_ascending_raises(intervals: list[int]) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    with pytest.raises(ComputeError, match="strictly ascending"):
        lf.select(pl.col("a").bin_intervals(intervals, labels=False)).collect_schema()


def test_bin_intervals_null_breakpoint_raises() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    with pytest.raises(ComputeError, match="cannot contain nulls"):
        lf.select(pl.col("a").bin_intervals([None], labels=False)).collect_schema()


def test_bin_intervals_not_representable_raises() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    # A strict cast would truncate 1.5 to 1 and silently move values between bins.
    with pytest.raises(InvalidOperationError, match="not exactly representable"):
        lf.select(pl.col("a").bin_intervals([1.5], labels=False)).collect_schema()


def test_bin_intervals_integral_float_breakpoint() -> None:
    s = pl.Series("a", [1, 2, 3])

    assert s.bin_intervals([2.0], labels=False).to_list() == [0, 1, 1]


def test_bin_intervals_null_values() -> None:
    s = pl.Series("a", [1, None, 3])

    assert s.bin_intervals([2], labels=False).to_list() == [0, None, 1]


def test_bin_intervals_nan_is_the_largest_value() -> None:
    s = pl.Series("a", [0.0, float("nan"), 2.0])

    # NaN sorts above every other float under total ordering, so it lands in the
    # last bin rather than becoming null.
    assert s.bin_intervals([1.0], labels=False).to_list() == [0, 1, 1]


@pytest.mark.parametrize("include_intervals", [False, True])
def test_bin_intervals_empty(include_intervals: bool) -> None:
    s = pl.Series("a", [], dtype=pl.Int64)

    result = s.bin_intervals([2], labels=False, include_intervals=include_intervals)

    assert result.len() == 0
    if include_intervals:
        assert result.dtype == pl.Struct(
            {"bin": pl.UInt32, "left": pl.Int64, "right": pl.Int64}
        )


@pytest.mark.parametrize("include_intervals", [False, True])
def test_bin_intervals_all_null(include_intervals: bool) -> None:
    s = pl.Series("a", [None, None], dtype=pl.Int64)

    result = s.bin_intervals([2], labels=False, include_intervals=include_intervals)

    assert result.len() == 2
    if include_intervals:
        assert result.dtype == pl.Struct(
            {"bin": pl.UInt32, "left": pl.Int64, "right": pl.Int64}
        )


@pytest.mark.parametrize(
    ("values", "dtype", "breaks"),
    [
        ([1, 2, 3, 4], pl.Int8, [2]),
        ([1, 2, 3, 4], pl.Int64, [2]),
        ([1, 2, 3, 4], pl.UInt32, [2]),
        ([1.0, 2.0, 3.0], pl.Float32, [2.0]),
        ([1.0, 2.0, 3.0], pl.Float64, [2.0]),
        (["a", "b", "c"], pl.String, ["b"]),
        ([True, False, True], pl.Boolean, [True]),
        (
            [datetime.date(2020, 1, 1), datetime.date(2022, 1, 1)],
            pl.Date,
            [datetime.date(2021, 1, 1)],
        ),
        (
            [datetime.datetime(2020, 1, 1), datetime.datetime(2022, 1, 1)],
            pl.Datetime("us"),
            [datetime.datetime(2021, 1, 1)],
        ),
        (
            [datetime.timedelta(days=1), datetime.timedelta(days=5)],
            pl.Duration,
            [datetime.timedelta(days=3)],
        ),
        ([datetime.time(1, 0), datetime.time(5, 0)], pl.Time, [datetime.time(3, 0)]),
        ([Decimal("1.5"), Decimal("2.5")], pl.Decimal(10, 2), [Decimal("2.0")]),
    ],
)
def test_bin_intervals_boundaries_keep_input_dtype(
    values: list[Any], dtype: pl.DataType, breaks: list[Any]
) -> None:
    s = pl.Series("a", values, dtype=dtype)

    result = s.bin_intervals(breaks, labels=False, include_intervals=True)

    # The whole point of these functions over `cut`: boundaries are not forced to f64.
    assert result.struct["left"].dtype == dtype
    assert result.struct["right"].dtype == dtype


def test_bin_intervals_series_breakpoints() -> None:
    s = pl.Series("a", [1, 2, 3])

    from_series = s.bin_intervals(pl.Series([2]), labels=False)
    from_list = s.bin_intervals([2], labels=False)

    assert_series_equal(from_series, from_list)


def test_bin_intervals_categorical_raises() -> None:
    s = pl.Series("a", ["a", "b"], dtype=pl.Categorical)

    # Deferred: `search_sorted` compares categoricals lexically, which is wrong for
    # Enum (declaration order) and meaningless for Categorical (insertion order).
    with pytest.raises(InvalidOperationError, match="not supported"):
        s.bin_intervals(["a"], labels=False)


def test_bin_intervals_over_is_a_noop() -> None:
    df = pl.DataFrame({"a": [1, 5, 9, 2, 6], "g": ["x", "x", "x", "y", "y"]})
    expr = pl.col("a").bin_intervals([4, 7], labels=False)

    # Explicit breakpoints make this elementwise, so grouping cannot change the result.
    assert_series_equal(
        df.select(expr.over("g")).to_series(), df.select(expr).to_series()
    )


def test_bin_intervals_streaming() -> None:
    lf = pl.LazyFrame({"a": [-2, -1, 0, 1, 2]})

    q = lf.select(pl.col("a").bin_intervals([-1, 1], labels=["a", "b", "c"]))

    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_bin_intervals_cse() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    expr = pl.col("a").bin_intervals([2], labels=False)

    result = lf.with_columns(x=expr, y=expr).collect()

    assert result["x"].to_list() == result["y"].to_list()


def test_bin_intervals_serde() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    q = lf.select(pl.col("a").bin_intervals([2], labels=["l", "h"]))

    assert_frame_equal(pl.LazyFrame.deserialize(q.serialize()).collect(), q.collect())


def test_bin_intervals_serde_temporal_breakpoints() -> None:
    lf = pl.LazyFrame({"d": [datetime.date(2020, 1, 1), datetime.date(2022, 1, 1)]})

    q = lf.select(pl.col("d").bin_intervals([datetime.date(2021, 1, 1)], labels=False))

    assert_frame_equal(pl.LazyFrame.deserialize(q.serialize()).collect(), q.collect())


def test_bin_intervals_uniform() -> None:
    s = pl.Series("a", [0, 10])

    # Breakpoints are `min + (i + 1)/n * (max - min)`, so a single break at 5.0.
    assert s.bin_intervals(2, labels=["low", "high"]).to_list() == ["low", "high"]


def test_bin_intervals_uniform_boundaries_are_float() -> None:
    s = pl.Series("a", [0, 100], dtype=pl.Int64)

    result = s.bin_intervals(4, labels=False, include_intervals=True)

    # The one exception to "boundaries carry the input dtype": equal-width breakpoints
    # need arithmetic, and truncating them back to Int64 could collapse two into one.
    assert result.struct["left"].dtype == pl.Float64
    assert result.struct["right"].dtype == pl.Float64
    assert result.struct["right"].to_list() == [25.0, None]


def test_bin_intervals_uniform_lazy_schema() -> None:
    lf = pl.LazyFrame({"a": [0, 100]})

    q = lf.select(pl.col("a").bin_intervals(4, labels=False, include_intervals=True))

    assert q.collect_schema() == q.collect().schema


def test_bin_intervals_uniform_non_numeric_raises() -> None:
    lf = pl.LazyFrame({"a": ["x", "y"]})

    with pytest.raises(InvalidOperationError, match="requires a numeric input"):
        lf.select(pl.col("a").bin_intervals(2, labels=False)).collect_schema()


def test_bin_intervals_uniform_zero_bins_raises() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    with pytest.raises(ComputeError, match="at least one bin"):
        lf.select(pl.col("a").bin_intervals(0, labels=False)).collect_schema()


def test_bin_intervals_uniform_over_is_per_group() -> None:
    df = pl.DataFrame({"a": [0, 10, 0, 1000], "g": ["x", "x", "y", "y"]})

    expr = pl.col("a").bin_intervals(2, labels=False)

    # min/max come from the data, so unlike explicit breakpoints this is per-group.
    assert df.select(expr.over("g")).to_series().to_list() == [0, 1, 0, 1]


@pytest.mark.parametrize("include_intervals", [False, True])
def test_bin_intervals_uniform_all_null(include_intervals: bool) -> None:
    s = pl.Series("a", [None, None], dtype=pl.Int64)

    result = s.bin_intervals(3, labels=False, include_intervals=include_intervals)

    assert result.len() == 2
    if include_intervals:
        assert result.dtype == pl.Struct(
            {"bin": pl.UInt32, "left": pl.Float64, "right": pl.Float64}
        )
