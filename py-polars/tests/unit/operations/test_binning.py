"""Behaviour shared by `bin_intervals`, `bin_quantiles` and `bin_ranks`.

Each of the three functions takes either an explicit sequence or a bin count, giving six
forms in total. Anything that should hold for all of them is parameterized here; the
per-function files cover only what genuinely differs between them.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError, ShapeError
from polars.testing import assert_frame_equal, assert_series_equal
from tests.unit.conftest import NUMERIC_DTYPES

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

# Every form is set up to produce exactly three bins over a four-element column, so the
# label, schema and shape assertions below are shared.
BinOp = Callable[..., Any]

ANY_ORD_OPS = [
    pytest.param(lambda o, **kw: o.bin_intervals([2, 4], **kw), id="intervals-list"),
    pytest.param(lambda o, **kw: o.bin_ranks([0.3, 0.6], **kw), id="ranks-list"),
    pytest.param(lambda o, **kw: o.bin_ranks(3, **kw), id="ranks-int"),
]
NUMERIC_ONLY_OPS = [
    pytest.param(lambda o, **kw: o.bin_intervals(3, **kw), id="intervals-int"),
    pytest.param(
        lambda o, **kw: o.bin_quantiles([0.3, 0.6], **kw), id="quantiles-list"
    ),
    pytest.param(lambda o, **kw: o.bin_quantiles(3, **kw), id="quantiles-int"),
]
BIN_OPS = ANY_ORD_OPS + NUMERIC_ONLY_OPS

# The dtype matrix runs over data types with no meaningful integer literal, so
# `bin_intervals` there takes its breakpoints out of the column itself.
MATRIX_ANY_ORD_OPS = [
    pytest.param(
        lambda o, **kw: o.bin_intervals(o.slice(1, 2), **kw), id="intervals-list"
    ),
    *ANY_ORD_OPS[1:],
]

LABELS = ["a", "b", "c"]

# `bin_quantiles` is numeric-only per the spec; the other two accept any ordered type.
NUMERIC_INPUT_DTYPES = [*NUMERIC_DTYPES, pl.Decimal(None, 2)]
NON_NUMERIC_ORD_DTYPES = [
    pl.String,
    pl.Boolean,
    pl.Date,
    pl.Datetime("us"),
    pl.Duration("us"),
    pl.Time,
    pl.Enum(["a", "b", "c", "d"]),
    pl.Categorical,
]


def values_of(dtype: PolarsDataType) -> pl.Series:
    """Four ascending values of `dtype`, named `a`."""
    if dtype in (pl.String, pl.Categorical) or isinstance(dtype, pl.Enum):
        return pl.Series("a", ["a", "b", "c", "d"], dtype=dtype)
    if dtype == pl.Boolean:
        return pl.Series("a", [False, False, True, True])
    if dtype == pl.Date:
        return pl.Series("a", [datetime.date(2020, 1, d) for d in (1, 2, 3, 4)])
    if isinstance(dtype, pl.Datetime):
        return pl.Series(
            "a", [datetime.datetime(2020, 1, d) for d in (1, 2, 3, 4)], dtype=dtype
        )
    if isinstance(dtype, pl.Duration):
        return pl.Series(
            "a", [datetime.timedelta(days=d) for d in (1, 2, 3, 4)], dtype=dtype
        )
    if dtype == pl.Time:
        return pl.Series("a", [datetime.time(h) for h in (1, 2, 3, 4)])
    return pl.Series("a", [1, 2, 3, 4]).cast(dtype)


@pytest.mark.parametrize("op", BIN_OPS)
def test_labels_false_is_u32(op: BinOp) -> None:
    result = op(pl.Series("a", [1, 2, 3, 4]), labels=False)

    assert result.dtype == pl.UInt32
    assert result.name == "a"


@pytest.mark.parametrize("op", BIN_OPS)
def test_labels_give_an_enum(op: BinOp) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3, 4]})

    q = lf.select(op(pl.col("a"), labels=LABELS))

    # The label set is known at plan time, so the Enum resolves without touching data.
    assert q.collect_schema()["a"] == pl.Enum(LABELS)
    assert_frame_equal(q.collect(), q.collect(), check_dtypes=True)
    assert q.collect_schema() == q.collect().schema


@pytest.mark.parametrize("op", BIN_OPS)
@pytest.mark.parametrize("labels", [False, LABELS])
def test_include_intervals_schema_matches_collect(
    op: BinOp, labels: list[str] | bool
) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3, 4]})

    q = lf.select(op(pl.col("a"), labels=labels, include_intervals=True))

    assert q.collect_schema() == q.collect().schema
    dtype = cast("pl.Struct", q.collect_schema()["a"])
    assert [f.name for f in dtype.fields] == ["bin", "left", "right"]


@pytest.mark.parametrize("op", BIN_OPS)
def test_label_count_mismatch_raises(op: BinOp) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3, 4]})

    # Resolved from the schema, so this fails before any data is touched.
    with pytest.raises(ShapeError, match="produces 3 bins but got 2 labels"):
        lf.select(op(pl.col("a"), labels=["x", "y"])).collect_schema()


@pytest.mark.parametrize("op", BIN_OPS)
def test_null_values_stay_null(op: BinOp) -> None:
    s = pl.Series("a", [1, None, 3, None, 5, 7])

    result = op(s, labels=False)

    # Nulls are excluded from the binning population and stay null in the output.
    assert [v is None for v in result.to_list()] == [False, True, False, True] + [
        False,
        False,
    ]


@pytest.mark.parametrize("op", BIN_OPS)
@pytest.mark.parametrize("include_intervals", [False, True])
def test_empty_input(op: BinOp, include_intervals: bool) -> None:
    s = pl.Series("a", [], dtype=pl.Int64)

    result = op(s, labels=False, include_intervals=include_intervals)

    assert result.len() == 0
    expected = pl.Struct({"bin": pl.UInt32, "left": pl.Int64, "right": pl.Int64})
    assert result.dtype == (expected if include_intervals else pl.UInt32)


@pytest.mark.parametrize("op", BIN_OPS)
@pytest.mark.parametrize("include_intervals", [False, True])
def test_all_null_input(op: BinOp, include_intervals: bool) -> None:
    s = pl.Series("a", [None, None], dtype=pl.Int64)

    result = op(s, labels=False, include_intervals=include_intervals)

    assert result.len() == 2
    expected = pl.Struct({"bin": pl.UInt32, "left": pl.Int64, "right": pl.Int64})
    assert result.dtype == (expected if include_intervals else pl.UInt32)
    if include_intervals:
        assert result.struct["bin"].null_count() == 2
    else:
        assert result.null_count() == 2


@pytest.mark.parametrize("op", BIN_OPS)
def test_streaming_matches_in_memory(op: BinOp) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5, 6]})

    q = lf.select(op(pl.col("a"), labels=LABELS, include_intervals=True))

    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


@pytest.mark.parametrize("op", BIN_OPS)
def test_serde_roundtrip(op: BinOp) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3, 4]})

    q = lf.select(op(pl.col("a"), labels=LABELS, include_intervals=True))

    assert_frame_equal(pl.LazyFrame.deserialize(q.serialize()).collect(), q.collect())


@pytest.mark.parametrize("op", BIN_OPS)
def test_cse(op: BinOp) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3, 4]})
    expr = op(pl.col("a"), labels=False)

    result = lf.with_columns(x=expr, y=expr).collect()

    assert result["x"].to_list() == result["y"].to_list()


@pytest.mark.parametrize("op", BIN_OPS)
def test_over_matches_manual_per_group(op: BinOp) -> None:
    df = pl.DataFrame(
        {"a": [1, 5, 9, 13, 2, 60, 400, 4000], "g": ["x"] * 4 + ["y"] * 4}
    )
    expr = op(pl.col("a"), labels=False)

    windowed = df.select(expr.over("g")).to_series()
    manual = pl.concat(
        [
            part.select(expr).to_series()
            for _, part in df.group_by("g", maintain_order=True)
        ]
    )

    assert windowed.to_list() == manual.to_list()


@pytest.mark.parametrize("op", BIN_OPS)
def test_nan_lands_in_the_last_bin(op: BinOp) -> None:
    s = pl.Series("a", [0.0, 1.0, 2.0, float("nan")])

    result = op(s, labels=False)

    # NaN sorts above every other float under total ordering, so it is never null.
    assert result.null_count() == 0
    assert result[3] == max(result.to_list())


@pytest.mark.parametrize("op", MATRIX_ANY_ORD_OPS)
@pytest.mark.parametrize("dtype", NUMERIC_INPUT_DTYPES + NON_NUMERIC_ORD_DTYPES)
def test_any_ord_ops_support_all_orderable_dtypes(
    op: BinOp, dtype: PolarsDataType
) -> None:
    s = values_of(dtype)

    result = op(s, labels=False, include_intervals=True)

    assert result.struct["bin"].dtype == pl.UInt32
    # Boundaries carry the input data type, with no exceptions.
    assert result.struct["left"].dtype == dtype
    assert result.struct["right"].dtype == dtype


@pytest.mark.parametrize("op", NUMERIC_ONLY_OPS)
@pytest.mark.parametrize("dtype", NUMERIC_INPUT_DTYPES)
def test_numeric_only_ops_support_all_numeric_dtypes(
    op: BinOp, dtype: PolarsDataType
) -> None:
    s = values_of(dtype)

    result = op(s, labels=False, include_intervals=True)

    assert result.struct["bin"].dtype == pl.UInt32
    assert result.struct["left"].dtype == dtype
    assert result.struct["right"].dtype == dtype


@pytest.mark.parametrize("op", NUMERIC_ONLY_OPS)
@pytest.mark.parametrize("dtype", NON_NUMERIC_ORD_DTYPES)
def test_numeric_only_ops_reject_non_numeric(op: BinOp, dtype: PolarsDataType) -> None:
    lf = values_of(dtype).to_frame().lazy()

    with pytest.raises(InvalidOperationError, match="requires a numeric input"):
        lf.select(op(pl.col("a"), labels=False)).collect_schema()


@pytest.mark.parametrize("op", BIN_OPS)
@pytest.mark.parametrize("value", [{"x": 1}, [1, 2]], ids=["struct", "list"])
def test_all_ops_reject_unorderable_input(op: BinOp, value: Any) -> None:
    lf = pl.LazyFrame({"a": [value, value]})

    with pytest.raises(
        InvalidOperationError, match=r"requires a(n orderable| numeric) input"
    ):
        lf.select(op(pl.col("a"), labels=False)).collect_schema()


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(lambda o, n: o.bin_intervals(n, labels=False), id="intervals"),
        pytest.param(lambda o, n: o.bin_quantiles(n, labels=False), id="quantiles"),
        pytest.param(lambda o, n: o.bin_ranks(n, labels=False), id="ranks"),
    ],
)
def test_zero_bins_raises(op: Callable[[pl.Expr, int], pl.Expr]) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    with pytest.raises(ComputeError, match="at least one bin"):
        lf.select(op(pl.col("a"), 0)).collect_schema()


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(lambda o, xs: o.bin_intervals(xs, labels=False), id="intervals"),
        pytest.param(lambda o, xs: o.bin_quantiles(xs, labels=False), id="quantiles"),
        pytest.param(lambda o, xs: o.bin_ranks(xs, labels=False), id="ranks"),
    ],
)
@pytest.mark.parametrize("args", [[0.6, 0.3], [0.5, 0.5]])
def test_not_ascending_raises(
    op: Callable[[pl.Expr, list[float]], pl.Expr], args: list[float]
) -> None:
    lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0]})

    with pytest.raises(ComputeError, match="strictly ascending"):
        lf.select(op(pl.col("a"), args)).collect_schema()


@pytest.mark.parametrize("dtype", [*NUMERIC_DTYPES, pl.String])
def test_bin_ranks_ties_are_stable(dtype: PolarsDataType) -> None:
    # Every value is identical, so only the sort's tie-breaking decides the bins. The
    # kernel asks for `maintain_order`, so ties must split in input order rather than
    # arbitrarily.
    n = 1000
    value = "x" if dtype == pl.String else 1
    s = pl.Series("a", [value] * n, dtype=dtype)

    result = s.bin_ranks(2, labels=False)

    expected = pl.Series("a", [0] * (n // 2) + [1] * (n // 2), dtype=pl.UInt32)
    assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", NUMERIC_INPUT_DTYPES)
def test_series_and_expr_agree(dtype: PolarsDataType) -> None:
    s = values_of(dtype)

    ops: list[BinOp] = [
        lambda o: o.bin_intervals(3, labels=LABELS),
        lambda o: o.bin_quantiles(3, labels=LABELS),
        lambda o: o.bin_ranks(3, labels=LABELS),
    ]
    for op in ops:
        eager = op(s)
        lazy = s.to_frame().select(op(pl.col("a"))).to_series()
        assert_series_equal(eager, lazy)


def test_decimal_is_supported_everywhere() -> None:
    # Decimal used to be rejected by the equal-width form, which gated on
    # `is_primitive_numeric` rather than `is_numeric`.
    s = pl.Series("a", [Decimal("1.50"), Decimal("2.50"), Decimal("3.50")])

    result = s.bin_intervals(2, labels=False, include_intervals=True)

    assert result.struct["left"].dtype == s.dtype
    assert result.struct["right"].to_list() == [Decimal("2.50"), None, None]
