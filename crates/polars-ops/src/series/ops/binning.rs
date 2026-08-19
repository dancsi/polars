use arrow::compute::concatenate::concatenate_validities;
use polars_core::prelude::*;
use polars_core::with_match_physical_integer_polars_type;

use crate::series::{SearchSortedSide, search_sorted};

/// How [`bin_intervals`] delimits its bins.
pub enum IntervalSpec<'a> {
    /// Explicit breakpoints, ascending and in the same dtype as the input.
    Breaks(&'a Series),
    /// `n` equal-width bins spanning `[min, max]`.
    Count(usize),
}

/// How [`bin_quantiles`] and [`bin_ranks`] delimit their bins.
pub enum FractionSpec<'a> {
    /// Explicit ascending fractions in `[0, 1]`.
    Explicit(&'a [f64]),
    /// `n` bins of equal probability, or of equal size for [`bin_ranks`].
    Count(usize),
}

/// Assign every element of `s` to a bin delimited by `breaks`.
///
/// `breaks` must be ascending, free of nulls, and of exactly `s`'s dtype. The result is
/// in `0..=breaks.len()`, and null wherever `s` is null.
///
/// Bin `0` has no left bound, bin `i > 0` has left bound `breaks[i - 1]`, and the last
/// bin has no right bound. So a left-closed binning wants the number of breaks `<= x`
/// and a right-closed one the number of breaks `< x`, which is exactly the difference
/// between searching from the right and from the left.
///
/// NaN sorts above every other float under `TotalOrd`, which `search_sorted` uses, so it
/// lands in the last bin rather than becoming null. This differs from `cut`.
///
/// `Enum` is compared by physical code, which is its declaration order, while
/// `Categorical` is compared lexically -- `search_sorted` casts it to `String`. Both agree
/// with how `sort`, `arg_sort` and the comparison operators order those types, so bins are
/// consistent with sorting either way. The `Categorical` path does materialise strings, so
/// it is the one dtype here that carries an avoidable allocation.
pub fn bins_from_breaks(s: &Series, breaks: &Series, right_closed: bool) -> PolarsResult<IdxCa> {
    polars_ensure!(
        s.dtype() == breaks.dtype(),
        ComputeError: "binning expects the breakpoints to be of the input dtype `{}`, got `{}`",
        s.dtype(), breaks.dtype()
    );

    let side = if right_closed {
        SearchSortedSide::Left
    } else {
        SearchSortedSide::Right
    };

    let mut idx = search_sorted(breaks, s, side, false)?;
    idx.rename(s.name().clone());

    // `search_sorted` hands back a fully valid `IdxCa`: a null needle is given a
    // positional index rather than a null, so the input validity is reattached here.
    if s.null_count() > 0 {
        idx = IdxCa::from_vec_validity(
            s.name().clone(),
            idx.into_no_null_iter().collect(),
            concatenate_validities(s.chunks()),
        );
    }
    Ok(idx)
}

/// Sort options shared by every form that works from sorted order.
///
/// `maintain_order` matters: [`bin_ranks`] splits ties across adjacent bins by position,
/// so without a stable sort `[5, 5]` over two bins could come back as `[1, 0]`. It is
/// load-bearing for the String and Binary sort paths, which branch on it directly; the
/// numeric paths happen to be stable either way.
fn sort_options() -> SortOptions {
    SortOptions {
        descending: false,
        nulls_last: true,
        maintain_order: true,
        ..Default::default()
    }
}

/// The 0-based position of every element within the non-null values in sorted order,
/// null wherever `s` is null.
///
/// Adapted from `RankMethod::Ordinal` (`crate::series::ops::rank`), which is 1-based.
/// Duplicated rather than reused so that `cutqcut` need not enable the `rank` feature,
/// which would pull in `rand`.
fn ranks_from_sort_idx(s: &Series, sort_idx: &IdxCa) -> IdxCa {
    let mut out = vec![0 as IdxSize; s.len()];
    let mut rank: IdxSize = 0;
    for arr in sort_idx.downcast_iter() {
        for i in arr.values_iter() {
            out[*i as usize] = rank;
            rank += 1;
        }
    }
    IdxCa::from_vec_validity(s.name().clone(), out, concatenate_validities(s.chunks()))
}

/// Gather the value at each given position within the non-null values in sorted order.
///
/// Positions at or past the non-null count yield null, which is what a trailing empty
/// bin needs. Only `positions.len()` elements are materialised, so this costs a small
/// gather rather than a fully sorted copy of the column.
fn gather_at_sorted_positions(
    s: &Series,
    sort_idx: &IdxCa,
    positions: &[IdxSize],
) -> PolarsResult<Series> {
    let non_null_len = sort_idx.len() as IdxSize;

    let wanted: IdxCa = positions
        .iter()
        .map(|p| (*p < non_null_len).then_some(*p))
        .collect_ca(PlSmallStr::EMPTY);

    let phys_idx = sort_idx.take(&wanted)?;
    s.take(&phys_idx)
}

/// Order-preserving map between an integer and `u128`.
///
/// Equal-width breakpoints are computed in this domain so that one accumulator width
/// covers every integer type without overflow, and so that `max - min` cannot overflow
/// the input's own width (`i64::MIN` to `i64::MAX` does).
trait OrderedBits: Copy {
    fn to_ordered(self) -> u128;
    fn from_ordered(bits: u128) -> Self;
}

macro_rules! impl_ordered_bits {
    ($($signed:ty => $unsigned:ty),* $(,)?) => {
        $(
            impl OrderedBits for $signed {
                fn to_ordered(self) -> u128 {
                    // Flipping the sign bit maps the signed range onto the unsigned one
                    // while preserving order.
                    ((self as $unsigned) ^ (1 << (<$unsigned>::BITS - 1))) as u128
                }
                fn from_ordered(bits: u128) -> Self {
                    ((bits as $unsigned) ^ (1 << (<$unsigned>::BITS - 1))) as $signed
                }
            }
            impl OrderedBits for $unsigned {
                fn to_ordered(self) -> u128 {
                    self as u128
                }
                fn from_ordered(bits: u128) -> Self {
                    bits as $unsigned
                }
            }
        )*
    };
}

impl_ordered_bits!(i8 => u8, i16 => u16, i32 => u32, i64 => u64, i128 => u128);

/// Representable thresholds for equal-width bins over `[min, max]`.
///
/// The exact breakpoints need not be representable in an integer dtype. For left-closed
/// bins the equivalent threshold is their ceiling; for right-closed bins it is their
/// floor. Quotient/remainder accumulation computes those thresholds without overflowing
/// and without going through `f64`.
fn uniform_integer_thresholds<N: OrderedBits>(
    min: N,
    max: N,
    n_bins: usize,
    right_closed: bool,
) -> Vec<N> {
    let n = n_bins as u128;
    let min = min.to_ordered();
    let span = max.to_ordered() - min;
    let (step, remainder) = (span / n, span % n);
    let mut offset = 0;
    let mut error = 0;

    (1..n_bins)
        .map(|_| {
            offset += step;
            error += remainder;
            if error >= n {
                offset += 1;
                error -= n;
            }

            let round_up = !right_closed && error != 0;
            N::from_ordered(min + offset + u128::from(round_up))
        })
        .collect()
}

/// Equal-width breakpoints `min + (i + 1)/n_bins * (max - min)` for `0 <= i < n_bins - 1`,
/// in the input dtype.
///
/// Floats go through `f64` and are narrowed back afterwards; integers and `Decimal` (via
/// its `Int128` physical) use [`uniform_integer_thresholds`]. All null when there is no
/// usable `min`/`max`.
fn uniform_interval_breaks(s: &Series, n_bins: usize, right_closed: bool) -> PolarsResult<Series> {
    let n_breaks = n_bins.saturating_sub(1);
    let dtype = s.dtype();
    let all_null = || Ok(Series::full_null(s.name().clone(), n_breaks, dtype));

    if dtype.is_float() {
        let f = s.cast(&DataType::Float64)?;
        let (Some(min), Some(max)) = (f.min::<f64>()?, f.max::<f64>()?) else {
            return all_null();
        };
        let breaks: Vec<f64> = (1..=n_breaks)
            .map(|i| {
                let t = i as f64 / n_bins as f64;
                min * (1.0 - t) + max * t
            })
            .collect();
        return Float64Chunked::from_vec(s.name().clone(), breaks)
            .into_series()
            .cast(dtype);
    }

    let phys = s.to_physical_repr();
    let phys: &Series = phys.as_ref();
    let breaks = with_match_physical_integer_polars_type!(phys.dtype(), |$T| {
        let ca: &ChunkedArray<$T> = phys.as_ref().as_ref();
        let (Some(min), Some(max)) = (ca.min(), ca.max()) else {
            return all_null();
        };
        ChunkedArray::<$T>::from_vec(
            s.name().clone(),
            uniform_integer_thresholds(min, max, n_bins, right_closed),
        )
        .into_series()
    });

    // Reattach the logical dtype: a no-op for plain integers, and restores the precision
    // and scale for `Decimal`, whose breakpoints were computed on its i128 mantissas.
    // SAFETY: the values came out of the column's own physical range.
    unsafe { breaks.from_physical_unchecked(dtype) }
}

/// Breakpoint positions for explicit quantile probabilities: `floor(q * (len - 1))`,
/// matching `QuantileMethod::Lower`.
fn quantile_break_positions(non_null_len: usize, probs: &[f64]) -> Vec<IdxSize> {
    if non_null_len == 0 {
        return vec![0; probs.len()];
    }
    let span = (non_null_len - 1) as f64;
    probs
        .iter()
        .map(|q| (span * q).floor() as IdxSize)
        .collect()
}

/// Breakpoint positions for `n_bins` equiprobable bins, in *integer* arithmetic.
///
/// This must not be expressed by expanding to `(i + 1)/n_bins` probabilities and reusing
/// [`quantile_break_positions`]: `(i + 1)/n_bins` is generally not representable as an
/// `f64`, so `floor(q * (len - 1))` can land one element low. For example `7/10` is
/// `0.69999999999999996`, so with 91 values `0.7 * 90 == 62.999999999999996` floors to
/// 62 where the exact answer is 63.
fn quantile_break_positions_uniform(non_null_len: usize, n_bins: usize) -> Vec<IdxSize> {
    let n_breaks = n_bins.saturating_sub(1);
    if non_null_len == 0 {
        return vec![0; n_breaks];
    }
    let span = non_null_len - 1;
    (1..=n_breaks)
        .map(|i| ((i * span) / n_bins) as IdxSize)
        .collect()
}

/// Cut positions for `n_bins` bins of near-equal size.
///
/// Bin `i` receives `k + 1` elements while `i < len % n_bins` and `k` afterwards, so the
/// earlier bins are the larger ones: 14 elements over 4 bins gives 4 + 4 + 3 + 3.
fn rank_cut_positions_uniform(non_null_len: usize, n_bins: usize) -> Vec<IdxSize> {
    let k = non_null_len / n_bins;
    let r = non_null_len % n_bins;
    (1..n_bins).map(|i| (i * k + i.min(r)) as IdxSize).collect()
}

/// Cut positions for explicit cumulative fractions: `round(f * len)`, rounding half away
/// from zero so that, as in [`rank_cut_positions_uniform`], earlier bins are the larger.
fn rank_cut_positions_fractions(non_null_len: usize, fractions: &[f64]) -> Vec<IdxSize> {
    fractions
        .iter()
        .map(|f| (f * non_null_len as f64).round() as IdxSize)
        .collect()
}

/// Turn a column of bin indices into the final output.
///
/// Without labels the bins are returned as `UInt32`, with labels as an `Enum` over them.
/// When `breaks` is given the result is instead a struct of `bin`, `left` and `right`,
/// where `left` is null for the first bin and `right` null for the last. `breaks` holds
/// the `n_bins - 1` boundary values; for rank-based binning these are the gathered
/// boundary *values*, not the rank positions.
fn finish_bins(
    out_name: PlSmallStr,
    bin_idx: &IdxCa,
    n_bins: usize,
    breaks: Option<&Series>,
    labels: Option<&[PlSmallStr]>,
) -> PolarsResult<Series> {
    let bin = match labels {
        None => bin_idx.cast(&DataType::UInt32)?,
        Some(labels) => {
            polars_ensure!(
                labels.len() == n_bins,
                ShapeMismatch: "binning produces {} bins but got {} labels",
                n_bins, labels.len()
            );
            let fcats = FrozenCategories::new(labels.iter().map(|s| s.as_str()))?;
            let dtype = DataType::from_frozen_categories(fcats.clone());
            with_match_categorical_physical_type!(fcats.physical(), |$C| {
                let cats: Vec<<$C as PolarsCategoricalType>::Native> = bin_idx
                    .iter()
                    .map(|opt| {
                        <$C as PolarsCategoricalType>::Native::from_cat(
                            opt.unwrap_or(0) as CatSize,
                        )
                    })
                    .collect();
                let phys = ChunkedArray::<<$C as PolarsCategoricalType>::PolarsPhysical>::from_vec_validity(
                    PlSmallStr::EMPTY,
                    cats,
                    concatenate_validities(bin_idx.chunks()),
                );
                // SAFETY: every index is `< n_bins`, which is the number of frozen
                // categories, and the physical width was taken from `fcats` itself.
                unsafe {
                    CategoricalChunked::<$C>::from_cats_and_dtype_unchecked(phys, dtype)
                }
                .into_series()
            })
        },
    };

    let Some(breaks) = breaks else {
        return Ok(bin.with_name(out_name));
    };
    debug_assert_eq!(breaks.len() + 1, n_bins);

    // `left[i]` is `(null ++ breaks)[bin_idx[i]]` and `right[i]` is
    // `(breaks ++ null)[bin_idx[i]]`. A null index gathers to null, so null inputs and
    // the open ends of the first and last bin all fall out of the gather for free.
    let null = Series::full_null(PlSmallStr::EMPTY, 1, breaks.dtype());
    let mut left_lookup = null.clone();
    left_lookup.append(breaks)?;
    let mut right_lookup = breaks.clone().with_name(PlSmallStr::EMPTY);
    right_lookup.append(&null)?;

    let bin = bin.with_name(PlSmallStr::from_static("bin"));
    let left = left_lookup
        .take(bin_idx)?
        .with_name(PlSmallStr::from_static("left"));
    let right = right_lookup
        .take(bin_idx)?
        .with_name(PlSmallStr::from_static("right"));

    Ok(
        StructChunked::from_series(out_name, bin_idx.len(), [&bin, &left, &right].into_iter())?
            .into_series(),
    )
}

/// Output for an input with no usable values, keeping the dtype and struct shape stable.
fn empty_bins(
    s: &Series,
    n_bins: usize,
    bound_dtype: &DataType,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
) -> PolarsResult<Series> {
    let breaks = Series::full_null(PlSmallStr::EMPTY, n_bins - 1, bound_dtype);
    let bin_idx = IdxCa::full_null(s.name().clone(), s.len());
    finish_bins(
        s.name().clone(),
        &bin_idx,
        n_bins,
        include_intervals.then_some(&breaks),
        labels,
    )
}

pub fn bin_intervals(
    s: &Series,
    spec: IntervalSpec<'_>,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
    right_closed: bool,
) -> PolarsResult<Series> {
    let (s, breaks) = match spec {
        // The DSL to IR conversion already reconciled the two dtypes, so this cast is a
        // no-op unless a numeric input was widened to meet its breakpoints.
        IntervalSpec::Breaks(breaks) => (s.cast(breaks.dtype())?, breaks.clone()),
        IntervalSpec::Count(n_bins) => {
            polars_ensure!(
                n_bins >= 1,
                ComputeError: "`bin_intervals` requires at least one bin"
            );
            let breaks = uniform_interval_breaks(s, n_bins, right_closed)?;
            // No usable min/max: an empty, all-null or all-NaN column.
            if n_bins > 1 && breaks.null_count() == breaks.len() {
                return empty_bins(s, n_bins, s.dtype(), labels, include_intervals);
            }
            (s.clone(), breaks)
        },
    };

    let bin_idx = bins_from_breaks(&s, &breaks, right_closed)?;
    finish_bins(
        s.name().clone(),
        &bin_idx,
        breaks.len() + 1,
        include_intervals.then_some(&breaks),
        labels,
    )
}

pub fn bin_quantiles(
    s: &Series,
    spec: FractionSpec<'_>,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
    right_closed: bool,
) -> PolarsResult<Series> {
    let non_null_len = s.len() - s.null_count();
    let (positions, n_bins) = match spec {
        FractionSpec::Explicit(probs) => (
            quantile_break_positions(non_null_len, probs),
            probs.len() + 1,
        ),
        FractionSpec::Count(n_bins) => {
            polars_ensure!(
                n_bins >= 1,
                ComputeError: "`bin_quantiles` requires at least one bin"
            );
            (
                quantile_break_positions_uniform(non_null_len, n_bins),
                n_bins,
            )
        },
    };
    bin_at_positions(
        s,
        &positions,
        n_bins,
        labels,
        include_intervals,
        Some(right_closed),
    )
}

pub fn bin_ranks(
    s: &Series,
    spec: FractionSpec<'_>,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
) -> PolarsResult<Series> {
    let non_null_len = s.len() - s.null_count();
    let (positions, n_bins) = match spec {
        FractionSpec::Explicit(fractions) => (
            rank_cut_positions_fractions(non_null_len, fractions),
            fractions.len() + 1,
        ),
        FractionSpec::Count(n_bins) => {
            polars_ensure!(
                n_bins >= 1,
                ComputeError: "`bin_ranks` requires at least one bin"
            );
            (rank_cut_positions_uniform(non_null_len, n_bins), n_bins)
        },
    };
    bin_at_positions(s, &positions, n_bins, labels, include_intervals, None)
}

/// Shared tail of the four position-derived forms.
///
/// `right_closed` is `Some` for quantile binning, which assigns bins by comparing values
/// against the gathered breakpoints, and `None` for rank binning, which instead compares
/// each row's ordinal rank against the cut positions. The rank form therefore splits ties
/// across adjacent bins, which is the whole point of it -- and never needs the boundary
/// values unless they are actually reported.
fn bin_at_positions(
    s: &Series,
    positions: &[IdxSize],
    n_bins: usize,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
    right_closed: Option<bool>,
) -> PolarsResult<Series> {
    let non_null_len = s.len() - s.null_count();
    if non_null_len == 0 {
        return empty_bins(s, n_bins, s.dtype(), labels, include_intervals);
    }

    // One sort feeds both the ranks and the boundary gather.
    let sort_idx = s.arg_sort(sort_options()).slice(0, non_null_len);

    let (bin_idx, breaks) = match right_closed {
        Some(right_closed) => {
            let breaks = gather_at_sorted_positions(s, &sort_idx, positions)?;
            (bins_from_breaks(s, &breaks, right_closed)?, Some(breaks))
        },
        None => {
            let ranks = ranks_from_sort_idx(s, &sort_idx).into_series();
            let cuts = IdxCa::from_vec(PlSmallStr::EMPTY, positions.to_vec()).into_series();
            let bin_idx = bins_from_breaks(&ranks, &cuts, false)?;
            let breaks = include_intervals
                .then(|| gather_at_sorted_positions(s, &sort_idx, positions))
                .transpose()?;
            (bin_idx, breaks)
        },
    };

    // The value-based path needs the boundaries to bin at all, so drop them again unless
    // they are actually being reported; `finish_bins` reads `Some` as "emit the struct".
    let bounds = breaks.filter(|_| include_intervals);
    finish_bins(s.name().clone(), &bin_idx, n_bins, bounds.as_ref(), labels)
}
