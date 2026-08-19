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

/// The two width-specific steps equal-width binning needs: measuring the span between
/// two values, and stepping back out from `min` by an offset within that span.
///
/// Wrapping arithmetic on the raw two's-complement bits is exact here: `min <= max`
/// guarantees the true span fits the unsigned width, whereas a signed subtraction would
/// overflow from `i64::MIN` to `i64::MAX`. Widening to a common type first is not an
/// option either -- `as u128` sign-extends, and `i128` holds neither a `u128` value nor
/// an `i128` span -- and nothing in the numeric trait hierarchy names the unsigned type
/// of a given width, so the two signednesses are spelled out separately.
trait BinWidth: Copy {
    /// `self - min`, non-negative because `self` is the column's max.
    fn span_from(self, min: Self) -> u128;
    /// `self + offset`, where `offset` lies within the span and so cannot overflow, nor
    /// lose anything on the way back down to this width.
    fn offset_by(self, offset: u128) -> Self;
}

macro_rules! impl_bin_width {
    (signed: $($t:ty),* $(,)?) => {
        $(
            impl BinWidth for $t {
                fn span_from(self, min: Self) -> u128 {
                    // `cast_unsigned` is a bit reinterpretation at the same width, so
                    // unlike `as` it cannot silently truncate or sign-extend.
                    self.cast_unsigned().wrapping_sub(min.cast_unsigned()).into()
                }
                fn offset_by(self, offset: u128) -> Self {
                    self.cast_unsigned().wrapping_add(offset as _).cast_signed()
                }
            }
        )*
    };
    (unsigned: $($t:ty),* $(,)?) => {
        $(
            impl BinWidth for $t {
                fn span_from(self, min: Self) -> u128 {
                    self.wrapping_sub(min).into()
                }
                fn offset_by(self, offset: u128) -> Self {
                    self.wrapping_add(offset as _)
                }
            }
        )*
    };
}

impl_bin_width!(signed: i8, i16, i32, i64, i128);
impl_bin_width!(unsigned: u8, u16, u32, u64, u128);

/// Offsets from `min` of the `n_bins - 1` thresholds of equal-width bins over a span.
///
/// The exact breakpoint `i * span / n_bins` need not be an integer. For left-closed bins
/// the equivalent threshold is its ceiling; for right-closed bins it is its floor. So
/// carry the division as a quotient plus a running remainder: `offset` is the floor and
/// `error` is the numerator of the fraction still owed, which is non-zero exactly when
/// the ceiling is one higher.
fn uniform_threshold_offsets(
    span: u128,
    n_bins: usize,
    right_closed: bool,
) -> impl Iterator<Item = u128> {
    let n = n_bins as u128;
    let (step, remainder) = (span / n, span % n);
    let mut offset = 0;
    let mut error = 0;

    (1..n_bins).map(move |_| {
        offset += step;
        error += remainder;
        if error >= n {
            offset += 1;
            error -= n;
        }

        let round_up = !right_closed && error != 0;
        offset + u128::from(round_up)
    })
}

/// Representable thresholds for equal-width bins over `[min, max]`, in `N`'s own width.
fn uniform_integer_thresholds<N: BinWidth>(
    min: N,
    max: N,
    n_bins: usize,
    right_closed: bool,
) -> Vec<N> {
    uniform_threshold_offsets(max.span_from(min), n_bins, right_closed)
        .map(|offset| min.offset_by(offset))
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

#[cfg(test)]
mod test {
    use super::*;

    fn offsets(span: u128, n_bins: usize, right_closed: bool) -> Vec<u128> {
        uniform_threshold_offsets(span, n_bins, right_closed).collect()
    }

    #[test]
    fn exact_division_needs_no_rounding() {
        // 100 over 4 bins divides evenly, so both closures agree.
        assert_eq!(offsets(100, 4, false), [25, 50, 75]);
        assert_eq!(offsets(100, 4, true), [25, 50, 75]);
    }

    #[test]
    fn left_closed_takes_the_ceiling_and_right_closed_the_floor() {
        // The exact breakpoints over a span of 3 in 10 bins are 0.3, 0.6, ... 2.7.
        assert_eq!(offsets(3, 10, false), [1, 1, 1, 2, 2, 2, 3, 3, 3]);
        assert_eq!(offsets(3, 10, true), [0, 0, 0, 1, 1, 1, 2, 2, 2]);
    }

    #[test]
    fn thresholds_are_non_decreasing_and_stay_within_the_span() {
        for span in [0, 1, 7, 1000, u128::MAX] {
            for n_bins in [1, 2, 3, 7, 64, 1000] {
                for right_closed in [false, true] {
                    let got = offsets(span, n_bins, right_closed);
                    assert_eq!(got.len(), n_bins - 1);
                    assert!(got.windows(2).all(|w| w[0] <= w[1]));
                    assert!(got.iter().all(|o| *o <= span));
                }
            }
        }
    }

    #[test]
    fn a_span_of_zero_puts_every_threshold_on_min() {
        assert_eq!(offsets(0, 5, false), [0, 0, 0, 0]);
    }

    #[test]
    fn the_span_is_measured_with_wrapping_arithmetic() {
        // A signed subtraction would overflow on either of these.
        assert_eq!(i64::MAX.span_from(i64::MIN), u64::MAX as u128);
        assert_eq!(i128::MAX.span_from(i128::MIN), u128::MAX);
        assert_eq!(u128::MAX.span_from(0), u128::MAX);
        assert_eq!(i8::MIN.offset_by(u8::MAX as u128), i8::MAX);
        assert_eq!(i128::MIN.offset_by(u128::MAX), i128::MAX);
    }
}
