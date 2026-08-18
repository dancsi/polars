use arrow::compute::concatenate::concatenate_validities;
use polars_core::prelude::*;

use crate::series::{SearchSortedSide, search_sorted};

/// Categorical and Enum inputs are rejected for now.
///
/// `search_sorted` compares them by casting both sides to `String`, which is wrong for
/// `Enum` (whose order is the declaration order) and meaningless for a global
/// `Categorical` (insertion order). Rank binning would in fact order `Enum` correctly via
/// `arg_sort`, but accepting it there while rejecting it elsewhere would be a confusing
/// split, so the restriction is applied uniformly until all three forms can support it.
fn ensure_binnable(dtype: &DataType) -> PolarsResult<()> {
    polars_ensure!(
        !dtype.is_categorical() && !dtype.is_enum(),
        InvalidOperation: "binning is not supported for dtype `{}`", dtype
    );
    Ok(())
}

fn ascending() -> SortOptions {
    SortOptions {
        descending: false,
        nulls_last: true,
        ..Default::default()
    }
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
pub fn bins_from_breaks(s: &Series, breaks: &Series, right_closed: bool) -> PolarsResult<IdxCa> {
    ensure_binnable(s.dtype())?;
    debug_assert_eq!(s.dtype(), breaks.dtype());

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

/// The 0-based position of every element within the non-null values in sorted order,
/// null wherever `s` is null.
///
/// Adapted from `RankMethod::Ordinal` (`crate::series::ops::rank`), which is 1-based.
/// Duplicated rather than reused so that `cutqcut` need not enable the `rank` feature,
/// which would pull in `rand`.
pub fn ordinal_ranks_0based(s: &Series) -> PolarsResult<IdxCa> {
    let len = s.len();
    let null_count = s.null_count();
    let sort_idx = s.arg_sort(ascending()).slice(0, len - null_count);

    let mut out = vec![0 as IdxSize; len];
    let mut rank: IdxSize = 0;
    for arr in sort_idx.downcast_iter() {
        for i in arr.values_iter() {
            out[*i as usize] = rank;
            rank += 1;
        }
    }
    Ok(IdxCa::from_vec_validity(
        s.name().clone(),
        out,
        concatenate_validities(s.chunks()),
    ))
}

/// Gather the value at each given position within the non-null values in sorted order.
///
/// Positions at or past the non-null count yield null, which is what a trailing empty
/// bin needs. Only `positions.len()` elements are materialised, so this costs one
/// `arg_sort` plus a small gather rather than a fully sorted copy of the column.
pub fn breaks_at_sorted_positions(s: &Series, positions: &[IdxSize]) -> PolarsResult<Series> {
    let non_null_len = s.len() - s.null_count();
    let sort_idx = s.arg_sort(ascending()).slice(0, non_null_len);

    let wanted: IdxCa = positions
        .iter()
        .map(|p| (*p < non_null_len as IdxSize).then_some(*p))
        .collect_ca(PlSmallStr::EMPTY);

    let phys_idx = sort_idx.take(&wanted)?;
    s.take(&phys_idx)
}

/// Equal-width breakpoints `min + (i + 1)/n_bins * (max - min)` for `0 <= i < n_bins - 1`.
///
/// Always `Float64`: the arithmetic is generally non-integral, and truncating it back to
/// an integer input dtype could even collapse two breakpoints into one.
pub fn uniform_interval_breaks(s: &Series, n_bins: usize) -> PolarsResult<Series> {
    let n_breaks = n_bins.saturating_sub(1);
    let f = s.cast(&DataType::Float64)?;
    let (Some(min), Some(max)) = (f.min::<f64>()?, f.max::<f64>()?) else {
        return Ok(Series::full_null(
            s.name().clone(),
            n_breaks,
            &DataType::Float64,
        ));
    };
    let width = max - min;
    let breaks: Vec<f64> = (0..n_breaks)
        .map(|i| min + ((i + 1) as f64 / n_bins as f64) * width)
        .collect();
    Ok(Float64Chunked::from_vec(s.name().clone(), breaks).into_series())
}

/// Breakpoint positions for explicit quantile probabilities: `floor(q * (len - 1))`,
/// matching `QuantileMethod::Lower`.
pub fn quantile_break_positions(non_null_len: usize, probs: &[f64]) -> Vec<IdxSize> {
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
pub fn quantile_break_positions_uniform(non_null_len: usize, n_bins: usize) -> Vec<IdxSize> {
    let n_breaks = n_bins.saturating_sub(1);
    if non_null_len == 0 {
        return vec![0; n_breaks];
    }
    let span = non_null_len - 1;
    (0..n_breaks)
        .map(|i| (((i + 1) * span) / n_bins) as IdxSize)
        .collect()
}

/// Cut positions for `n_bins` bins of near-equal size.
///
/// Bin `i` receives `k + 1` elements while `i < len % n_bins` and `k` afterwards, so the
/// earlier bins are the larger ones: 14 elements over 4 bins gives 4 + 4 + 3 + 3.
pub fn rank_cut_positions_uniform(non_null_len: usize, n_bins: usize) -> Vec<IdxSize> {
    let k = non_null_len / n_bins;
    let r = non_null_len % n_bins;
    (0..n_bins.saturating_sub(1))
        .map(|i| ((i + 1) * k + (i + 1).min(r)) as IdxSize)
        .collect()
}

/// Cut positions for explicit cumulative fractions: `round(f * len)`, rounding half away
/// from zero so that, as in [`rank_cut_positions_uniform`], earlier bins are the larger.
pub fn rank_cut_positions_fractions(non_null_len: usize, fractions: &[f64]) -> Vec<IdxSize> {
    fractions
        .iter()
        .map(|f| (f * non_null_len as f64).round() as IdxSize)
        .collect()
}

/// Turn a column of bin indices into the final output.
///
/// Without labels the bins are returned as `UInt32`, with labels as an `Enum` over them.
/// If `include_intervals` the result is instead a struct of `bin`, `left` and `right`,
/// where `left` is null for the first bin and `right` null for the last.
///
/// `breaks` holds the `n_bins - 1` boundary values. For rank-based binning these are the
/// gathered boundary *values*, not the rank positions.
pub fn finish_bins(
    out_name: PlSmallStr,
    bin_idx: &IdxCa,
    breaks: &Series,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
) -> PolarsResult<Series> {
    let n_bins = breaks.len() + 1;

    let bin = match labels {
        None => bin_idx.cast(&DataType::UInt32)?,
        Some(labels) => {
            polars_ensure!(
                labels.len() == n_bins,
                ShapeMismatch: "binning into {} bins requires {} labels, got {}",
                n_bins, n_bins, labels.len()
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

    if !include_intervals {
        return Ok(bin.with_name(out_name));
    }

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
    let breaks = Series::full_null(s.name().clone(), n_bins - 1, bound_dtype);
    let bin_idx = IdxCa::full_null(s.name().clone(), s.len());
    finish_bins(
        s.name().clone(),
        &bin_idx,
        &breaks,
        labels,
        include_intervals,
    )
}

pub fn bin_intervals(
    s: &Series,
    breaks: &Series,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
    right_closed: bool,
) -> PolarsResult<Series> {
    let bin_idx = bins_from_breaks(s, breaks, right_closed)?;
    finish_bins(
        s.name().clone(),
        &bin_idx,
        breaks,
        labels,
        include_intervals,
    )
}

pub fn bin_intervals_uniform(
    s: &Series,
    n_bins: usize,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
    right_closed: bool,
) -> PolarsResult<Series> {
    // Equal-width breakpoints need arithmetic, so this form works and reports in f64.
    let f = s.cast(&DataType::Float64)?;
    let breaks = uniform_interval_breaks(&f, n_bins)?;
    if breaks.null_count() == breaks.len() && n_bins > 1 {
        return empty_bins(s, n_bins, &DataType::Float64, labels, include_intervals);
    }
    let bin_idx = bins_from_breaks(&f, &breaks, right_closed)?;
    finish_bins(
        s.name().clone(),
        &bin_idx,
        &breaks,
        labels,
        include_intervals,
    )
}

pub fn bin_quantiles(
    s: &Series,
    probs: &[f64],
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
    right_closed: bool,
) -> PolarsResult<Series> {
    let positions = quantile_break_positions(s.len() - s.null_count(), probs);
    bin_at_positions(
        s,
        &positions,
        probs.len() + 1,
        labels,
        include_intervals,
        Some(right_closed),
    )
}

pub fn bin_quantiles_uniform(
    s: &Series,
    n_bins: usize,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
    right_closed: bool,
) -> PolarsResult<Series> {
    let positions = quantile_break_positions_uniform(s.len() - s.null_count(), n_bins);
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
    fractions: &[f64],
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
) -> PolarsResult<Series> {
    let positions = rank_cut_positions_fractions(s.len() - s.null_count(), fractions);
    bin_at_positions(
        s,
        &positions,
        fractions.len() + 1,
        labels,
        include_intervals,
        None,
    )
}

pub fn bin_ranks_uniform(
    s: &Series,
    n_bins: usize,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
) -> PolarsResult<Series> {
    let positions = rank_cut_positions_uniform(s.len() - s.null_count(), n_bins);
    bin_at_positions(s, &positions, n_bins, labels, include_intervals, None)
}

/// Shared tail of the four position-derived forms.
///
/// `right_closed` is `Some` for quantile binning, which assigns bins by comparing values
/// against the gathered breakpoints, and `None` for rank binning, which instead compares
/// each row's ordinal rank against the cut positions. The rank form therefore splits ties
/// across adjacent bins, which is the whole point of it.
fn bin_at_positions(
    s: &Series,
    positions: &[IdxSize],
    n_bins: usize,
    labels: Option<&[PlSmallStr]>,
    include_intervals: bool,
    right_closed: Option<bool>,
) -> PolarsResult<Series> {
    // Rank binning compares ranks rather than values, so it never reaches the check in
    // `bins_from_breaks`.
    ensure_binnable(s.dtype())?;

    if s.len() == s.null_count() {
        return empty_bins(s, n_bins, s.dtype(), labels, include_intervals);
    }

    // Boundary values for `left`/`right`, always in the input dtype.
    let breaks = breaks_at_sorted_positions(s, positions)?;

    let bin_idx = match right_closed {
        Some(right_closed) => bins_from_breaks(s, &breaks, right_closed)?,
        None => {
            let ranks = ordinal_ranks_0based(s)?.into_series();
            let cuts = IdxCa::from_vec(PlSmallStr::EMPTY, positions.to_vec()).into_series();
            bins_from_breaks(&ranks, &cuts, false)?
        },
    };

    finish_bins(
        s.name().clone(),
        &bin_idx,
        &breaks,
        labels,
        include_intervals,
    )
}
