use polars_core::CHEAP_SERIES_HASH_LIMIT;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::*;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug, Hash)]
pub struct BinOptions {
    pub method: BinMethod,
    /// `None` corresponds to Python `labels=False`: emit the integer bin index.
    pub labels: Option<Vec<PlSmallStr>>,
    pub include_intervals: bool,
}

/// How the bins of a binning operation are delimited.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, PartialEq, Debug)]
pub enum BinMethod {
    /// Explicit breakpoints. Reconciled with the input during DSL to IR conversion:
    /// numeric breakpoints and a numeric input are both widened to their supertype,
    /// anything else is cast down to the input dtype. Either way, by the time this
    /// reaches the IR the dtype is the one both sides are compared in, which is also the
    /// dtype of the reported boundaries.
    ///
    /// `Series` carries its own `PartialEq`, serde and schema impls, so the only thing
    /// it does not supply for this enum is `Hash`, hand-written below.
    Intervals { breaks: Series, right_closed: bool },
    /// `n_bins` equal-width bins spanning `[min, max]`. The breakpoints are derived at
    /// runtime, in the input dtype.
    UniformIntervals { n_bins: usize, right_closed: bool },
    /// Quantile probabilities in `[0, 1]`, strictly ascending.
    Quantiles { probs: Vec<f64>, right_closed: bool },
    /// `n_bins` equiprobable bins. Kept distinct from [`BinMethod::Quantiles`] because
    /// `(i + 1)/n_bins` is generally not representable as an `f64`, so expanding it here
    /// would shift breakpoints by one element.
    UniformQuantiles { n_bins: usize, right_closed: bool },
    /// Cumulative fractions in `[0, 1]`, strictly ascending. There is no `right_closed`:
    /// bins are delimited by position in sorted order, so there is no value boundary to
    /// close on.
    Ranks { fractions: Vec<f64> },
    /// `n_bins` bins of near-equal size, the first `len % n_bins` of them one larger.
    /// Kept distinct from [`BinMethod::Ranks`] because that distribution is not
    /// expressible as uniform fractions.
    UniformRanks { n_bins: usize },
}

impl BinMethod {
    /// The explicit breakpoints, if this is [`BinMethod::Intervals`].
    pub fn breaks(&self) -> Option<&Series> {
        match self {
            Self::Intervals { breaks, .. } => Some(breaks),
            _ => None,
        }
    }

    /// The number of bins, and therefore the number of labels required.
    pub fn n_bins(&self) -> usize {
        match self {
            Self::Intervals { .. } => self.breaks().unwrap().len() + 1,
            Self::Quantiles { probs, .. } => probs.len() + 1,
            Self::Ranks { fractions } => fractions.len() + 1,
            Self::UniformIntervals { n_bins, .. }
            | Self::UniformQuantiles { n_bins, .. }
            | Self::UniformRanks { n_bins } => *n_bins,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Intervals { .. } | Self::UniformIntervals { .. } => "bin_intervals",
            Self::Quantiles { .. } | Self::UniformQuantiles { .. } => "bin_quantiles",
            Self::Ranks { .. } | Self::UniformRanks { .. } => "bin_ranks",
        }
    }

    /// Whether this form does arithmetic on the values, and so needs a numeric input.
    ///
    /// The quantile forms only gather values, but the spec restricts them to numerics
    /// anyway. Interval and rank binning accept anything with an order.
    pub fn requires_numeric_input(&self) -> bool {
        matches!(
            self,
            Self::UniformIntervals { .. } | Self::Quantiles { .. } | Self::UniformQuantiles { .. }
        )
    }

    /// The dtype of the `left`/`right` interval boundaries.
    pub fn bound_dtype(&self, input: &DataType) -> DataType {
        match self {
            // Reconciled with the input during DSL to IR conversion.
            Self::Intervals { .. } => self.breaks().unwrap().dtype().clone(),
            // Every other form reports boundaries in the input dtype: equal-width
            // breakpoints are computed in it, and the rest gather values out of the
            // column.
            _ => input.clone(),
        }
    }
}

impl Hash for BinMethod {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Intervals {
                breaks,
                right_closed,
            } => {
                // `Series` has no `Hash`. Hashing the leading values keeps this bounded
                // for a pathologically long breakpoint list while still honouring the
                // contract that equal breakpoints hash equally -- which mixing in the
                // buffer address, as `LiteralValue` does, would not.
                breaks.dtype().hash(state);
                breaks.len().hash(state);
                for av in breaks.iter().take(CHEAP_SERIES_HASH_LIMIT) {
                    av.hash(state);
                }
                right_closed.hash(state);
            },
            Self::UniformIntervals {
                n_bins,
                right_closed,
            }
            | Self::UniformQuantiles {
                n_bins,
                right_closed,
            } => {
                n_bins.hash(state);
                right_closed.hash(state);
            },
            Self::Quantiles {
                probs,
                right_closed,
            } => {
                bytemuck::cast_slice::<_, u64>(probs).hash(state);
                right_closed.hash(state);
            },
            Self::Ranks { fractions } => bytemuck::cast_slice::<_, u64>(fractions).hash(state),
            Self::UniformRanks { n_bins } => n_bins.hash(state),
        }
    }
}
