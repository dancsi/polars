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
    /// Explicit breakpoints, held as one `List` value. Cast to the input dtype during
    /// DSL to IR conversion, so by the time this reaches the IR the inner dtype is the
    /// input dtype.
    Intervals { breaks: Scalar, right_closed: bool },
    /// `n_bins` equal-width bins spanning `[min, max]`. The breakpoints are derived at
    /// runtime and are always `Float64`, the one case where the interval boundaries do
    /// not carry the input dtype.
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
    /// Bin by explicit breakpoints.
    ///
    /// The breakpoints are wrapped into a single `List` value so that the payload
    /// inherits [`Scalar`]'s `Hash`, serde and schema impls.
    pub fn intervals(breaks: Series, right_closed: bool) -> Self {
        let dtype = DataType::List(Box::new(breaks.dtype().clone()));
        Self::Intervals {
            breaks: Scalar::new(dtype, AnyValue::List(breaks)),
            right_closed,
        }
    }

    /// The explicit breakpoints, if this is [`BinMethod::Intervals`].
    pub fn breaks(&self) -> Option<&Series> {
        match self {
            Self::Intervals { breaks, .. } => match breaks.value() {
                AnyValue::List(s) => Some(s),
                _ => unreachable!("bin_intervals breakpoints must be a List value"),
            },
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

    /// Whether the interval boundaries are reported in `Float64` rather than in the
    /// input dtype.
    pub fn has_float_bounds(&self) -> bool {
        matches!(self, Self::UniformIntervals { .. })
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
                breaks.hash(state);
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
