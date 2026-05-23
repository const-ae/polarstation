import math
from collections.abc import Callable, Sequence
from typing import Literal

import polars as pl

from polarstation.frame_expr import FrameExpr


def _fmt_bound(x: float, fmt: str | Callable[[float], str]) -> str:
  if x == float("-inf"):
    return "-∞"
  if x == float("inf"):
    return "∞"
  return fmt(x) if callable(fmt) else format(x, fmt)


def _make_labels(
  breaks: list[float],
  left_closed: bool,
  fmt: str | Callable[[float], str],
  lo: float = float("-inf"),
  hi: float = float("inf"),
) -> list[str]:
  bounds = [lo] + breaks + [hi]
  n = len(bounds) - 1
  result = []
  for i, (lo_, hi_) in enumerate(zip(bounds, bounds[1:])):
    if left_closed:
      rb = "]" if i == n - 1 and math.isfinite(hi_) else ")"
      result.append(f"[{_fmt_bound(lo_, fmt)}, {_fmt_bound(hi_, fmt)}{rb}")
    else:
      lb = "[" if i == 0 and math.isfinite(lo_) else "("
      result.append(f"{lb}{_fmt_bound(lo_, fmt)}, {_fmt_bound(hi_, fmt)}]")
  return result


def _fmt_pct(p: float, fmt: str | Callable[[float], str]) -> str:
  return fmt(p) if callable(fmt) else format(p, fmt)


def _make_quantile_labels(
  probs: list[float], left_closed: bool, fmt: str | Callable[[float], str]
) -> list[str]:
  bounds = [0.0] + probs + [1.0]
  labels = []
  for i, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
    lo_s = _fmt_pct(lo, fmt)
    hi_s = _fmt_pct(hi, fmt)
    is_last = i == len(bounds) - 2
    is_first = i == 0
    if left_closed:
      labels.append(f"[{lo_s}, {hi_s}]" if is_last else f"[{lo_s}, {hi_s})")
    else:
      labels.append(f"[{lo_s}, {hi_s}]" if is_first else f"({lo_s}, {hi_s}]")
  return labels


def _brk_n(xs: list[float], n: int, tail: str) -> list[float]:
  """Interior breakpoints for equal-count bins (santoku brk_n algorithm).

  Walks sorted values placing a boundary every n elements, advancing past ties
  so that identical values are never split across bins.
  """
  breaks: list[float] = []
  group_starts: list[int] = [0]
  i = 0
  while i + n < len(xs):
    next_start = i + n
    while next_start < len(xs) and xs[next_start] == xs[next_start - 1]:
      next_start += 1
    if next_start >= len(xs):
      break
    breaks.append(xs[next_start])
    group_starts.append(next_start)
    i = next_start
  if tail == "merge" and breaks:
    if len(xs) - group_starts[-1] < n:
      breaks.pop()
  return breaks


def _cut_col(
  col: str,
  breaks: list[float],
  labels: Sequence[str] | None,
  left_closed: bool,
  fmt: str | Callable[[float], str],
  include_breaks: bool,
  lo: float = float("-inf"),
  hi: float = float("inf"),
) -> pl.Expr:
  bounds = [lo] + breaks + [hi]
  labs = (
    list(labels) if labels is not None else _make_labels(breaks, left_closed, fmt, lo=lo, hi=hi)
  )
  cat_expr = pl.col(col).cut(breaks, labels=labs, left_closed=left_closed)
  if include_breaks:
    idx = cat_expr.to_physical()
    lo_lit = pl.lit(pl.Series(bounds[:-1], dtype=pl.Float64))
    hi_lit = pl.lit(pl.Series(bounds[1:], dtype=pl.Float64))
    return pl.struct(lo=lo_lit.gather(idx), hi=hi_lit.gather(idx)).alias(col)
  return cat_expr.alias(col)


@pl.api.register_expr_namespace("ps_chop")
class PolarstationChopExpression:
  def __init__(self, expr: pl.Expr) -> None:
    self._expr = expr

  def chop(
    self,
    breaks: Sequence[float],
    labels: Sequence[str] | None = None,
    left_closed: bool = True,
    fmt: str | Callable[[float], str] = "g",
    extend: bool = True,
    include_breaks: bool = False,
  ) -> pl.Expr | FrameExpr:
    """Cut into intervals at explicit breakpoints.

    When extend=True (default), all labels are known statically and a plain Expr is returned.
    When extend=False, the outermost labels use the data min/max as bounds,
    requiring a FrameExpr.

    Args:
      breaks: Interior breakpoints; sorted automatically.
      labels: Category labels (must be len(breaks) + 1). Auto-generated if omitted.
      left_closed: If True (default), intervals are [lo, hi); otherwise (lo, hi].
      fmt: Number formatter — a callable or a format-spec string (e.g. ".2f").
      extend: If True (default), outermost labels extend to -∞ / +∞ and a plain Expr is returned.
              If False, outermost labels use the data min/max (returns FrameExpr).
      include_breaks: If True, return a struct {lo, hi} instead of just the label.
    """
    breaks_list = sorted(float(b) for b in breaks)

    if extend:
      labs = list(labels) if labels is not None else _make_labels(breaks_list, left_closed, fmt)
      cat_expr = self._expr.cut(breaks_list, labels=labs, left_closed=left_closed)
      if include_breaks:
        bounds = [float("-inf")] + breaks_list + [float("inf")]
        idx = cat_expr.to_physical()
        lo_lit = pl.lit(pl.Series(bounds[:-1], dtype=pl.Float64))
        hi_lit = pl.lit(pl.Series(bounds[1:], dtype=pl.Float64))
        return pl.struct(lo=lo_lit.gather(idx), hi=hi_lit.gather(idx)).alias(
          self._expr.meta.output_name()
        )
      return cat_expr

    expr = self._expr

    def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
      cols = lf.select(expr).collect_schema().names()
      result = []
      for col in cols:
        stats = lf.select(pl.col(col).min().alias("mn"), pl.col(col).max().alias("mx")).collect()
        raw_lo, raw_hi = stats["mn"][0], stats["mx"][0]
        if raw_lo is None or raw_hi is None:
          label_lo, label_hi = float("-inf"), float("inf")
        else:
          label_lo, label_hi = float(raw_lo), float(raw_hi)
        result.append(
          _cut_col(col, breaks_list, labels, left_closed, fmt, include_breaks,
                   lo=label_lo, hi=label_hi)
        )
      return result

    return FrameExpr(expr, resolver)

  def width(
    self,
    size: float,
    start: float | None = None,
    labels: Sequence[str] | None = None,
    left_closed: bool = True,
    fmt: str | Callable[[float], str] = "g",
    extend: bool = False,
    include_breaks: bool = False,
  ) -> FrameExpr:
    """Chop into equal-width bins of given size.

    Args:
      size: Width of each bin.
      start: Left edge of the first bin. Defaults to the column minimum.
      labels: Category labels. Auto-generated if omitted.
      left_closed: If True (default), intervals are [lo, hi); otherwise (lo, hi].
      fmt: Number formatter for auto-generated labels forwarded to `format`.
      extend: If True, extend outermost labels to -∞ / +∞. If False (default),
              the first label opens at the anchor and the last closes at
              anchor + n_bins * size.
      include_breaks: If True, return a struct instead of just the label.
    """
    expr = self._expr

    def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
      cols = lf.select(expr).collect_schema().names()
      result = []
      for col in cols:
        finite = pl.col(col).filter(pl.col(col).is_finite())
        stats = lf.select(
          finite.min().alias("mn_f"),
          finite.max().alias("mx_f"),
        ).collect()
        raw_lo_f = stats["mn_f"][0]
        raw_hi_f = stats["mx_f"][0]
        if raw_lo_f is None or raw_hi_f is None:
          breaks_list: list[float] = []
          label_lo, label_hi = float("-inf"), float("inf")
        else:
          lo = float(start) if start is not None else float(raw_lo_f)
          hi = float(raw_hi_f)
          n_bins = max(1, math.ceil((hi - lo) / size))
          breaks_list = [lo + size * i for i in range(1, n_bins)]
          if extend:
            label_lo, label_hi = float("-inf"), float("inf")
          else:
            label_lo, label_hi = lo, lo + n_bins * size
        result.append(
          _cut_col(col, breaks_list, labels, left_closed, fmt, include_breaks,
                   lo=label_lo, hi=label_hi)
        )
      return result

    return FrameExpr(expr, resolver)

  def n_elements(
    self,
    n: int,
    tail: Literal["split", "merge"] = "split",
    labels: Sequence[str] | None = None,
    left_closed: bool = True,
    fmt: str | Callable[[float], str] = "g",
    extend: bool = False,
    include_breaks: bool = False,
  ) -> FrameExpr:
    """Chop into groups of n observations each.

    Boundaries are drawn after every nth element (sorted order). Ties are never
    split — the boundary advances to the next distinct value if needed.

    Args:
      n: Number of observations per group.
      tail: What to do when the total doesn't divide evenly. "split" (default) keeps
            the smaller final group; "merge" absorbs it into the preceding group.
      labels: Category labels. Auto-generated if omitted.
      left_closed: If True (default), intervals are [lo, hi); otherwise (lo, hi].
      fmt: Number formatter for auto-generated labels forwarded to `format`.
      extend: If True, extend outermost labels to -∞ / +∞. If False (default),
              the first label opens at the data minimum and the last closes at
              the data maximum.
      include_breaks: If True, return a struct instead of just the label.
    """
    expr = self._expr

    def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
      cols = lf.select(expr).collect_schema().names()
      result = []
      for col in cols:
        xs = lf.select(pl.col(col).drop_nulls().sort()).collect()[col].to_list()
        xs_f = [float(v) for v in xs]
        breaks_list = _brk_n(xs_f, n, tail)
        if extend or not xs_f:
          label_lo, label_hi = float("-inf"), float("inf")
        else:
          label_lo, label_hi = xs_f[0], xs_f[-1]
        result.append(
          _cut_col(col, breaks_list, labels, left_closed, fmt, include_breaks,
                   lo=label_lo, hi=label_hi)
        )
      return result

    return FrameExpr(expr, resolver)

  def n_groups(
    self,
    k: int,
    labels: Sequence[str] | None = None,
    left_closed: bool = True,
    fmt: str | Callable[[float], str] | None = None,
    raw: bool = True,
    extend: bool = False,
    include_breaks: bool = False,
  ) -> FrameExpr:
    """Chop into k equal-count groups (by quantile boundaries).

    Args:
      k: Number of groups.
      labels: Category labels (must be k). Auto-generated if omitted.
      left_closed: If True (default), intervals are [lo, hi); otherwise (lo, hi].
      fmt: Formatter for auto-generated labels. Defaults to "g" when raw=True,
           ".0%" when raw=False. Forwarded to `format`.
      raw: If True (default), label with the actual break values. If False, use
           percentage labels (e.g. [0%, 25%)).
      extend: If True, extend outermost labels to -∞ / +∞ (only affects raw=True).
              Default False.
      include_breaks: If True, return a struct instead of just the label.
    """
    expr = self._expr
    probs_sorted = [i / k for i in range(1, k)]
    effective_fmt: str | Callable[[float], str] = (
      fmt if fmt is not None else ("g" if raw else ".0%")
    )

    def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
      cols = lf.select(expr).collect_schema().names()
      result = []
      for col in cols:
        q_df = lf.select(
          [pl.col(col).quantile(p, interpolation="linear").alias(f"__q{i}__")
           for i, p in enumerate(probs_sorted)]
          + [pl.col(col).min().alias("__mn__"), pl.col(col).max().alias("__mx__")]
        ).collect()
        seen: set[float] = set()
        breaks_list: list[float] = []
        kept_probs: list[float] = []
        for i, p in enumerate(probs_sorted):
          v = q_df[f"__q{i}__"][0]
          if v is not None:
            v = float(v)
            if v not in seen:
              seen.add(v)
              breaks_list.append(v)
              kept_probs.append(p)
        mn = q_df["__mn__"][0]
        mx = q_df["__mx__"][0]
        if extend or mn is None or mx is None:
          bound_lo, bound_hi = float("-inf"), float("inf")
        else:
          bound_lo, bound_hi = float(mn), float(mx)
        if raw:
          auto_labels: list[str] = (
            _make_labels(breaks_list, left_closed, effective_fmt, lo=bound_lo, hi=bound_hi)
            if labels is None else list(labels)
          )
        else:
          auto_labels = (
            _make_quantile_labels(kept_probs, left_closed, effective_fmt)
            if labels is None else list(labels)
          )
        result.append(
          _cut_col(col, breaks_list, auto_labels, left_closed, effective_fmt, include_breaks,
                   lo=bound_lo, hi=bound_hi)
        )
      return result

    return FrameExpr(expr, resolver)

  def quantiles(
    self,
    probs: Sequence[float],
    labels: Sequence[str] | None = None,
    left_closed: bool = True,
    fmt: str | Callable[[float], str] | None = None,
    raw: bool = False,
    extend: bool = False,
    include_breaks: bool = False,
  ) -> FrameExpr:
    """Chop at quantile boundaries.

    Args:
      probs: Quantile probabilities in (0, 1), e.g. [0.25, 0.5, 0.75] for quartiles.
      labels: Category labels (must be len(probs) + 1). Auto-generated if omitted.
      left_closed: If True (default), intervals are [lo, hi); otherwise (lo, hi].
      fmt: Formatter for auto-generated labels. Defaults to ".0%" (percentages) when
           raw=False, and "g" (numeric values) when raw=True. Forwarded to `format`.
      raw: If True, label with the actual break values instead of percentages.
      extend: If True, extend outermost labels to -∞ / +∞ (only affects raw=True;
              percentage labels always span [0%, 100%]). Default False.
      include_breaks: If True, return a struct instead of just the label.
    """
    expr = self._expr
    probs_sorted = sorted(float(p) for p in probs)
    effective_fmt: str | Callable[[float], str] = (
      fmt if fmt is not None else ("g" if raw else ".0%")
    )

    def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
      cols = lf.select(expr).collect_schema().names()
      result = []
      for col in cols:
        q_df = lf.select(
          [pl.col(col).quantile(p, interpolation="linear").alias(f"__q{i}__")
           for i, p in enumerate(probs_sorted)]
          + [pl.col(col).min().alias("__mn__"), pl.col(col).max().alias("__mx__")]
        ).collect()
        seen: set[float] = set()
        breaks_list: list[float] = []
        kept_probs: list[float] = []
        for i, p in enumerate(probs_sorted):
          v = q_df[f"__q{i}__"][0]
          if v is not None:
            v = float(v)
            if v not in seen:
              seen.add(v)
              breaks_list.append(v)
              kept_probs.append(p)
        mn = q_df["__mn__"][0]
        mx = q_df["__mx__"][0]
        if extend or mn is None or mx is None:
          bound_lo, bound_hi = float("-inf"), float("inf")
        else:
          bound_lo, bound_hi = float(mn), float(mx)
        if raw:
          auto_labels: list[str] = (
            _make_labels(breaks_list, left_closed, effective_fmt, lo=bound_lo, hi=bound_hi)
            if labels is None else list(labels)
          )
        else:
          auto_labels = (
            _make_quantile_labels(kept_probs, left_closed, effective_fmt)
            if labels is None else list(labels)
          )
        result.append(
          _cut_col(col, breaks_list, auto_labels, left_closed, effective_fmt, include_breaks,
                   lo=bound_lo, hi=bound_hi)
        )
      return result

    return FrameExpr(expr, resolver)
