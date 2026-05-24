import datetime as _dt
import math
from collections.abc import Callable, Sequence
from typing import Any, Literal

import polars as pl

from polarstation.frame_expr import FrameExpr

# ── dtype helpers ─────────────────────────────────────────────────────────────

_TEMPORAL_POLARS_TYPES = (pl.Date, pl.Datetime, pl.Duration, pl.Time)
_UNSIGNED_POLARS_TYPES = (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
_INTEGER_POLARS_TYPES = (
  pl.Int8, pl.Int16, pl.Int32, pl.Int64,
  pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
)


def _is_temporal(dtype: pl.DataType) -> bool:
  return isinstance(dtype, _TEMPORAL_POLARS_TYPES)


def _is_unsigned(dtype: pl.DataType) -> bool:
  return isinstance(dtype, _UNSIGNED_POLARS_TYPES)


def _is_integer(dtype: pl.DataType) -> bool:
  return isinstance(dtype, _INTEGER_POLARS_TYPES)


def _is_categorical(dtype: pl.DataType) -> bool:
  return isinstance(dtype, (pl.Enum, pl.Categorical, pl.String))


# ── numeric label helpers ─────────────────────────────────────────────────────


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


# ── temporal label helpers ────────────────────────────────────────────────────


def _make_phys_labels(
  breaks_phys: list[int],
  left_closed: bool,
  lo_phys: int,
  hi_phys: int,
  dtype: pl.DataType,
  fmt: str | Callable | None = None,
) -> list[str]:
  bounds = [lo_phys] + breaks_phys + [hi_phys]
  n = len(bounds) - 1
  result = []
  for i, (lo_, hi_) in enumerate(zip(bounds, bounds[1:])):
    v_lo = pl.Series([lo_], dtype=pl.Int64).cast(dtype)[0]
    v_hi = pl.Series([hi_], dtype=pl.Int64).cast(dtype)[0]
    lo_s = fmt(v_lo) if callable(fmt) else str(v_lo)
    hi_s = fmt(v_hi) if callable(fmt) else str(v_hi)
    if left_closed:
      rb = "]" if i == n - 1 else ")"
      result.append(f"[{lo_s}, {hi_s}{rb}")
    else:
      lb = "[" if i == 0 else "("
      result.append(f"{lb}{lo_s}, {hi_s}]")
  return result


def _timedelta_to_phys(td: _dt.timedelta, dtype: pl.DataType) -> int:
  ns = int(td.total_seconds() * 1e9)
  if isinstance(dtype, (pl.Datetime, pl.Duration)):
    tu = dtype.time_unit
    if tu == "ns":
      return ns
    if tu == "us":
      return ns // 1_000
    if tu == "ms":
      return ns // 1_000_000
    if tu == "s":
      return ns // 1_000_000_000
  if isinstance(dtype, pl.Date):
    return int(td.days)
  if isinstance(dtype, pl.Time):
    return ns
  raise TypeError(f"Cannot convert timedelta to physical units of {dtype}")


# ── integer label helpers ─────────────────────────────────────────────────────


def _make_int_labels(
  breaks: list[float],
  left_closed: bool,
  lo: float,
  hi: float,
  fmt: str | Callable[[float], str],
) -> list[str]:
  """Like _make_labels but uses fully-closed [a, b] notation for integer bins.

  Bins that touch ±∞ keep half-open notation; single-element bins use {x}.
  """
  bounds = [lo] + breaks + [hi]
  n = len(bounds) - 1
  result = []
  for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
    is_first = i == 0
    is_last = i == n - 1
    if left_closed:
      lo_val = a
      hi_val = b if is_last else float(math.ceil(b) - 1)
    else:
      lo_val = a if is_first else float(math.floor(a) + 1)
      hi_val = b
    lo_s = _fmt_bound(lo_val, fmt)
    hi_s = _fmt_bound(hi_val, fmt)
    lo_fin = math.isfinite(lo_val)
    hi_fin = math.isfinite(hi_val)
    if lo_fin and hi_fin and lo_val == hi_val:
      result.append(f"{{{lo_s}}}")
    elif lo_fin and hi_fin:
      result.append(f"[{lo_s}, {hi_s}]")
    elif not lo_fin:
      result.append(f"(-∞, {hi_s}]")
    else:
      result.append(f"[{lo_s}, +∞)")
  return result


# ── string/enum helpers ───────────────────────────────────────────────────────


def _get_categories(col: str, lf: pl.LazyFrame, dtype: pl.DataType) -> list[str]:
  """Ordered category list: Enum uses its defined order; String/Categorical sort alphabetically."""
  if isinstance(dtype, pl.Enum):
    return dtype.categories.to_list()
  return lf.select(pl.col(col).drop_nulls().unique().sort()).collect()[col].to_list()


def _make_enum_labels(
  breaks_phys: list[int],
  left_closed: bool,
  lo_phys: int,
  hi_phys: int,
  categories: list[str],
  fmt: Callable | None = None,
) -> list[str]:
  """Fully-closed label style for categorical data: [apple, cherry] or {apple}."""
  bounds = [lo_phys] + breaks_phys + [hi_phys]
  n = len(bounds) - 1
  result = []
  for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
    is_first = i == 0
    is_last = i == n - 1
    if left_closed:
      lo_idx, hi_idx = a, b if is_last else b - 1
    else:
      lo_idx, hi_idx = (a if is_first else a + 1), b
    lo_s = fmt(categories[lo_idx]) if callable(fmt) else categories[lo_idx]
    hi_s = fmt(categories[hi_idx]) if callable(fmt) else categories[hi_idx]
    if lo_idx == hi_idx:
      result.append(f"{{{lo_s}}}")
    else:
      result.append(f"[{lo_s}, {hi_s}]")
  return result


def _cut_enum_col(
  col: str,
  breaks_phys: list[int],
  labels: list[str],
  left_closed: bool,
  return_struct: bool,
  lo_phys: int,
  hi_phys: int,
  categories: list[str],
) -> pl.Expr:
  """Like _cut_phys_col but for string/enum columns; struct bounds are category name strings."""
  bounds_phys = [lo_phys] + breaks_phys + [hi_phys]
  enum_dtype = pl.Enum(categories)
  cat_expr = pl.col(col).cast(enum_dtype).to_physical().cut(
    [float(b) for b in breaks_phys], labels=labels, left_closed=left_closed
  )
  if return_struct:
    idx = cat_expr.to_physical()
    n_bins = len(breaks_phys) + 1
    lo_cats: list[str] = []
    hi_cats: list[str] = []
    for i in range(n_bins):
      is_first = i == 0
      is_last = i == n_bins - 1
      a, b = bounds_phys[i], bounds_phys[i + 1]
      if left_closed:
        lo_idx, hi_idx = a, b if is_last else b - 1
      else:
        lo_idx, hi_idx = (a if is_first else a + 1), b
      lo_cats.append(categories[lo_idx])
      hi_cats.append(categories[hi_idx])
    return pl.struct(
      lo=pl.lit(pl.Series(lo_cats, dtype=pl.String)).gather(idx),
      hi=pl.lit(pl.Series(hi_cats, dtype=pl.String)).gather(idx),
    ).alias(col)
  return cat_expr.alias(col)


def _enum_null_result(col: str, labels: Sequence[str] | None, return_struct: bool) -> pl.Expr:
  """Fallback for empty/all-null categorical columns."""
  labs = list(labels) if labels is not None else ["[-∞, ∞)"]
  if return_struct:
    return pl.struct(
      lo=pl.lit(None, dtype=pl.String), hi=pl.lit(None, dtype=pl.String)
    ).alias(col)
  return pl.lit(None).cast(pl.Enum(labs)).alias(col)


# ── cut helpers ───────────────────────────────────────────────────────────────


def _brk_n(xs: list, n: int, tail: str) -> list:
  """Interior breakpoints for equal-count bins (santoku brk_n algorithm).

  Walks sorted values placing a boundary every n elements, advancing past ties
  so that identical values are never split across bins.
  """
  breaks = []
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
  return_struct: bool,
  lo: float = float("-inf"),
  hi: float = float("inf"),
  discrete: bool = False,
) -> pl.Expr:
  bounds = [lo] + breaks + [hi]
  n = len(bounds) - 1
  labs = (
    list(labels) if labels is not None else _make_labels(breaks, left_closed, fmt, lo=lo, hi=hi)
  )
  cat_expr = pl.col(col).cut(breaks, labels=labs, left_closed=left_closed)
  if return_struct:
    idx = cat_expr.to_physical()
    if discrete:
      lo_vals: list[float] = []
      hi_vals: list[float] = []
      for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
        is_first = i == 0
        is_last = i == n - 1
        if left_closed:
          lo_val = a
          hi_val = b if is_last else float(math.ceil(b) - 1)
        else:
          lo_val = a if is_first else float(math.floor(a) + 1)
          hi_val = b
        lo_vals.append(lo_val)
        hi_vals.append(hi_val)
      lo_lit = pl.lit(pl.Series(lo_vals, dtype=pl.Float64))
      hi_lit = pl.lit(pl.Series(hi_vals, dtype=pl.Float64))
    else:
      lo_lit = pl.lit(pl.Series(bounds[:-1], dtype=pl.Float64))
      hi_lit = pl.lit(pl.Series(bounds[1:], dtype=pl.Float64))
    return pl.struct(lo=lo_lit.gather(idx), hi=hi_lit.gather(idx)).alias(col)
  return cat_expr.alias(col)


def _cut_phys_col(
  col: str,
  breaks_phys: list[int],
  labels: list[str],
  left_closed: bool,
  return_struct: bool,
  lo_phys: int,
  hi_phys: int,
  dtype: pl.DataType,
) -> pl.Expr:
  """Like _cut_col but operates on the physical (integer) representation of the column."""
  bounds_phys = [lo_phys] + breaks_phys + [hi_phys]
  cat_expr = pl.col(col).to_physical().cut(
    [float(b) for b in breaks_phys], labels=labels, left_closed=left_closed
  )
  if return_struct:
    idx = cat_expr.to_physical()
    lo_series = pl.Series(bounds_phys[:-1], dtype=pl.Int64).cast(dtype)
    hi_series = pl.Series(bounds_phys[1:], dtype=pl.Int64).cast(dtype)
    return pl.struct(
      lo=pl.lit(lo_series).gather(idx),
      hi=pl.lit(hi_series).gather(idx),
    ).alias(col)
  return cat_expr.alias(col)


# ── expression namespace ──────────────────────────────────────────────────────


@pl.api.register_expr_namespace("ps_chop")
class PolarstationChopExpression:
  def __init__(self, expr: pl.Expr) -> None:
    self._expr = expr

  def chop(
    self,
    breaks: Sequence[Any],
    labels: Sequence[str] | None = None,
    left_closed: bool = True,
    fmt: str | Callable | None = None,
    extend: bool = True,
    return_struct: bool = False,
  ) -> pl.Expr | FrameExpr:
    """Cut into intervals at explicit breakpoints.

    For numeric breaks with extend=True (default), labels are known statically
    and a plain Expr is returned. All other cases return a FrameExpr.

    Args:
      breaks: Interior breakpoints; sorted automatically. Accepts numeric or
              temporal Python values (datetime, date, timedelta, time).
      labels: Category labels (must be len(breaks) + 1). Auto-generated if omitted.
      left_closed: If True (default), intervals are [lo, hi); otherwise (lo, hi].
      fmt: Formatter for auto-generated labels. For numeric, a format-spec string
           (e.g. ".2f") or callable. For temporal, a callable or None (uses str()).
      extend: For numeric only — if True (default), outermost labels extend to
              -∞/+∞ (returns plain Expr). If False, uses data min/max (FrameExpr).
              Temporal breaks always use data bounds regardless of this setting.
      return_struct: If True, return a struct {lo, hi} instead of just the label.
    """
    all_numeric = all(isinstance(b, (int, float)) and not isinstance(b, bool) for b in breaks)

    if all_numeric:
      breaks_list = sorted(float(b) for b in breaks)
      effective_fmt: str | Callable[[float], str] = fmt if fmt is not None else "g"

      expr = self._expr

      def _numeric_resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
        cols = lf.select(expr).collect_schema().names()
        result = []
        for col in cols:
          dtype = lf.collect_schema()[col]
          if extend:
            label_lo = 0.0 if _is_unsigned(dtype) else float("-inf")
            label_hi = float("inf")
          else:
            stats = lf.select(
              pl.col(col).min().alias("mn"), pl.col(col).max().alias("mx")
            ).collect()
            raw_lo, raw_hi = stats["mn"][0], stats["mx"][0]
            if raw_lo is None or raw_hi is None:
              label_lo, label_hi = float("-inf"), float("inf")
            else:
              label_lo, label_hi = float(raw_lo), float(raw_hi)
          if labels is None:
            auto = (
              _make_int_labels(breaks_list, left_closed, label_lo, label_hi, effective_fmt)
              if _is_integer(dtype)
              else _make_labels(breaks_list, left_closed, effective_fmt, lo=label_lo, hi=label_hi)
            )
          else:
            auto = list(labels)
          result.append(
            _cut_col(col, breaks_list, auto, left_closed, effective_fmt, return_struct,
                     lo=label_lo, hi=label_hi, discrete=_is_integer(dtype))
          )
        return result

      return FrameExpr(expr, _numeric_resolver)

    all_strings = all(isinstance(b, str) for b in breaks)

    if all_strings:
      breaks_strings = list(breaks)
      expr = self._expr

      def _string_resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
        cols = lf.select(expr).collect_schema().names()
        result = []
        for col in cols:
          dtype = lf.collect_schema()[col]
          categories = _get_categories(col, lf, dtype)
          if not categories:
            result.append(_enum_null_result(col, labels, return_struct))
            continue
          try:
            breaks_phys = sorted(categories.index(b) for b in breaks_strings)
          except ValueError as e:
            raise ValueError(
              f"Break value not found in categories of column '{col}': {e}"
            ) from e
          phys_expr = pl.col(col).cast(pl.Enum(categories)).to_physical()
          stats = lf.select(
            phys_expr.min().alias("mn"), phys_expr.max().alias("mx")
          ).collect()
          mn_p, mx_p = stats["mn"][0], stats["mx"][0]
          if mn_p is None or mx_p is None:
            result.append(_enum_null_result(col, labels, return_struct))
            continue
          lo_phys, hi_phys = int(mn_p), int(mx_p)
          labs = (
            list(labels) if labels is not None
            else _make_enum_labels(breaks_phys, left_closed, lo_phys, hi_phys, categories, fmt)
          )
          result.append(
            _cut_enum_col(col, breaks_phys, labs, left_closed, return_struct,
                          lo_phys, hi_phys, categories)
          )
        return result

      return FrameExpr(expr, _string_resolver)

    # Temporal or other orderable breaks
    breaks_list_any = sorted(breaks)
    expr = self._expr

    def _temporal_resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
      cols = lf.select(expr).collect_schema().names()
      result = []
      for col in cols:
        dtype = lf.collect_schema()[col]
        phys_breaks = [
          int(pl.Series([b]).cast(dtype).to_physical()[0]) for b in breaks_list_any
        ]
        stats = lf.select(
          pl.col(col).min().alias("mn"), pl.col(col).max().alias("mx")
        ).collect()
        raw_lo, raw_hi = stats["mn"][0], stats["mx"][0]
        if raw_lo is None or raw_hi is None:
          lo_phys = phys_breaks[0] if phys_breaks else 0
          hi_phys = phys_breaks[-1] if phys_breaks else 0
        else:
          lo_phys = int(pl.Series([raw_lo]).cast(dtype).to_physical()[0])
          hi_phys = int(pl.Series([raw_hi]).cast(dtype).to_physical()[0])
        labs = (
          list(labels) if labels is not None
          else _make_phys_labels(phys_breaks, left_closed, lo_phys, hi_phys, dtype, fmt)
        )
        result.append(
          _cut_phys_col(col, phys_breaks, labs, left_closed, return_struct,
                        lo_phys, hi_phys, dtype)
        )
      return result

    return FrameExpr(expr, _temporal_resolver)

  def width(
    self,
    size: float | _dt.timedelta,
    start: Any | None = None,
    labels: Sequence[str] | None = None,
    left_closed: bool = True,
    fmt: str | Callable | None = None,
    extend: bool = False,
    return_struct: bool = False,
  ) -> FrameExpr:
    """Chop into equal-width bins of given size.

    Args:
      size: Width of each bin. For numeric columns, a number. For temporal
            columns, a datetime.timedelta.
      start: Left edge of the first bin. Defaults to the column minimum
             (or 0 for unsigned integer columns).
      labels: Category labels. Auto-generated if omitted.
      left_closed: If True (default), intervals are [lo, hi); otherwise (lo, hi].
      fmt: Formatter for auto-generated labels. For numeric, a format-spec string
           or callable; defaults to "g". For temporal, a callable or None (uses str()).
      extend: If True, extend outermost labels to -∞ / +∞. If False (default),
              the first label opens at the anchor and the last closes at
              anchor + n_bins * size.
      return_struct: If True, return a struct instead of just the label.
    """
    expr = self._expr
    numeric_fmt: str | Callable[[float], str] = fmt if fmt is not None else "g"

    def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
      cols = lf.select(expr).collect_schema().names()
      result = []
      for col in cols:
        dtype = lf.collect_schema()[col]

        if _is_temporal(dtype):
          if not isinstance(size, _dt.timedelta):
            raise TypeError(
              f"Column '{col}' has temporal dtype {dtype}; "
              f"'size' must be a datetime.timedelta, got {type(size).__name__}"
            )
          size_phys = _timedelta_to_phys(size, dtype)
          stats = lf.select(
            pl.col(col).min().alias("mn"), pl.col(col).max().alias("mx")
          ).collect()
          raw_lo, raw_hi = stats["mn"][0], stats["mx"][0]
          if raw_lo is None or raw_hi is None:
            result.append(
              _cut_phys_col(col, [], ["[-∞, ∞)"], left_closed, return_struct, 0, 0, dtype)
            )
            continue
          lo_phys = (
            int(pl.Series([start]).cast(dtype).to_physical()[0])
            if start is not None
            else int(pl.Series([raw_lo]).cast(dtype).to_physical()[0])
          )
          hi_phys = int(pl.Series([raw_hi]).cast(dtype).to_physical()[0])
          n_bins = max(1, math.ceil((hi_phys - lo_phys) / size_phys))
          breaks_phys = [lo_phys + size_phys * i for i in range(1, n_bins)]
          label_lo_phys = lo_phys
          label_hi_phys = lo_phys + n_bins * size_phys
          labs = (
            list(labels) if labels is not None
            else _make_phys_labels(breaks_phys, left_closed, label_lo_phys, label_hi_phys,
                                   dtype, fmt)
          )
          result.append(
            _cut_phys_col(col, breaks_phys, labs, left_closed, return_struct,
                          label_lo_phys, label_hi_phys, dtype)
          )
          continue

        if _is_categorical(dtype):
          if not isinstance(size, int):
            raise TypeError(
              f"Column '{col}' has categorical dtype {dtype}; "
              f"'size' must be an int (number of categories), got {type(size).__name__}"
            )
          categories = _get_categories(col, lf, dtype)
          if not categories:
            result.append(_enum_null_result(col, labels, return_struct))
            continue
          phys_expr = pl.col(col).cast(pl.Enum(categories)).to_physical()
          stats = lf.select(phys_expr.min().alias("mn"), phys_expr.max().alias("mx")).collect()
          mn_p, mx_p = stats["mn"][0], stats["mx"][0]
          if mn_p is None or mx_p is None:
            result.append(_enum_null_result(col, labels, return_struct))
            continue
          lo_phys = (
            categories.index(start) if start is not None else int(mn_p)
          )
          hi_data = int(mx_p)
          n_bins = max(1, math.ceil((hi_data - lo_phys) / size))
          breaks_phys = [lo_phys + size * i for i in range(1, n_bins)]
          if extend:
            label_lo_phys, label_hi_phys = 0, len(categories) - 1
          else:
            label_lo_phys = lo_phys
            label_hi_phys = min(lo_phys + n_bins * size, len(categories) - 1)
          labs = (
            list(labels) if labels is not None
            else _make_enum_labels(
              breaks_phys, left_closed, label_lo_phys, label_hi_phys, categories, fmt
            )
          )
          result.append(
            _cut_enum_col(col, breaks_phys, labs, left_closed, return_struct,
                          label_lo_phys, label_hi_phys, categories)
          )
          continue

        # Numeric path
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
          unsigned_start = _is_unsigned(dtype) and extend and start is None
          default_lo = 0.0 if unsigned_start else float(raw_lo_f)
          lo = float(start) if start is not None else default_lo
          hi = float(raw_hi_f)
          n_bins = max(1, math.ceil((hi - lo) / float(size)))
          breaks_list = [lo + float(size) * i for i in range(1, n_bins)]
          if extend:
            label_lo, label_hi = float("-inf"), float("inf")
          else:
            label_lo = lo
            # For integer dtypes, cap at data max so discrete labels don't overshoot
            label_hi = float(raw_hi_f) if _is_integer(dtype) else lo + n_bins * float(size)
        if labels is None:
          auto = (
            _make_int_labels(breaks_list, left_closed, label_lo, label_hi, numeric_fmt)
            if _is_integer(dtype)
            else _make_labels(breaks_list, left_closed, numeric_fmt, lo=label_lo, hi=label_hi)
          )
        else:
          auto = list(labels)
        result.append(
          _cut_col(col, breaks_list, auto, left_closed, numeric_fmt, return_struct,
                   lo=label_lo, hi=label_hi, discrete=_is_integer(dtype))
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
    return_struct: bool = False,
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
      fmt: Number formatter for auto-generated labels (numeric columns only).
      extend: If True, extend outermost labels to -∞ / +∞ (or 0 / +∞ for unsigned
              integers). If False (default), the first label opens at the data
              minimum and the last closes at the data maximum.
      return_struct: If True, return a struct instead of just the label.
    """
    expr = self._expr

    def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
      cols = lf.select(expr).collect_schema().names()
      result = []
      for col in cols:
        dtype = lf.collect_schema()[col]

        if _is_temporal(dtype):
          xs_phys = lf.select(
            pl.col(col).drop_nulls().sort().to_physical()
          ).collect()[col].to_list()
          breaks_phys = _brk_n(xs_phys, n, tail)
          if not xs_phys:
            labs = list(labels) if labels is not None else ["[-∞, ∞)"]
            result.append(_cut_phys_col(col, [], labs, left_closed, return_struct, 0, 0, dtype))
          else:
            lo_phys, hi_phys = xs_phys[0], xs_phys[-1]
            labs = (
              list(labels) if labels is not None
              else _make_phys_labels(breaks_phys, left_closed, lo_phys, hi_phys, dtype)
            )
            result.append(
              _cut_phys_col(col, breaks_phys, labs, left_closed, return_struct,
                            lo_phys, hi_phys, dtype)
            )
          continue

        if _is_categorical(dtype):
          categories = _get_categories(col, lf, dtype)
          xs_phys = lf.select(
            pl.col(col).cast(pl.Enum(categories)).to_physical().drop_nulls().sort()
          ).collect()[col].to_list()
          breaks_phys = _brk_n(xs_phys, n, tail)
          if not xs_phys:
            result.append(_enum_null_result(col, labels, return_struct))
          else:
            if extend:
              lo_phys, hi_phys = 0, len(categories) - 1
            else:
              lo_phys, hi_phys = xs_phys[0], xs_phys[-1]
            labs = (
              list(labels) if labels is not None
              else _make_enum_labels(breaks_phys, left_closed, lo_phys, hi_phys, categories)
            )
            result.append(
              _cut_enum_col(col, breaks_phys, labs, left_closed, return_struct,
                            lo_phys, hi_phys, categories)
            )
          continue

        xs = lf.select(pl.col(col).drop_nulls().sort()).collect()[col].to_list()
        xs_f = [float(v) for v in xs]
        breaks_list = _brk_n(xs_f, n, tail)
        if not xs_f:
          label_lo, label_hi = float("-inf"), float("inf")
        elif extend:
          label_lo = 0.0 if _is_unsigned(dtype) else float("-inf")
          label_hi = float("inf")
        else:
          label_lo, label_hi = xs_f[0], xs_f[-1]
        if labels is None:
          auto = (
            _make_int_labels(breaks_list, left_closed, label_lo, label_hi, fmt)
            if _is_integer(dtype)
            else _make_labels(breaks_list, left_closed, fmt, lo=label_lo, hi=label_hi)
          )
        else:
          auto = list(labels)
        result.append(
          _cut_col(col, breaks_list, auto, left_closed, fmt, return_struct,
                   lo=label_lo, hi=label_hi, discrete=_is_integer(dtype))
        )
      return result

    return FrameExpr(expr, resolver)

  def n_groups(
    self,
    k: int,
    labels: Sequence[str] | None = None,
    left_closed: bool = True,
    fmt: str | Callable | None = None,
    raw: bool = True,
    extend: bool = False,
    return_struct: bool = False,
  ) -> FrameExpr:
    """Chop into k equal-count groups (by quantile boundaries).

    Args:
      k: Number of groups.
      labels: Category labels (must be k). Auto-generated if omitted.
      left_closed: If True (default), intervals are [lo, hi); otherwise (lo, hi].
      fmt: Formatter for auto-generated labels. For numeric, defaults to "g" when
           raw=True and ".0%" when raw=False. For temporal, a callable or None.
      raw: If True (default), label with the actual break values. If False, use
           percentage labels (e.g. [0%, 25%)). Ignored for temporal columns.
      extend: If True, extend outermost labels to -∞ / +∞ (only affects numeric
              raw=True). Default False. For unsigned columns, lower bound is 0.
      return_struct: If True, return a struct instead of just the label.
    """
    expr = self._expr
    probs_sorted = [i / k for i in range(1, k)]
    numeric_fmt: str | Callable[[float], str] = (
      fmt if fmt is not None else ("g" if raw else ".0%")
    )

    def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
      cols = lf.select(expr).collect_schema().names()
      result = []
      for col in cols:
        dtype = lf.collect_schema()[col]

        if _is_temporal(dtype):
          q_df = lf.select(
            [pl.col(col).to_physical().quantile(p, interpolation="nearest").alias(f"__q{i}__")
             for i, p in enumerate(probs_sorted)]
            + [pl.col(col).to_physical().min().alias("__mn__"),
               pl.col(col).to_physical().max().alias("__mx__")]
          ).collect()
          seen: set[int] = set()
          breaks_phys: list[int] = []
          for i in range(len(probs_sorted)):
            v = q_df[f"__q{i}__"][0]
            if v is not None:
              v_int = int(v)
              if v_int not in seen:
                seen.add(v_int)
                breaks_phys.append(v_int)
          mn_p = q_df["__mn__"][0]
          mx_p = q_df["__mx__"][0]
          lo_phys = int(mn_p) if mn_p is not None else 0
          hi_phys = int(mx_p) if mx_p is not None else 0
          labs = (
            list(labels) if labels is not None
            else _make_phys_labels(breaks_phys, left_closed, lo_phys, hi_phys, dtype, fmt)
          )
          result.append(
            _cut_phys_col(col, breaks_phys, labs, left_closed, return_struct,
                          lo_phys, hi_phys, dtype)
          )
          continue

        if _is_categorical(dtype):
          categories = _get_categories(col, lf, dtype)
          if not categories:
            result.append(_enum_null_result(col, labels, return_struct))
            continue
          phys_expr = pl.col(col).cast(pl.Enum(categories)).to_physical()
          q_df = lf.select(
            [phys_expr.quantile(p, interpolation="nearest").alias(f"__q{i}__")
             for i, p in enumerate(probs_sorted)]
            + [phys_expr.min().alias("__mn__"), phys_expr.max().alias("__mx__")]
          ).collect()
          seen_e: set[int] = set()
          breaks_phys_e: list[int] = []
          for i in range(len(probs_sorted)):
            v = q_df[f"__q{i}__"][0]
            if v is not None:
              v_int = int(v)
              if v_int not in seen_e:
                seen_e.add(v_int)
                breaks_phys_e.append(v_int)
          mn_p = q_df["__mn__"][0]
          mx_p = q_df["__mx__"][0]
          if mn_p is None or mx_p is None:
            result.append(_enum_null_result(col, labels, return_struct))
            continue
          if extend:
            lo_phys, hi_phys = 0, len(categories) - 1
          else:
            lo_phys, hi_phys = int(mn_p), int(mx_p)
          labs = (
            list(labels) if labels is not None
            else _make_enum_labels(
              breaks_phys_e, left_closed, lo_phys, hi_phys, categories, fmt
            )
          )
          result.append(
            _cut_enum_col(col, breaks_phys_e, labs, left_closed, return_struct,
                          lo_phys, hi_phys, categories)
          )
          continue

        q_df = lf.select(
          [pl.col(col).quantile(p, interpolation="linear").alias(f"__q{i}__")
           for i, p in enumerate(probs_sorted)]
          + [pl.col(col).min().alias("__mn__"), pl.col(col).max().alias("__mx__")]
        ).collect()
        seen_f: set[float] = set()
        breaks_list: list[float] = []
        kept_probs: list[float] = []
        for i, p in enumerate(probs_sorted):
          v = q_df[f"__q{i}__"][0]
          if v is not None:
            v = float(v)
            if v not in seen_f:
              seen_f.add(v)
              breaks_list.append(v)
              kept_probs.append(p)
        mn = q_df["__mn__"][0]
        mx = q_df["__mx__"][0]
        if mn is None or mx is None:
          bound_lo, bound_hi = float("-inf"), float("inf")
        elif extend:
          bound_lo = 0.0 if _is_unsigned(dtype) else float("-inf")
          bound_hi = float("inf")
        else:
          bound_lo, bound_hi = float(mn), float(mx)
        if raw:
          auto_labels: list[str] = (
            (
              _make_int_labels(breaks_list, left_closed, bound_lo, bound_hi, numeric_fmt)
              if _is_integer(dtype)
              else _make_labels(breaks_list, left_closed, numeric_fmt, lo=bound_lo, hi=bound_hi)
            )
            if labels is None else list(labels)
          )
        else:
          auto_labels = (
            _make_quantile_labels(kept_probs, left_closed, numeric_fmt)
            if labels is None else list(labels)
          )
        result.append(
          _cut_col(col, breaks_list, auto_labels, left_closed, numeric_fmt, return_struct,
                   lo=bound_lo, hi=bound_hi, discrete=_is_integer(dtype))
        )
      return result

    return FrameExpr(expr, resolver)

  def quantiles(
    self,
    probs: Sequence[float],
    labels: Sequence[str] | None = None,
    left_closed: bool = True,
    fmt: str | Callable | None = None,
    raw: bool = False,
    extend: bool = False,
    return_struct: bool = False,
  ) -> FrameExpr:
    """Chop at quantile boundaries.

    Args:
      probs: Quantile probabilities in (0, 1), e.g. [0.25, 0.5, 0.75] for quartiles.
      labels: Category labels (must be len(probs) + 1). Auto-generated if omitted.
      left_closed: If True (default), intervals are [lo, hi); otherwise (lo, hi].
      fmt: Formatter for auto-generated labels. For numeric, defaults to ".0%"
           (percentages) when raw=False and "g" when raw=True. For temporal, a
           callable or None (uses str()).
      raw: If True, label with the actual break values instead of percentages.
           Ignored for temporal columns (always uses actual values).
      extend: If True, extend outermost labels to -∞ / +∞ (only affects numeric
              raw=True). Default False. For unsigned columns, lower bound is 0.
      return_struct: If True, return a struct instead of just the label.
    """
    expr = self._expr
    probs_sorted = sorted(float(p) for p in probs)
    numeric_fmt: str | Callable[[float], str] = (
      fmt if fmt is not None else ("g" if raw else ".0%")
    )

    def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
      cols = lf.select(expr).collect_schema().names()
      result = []
      for col in cols:
        dtype = lf.collect_schema()[col]

        if _is_temporal(dtype):
          q_df = lf.select(
            [pl.col(col).to_physical().quantile(p, interpolation="nearest").alias(f"__q{i}__")
             for i, p in enumerate(probs_sorted)]
            + [pl.col(col).to_physical().min().alias("__mn__"),
               pl.col(col).to_physical().max().alias("__mx__")]
          ).collect()
          seen: set[int] = set()
          breaks_phys: list[int] = []
          for i in range(len(probs_sorted)):
            v = q_df[f"__q{i}__"][0]
            if v is not None:
              v_int = int(v)
              if v_int not in seen:
                seen.add(v_int)
                breaks_phys.append(v_int)
          mn_p = q_df["__mn__"][0]
          mx_p = q_df["__mx__"][0]
          lo_phys = int(mn_p) if mn_p is not None else 0
          hi_phys = int(mx_p) if mx_p is not None else 0
          labs = (
            list(labels) if labels is not None
            else _make_phys_labels(breaks_phys, left_closed, lo_phys, hi_phys, dtype, fmt)
          )
          result.append(
            _cut_phys_col(col, breaks_phys, labs, left_closed, return_struct,
                          lo_phys, hi_phys, dtype)
          )
          continue

        if _is_categorical(dtype):
          categories = _get_categories(col, lf, dtype)
          if not categories:
            result.append(_enum_null_result(col, labels, return_struct))
            continue
          phys_expr = pl.col(col).cast(pl.Enum(categories)).to_physical()
          q_df = lf.select(
            [phys_expr.quantile(p, interpolation="nearest").alias(f"__q{i}__")
             for i, p in enumerate(probs_sorted)]
            + [phys_expr.min().alias("__mn__"), phys_expr.max().alias("__mx__")]
          ).collect()
          seen_e: set[int] = set()
          breaks_phys_e: list[int] = []
          for i in range(len(probs_sorted)):
            v = q_df[f"__q{i}__"][0]
            if v is not None:
              v_int = int(v)
              if v_int not in seen_e:
                seen_e.add(v_int)
                breaks_phys_e.append(v_int)
          mn_p = q_df["__mn__"][0]
          mx_p = q_df["__mx__"][0]
          if mn_p is None or mx_p is None:
            result.append(_enum_null_result(col, labels, return_struct))
            continue
          if extend:
            lo_phys, hi_phys = 0, len(categories) - 1
          else:
            lo_phys, hi_phys = int(mn_p), int(mx_p)
          labs = (
            list(labels) if labels is not None
            else _make_enum_labels(
              breaks_phys_e, left_closed, lo_phys, hi_phys, categories, fmt
            )
          )
          result.append(
            _cut_enum_col(col, breaks_phys_e, labs, left_closed, return_struct,
                          lo_phys, hi_phys, categories)
          )
          continue

        q_df = lf.select(
          [pl.col(col).quantile(p, interpolation="linear").alias(f"__q{i}__")
           for i, p in enumerate(probs_sorted)]
          + [pl.col(col).min().alias("__mn__"), pl.col(col).max().alias("__mx__")]
        ).collect()
        seen_f: set[float] = set()
        breaks_list: list[float] = []
        kept_probs: list[float] = []
        for i, p in enumerate(probs_sorted):
          v = q_df[f"__q{i}__"][0]
          if v is not None:
            v = float(v)
            if v not in seen_f:
              seen_f.add(v)
              breaks_list.append(v)
              kept_probs.append(p)
        mn = q_df["__mn__"][0]
        mx = q_df["__mx__"][0]
        if mn is None or mx is None:
          bound_lo, bound_hi = float("-inf"), float("inf")
        elif extend:
          bound_lo = 0.0 if _is_unsigned(dtype) else float("-inf")
          bound_hi = float("inf")
        else:
          bound_lo, bound_hi = float(mn), float(mx)
        if raw:
          auto_labels: list[str] = (
            (
              _make_int_labels(breaks_list, left_closed, bound_lo, bound_hi, numeric_fmt)
              if _is_integer(dtype)
              else _make_labels(breaks_list, left_closed, numeric_fmt, lo=bound_lo, hi=bound_hi)
            )
            if labels is None else list(labels)
          )
        else:
          auto_labels = (
            _make_quantile_labels(kept_probs, left_closed, numeric_fmt)
            if labels is None else list(labels)
          )
        result.append(
          _cut_col(col, breaks_list, auto_labels, left_closed, numeric_fmt, return_struct,
                   lo=bound_lo, hi=bound_hi, discrete=_is_integer(dtype))
        )
      return result

    return FrameExpr(expr, resolver)