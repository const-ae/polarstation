import datetime as _dt
import math
from collections.abc import Callable, Sequence
from typing import Any, Literal

import polars as pl

from polarstation.frame_expr import FrameExpr, resolve_across_columns

# ── dtype helpers ─────────────────────────────────────────────────────────────

_TEMPORAL_POLARS_TYPES = (pl.Date, pl.Datetime, pl.Duration, pl.Time)
_UNSIGNED_POLARS_TYPES = (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
_INTEGER_POLARS_TYPES = (
  pl.Int8,
  pl.Int16,
  pl.Int32,
  pl.Int64,
  pl.UInt8,
  pl.UInt16,
  pl.UInt32,
  pl.UInt64,
)
_FLOAT_POLARS_TYPES = (pl.Float32, pl.Float64)
_NUMERIC_POLARS_TYPES = _INTEGER_POLARS_TYPES + _FLOAT_POLARS_TYPES
_CATEGORICAL_POLARS_TYPES = (pl.Enum, pl.Categorical, pl.String)
_TEMPORAL_PYTHON_TYPES = (_dt.date, _dt.timedelta, _dt.time)  # date covers datetime


def _is_temporal(dtype: pl.DataType) -> bool:
  return isinstance(dtype, _TEMPORAL_POLARS_TYPES)


def _is_unsigned(dtype: pl.DataType) -> bool:
  return isinstance(dtype, _UNSIGNED_POLARS_TYPES)


def _is_integer(dtype: pl.DataType) -> bool:
  return isinstance(dtype, _INTEGER_POLARS_TYPES)


def _is_categorical(dtype: pl.DataType) -> bool:
  return isinstance(dtype, _CATEGORICAL_POLARS_TYPES)


def _assert_choppable_dtype(name: str, dtype: pl.DataType) -> None:
  """Raise TypeError for any dtype that no ps_chop function supports."""
  if not (
    isinstance(dtype, _NUMERIC_POLARS_TYPES) or _is_temporal(dtype) or _is_categorical(dtype)
  ):
    raise TypeError(
      f"Column '{name}' has unsupported dtype {dtype}; ps_chop functions require "
      f"a numeric, temporal, String, Categorical, or Enum column."
    )


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
  for i, (lo_, hi_) in enumerate(zip(bounds, bounds[1:], strict=False)):
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
  for i, (lo, hi) in enumerate(zip(bounds, bounds[1:], strict=False)):
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


def _make_physical_representation_labels(
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
  for i, (lo_, hi_) in enumerate(zip(bounds, bounds[1:], strict=False)):
    v_lo = pl.Series([lo_], dtype=dtype)[0]
    v_hi = pl.Series([hi_], dtype=dtype)[0]
    lo_s = fmt(v_lo) if callable(fmt) else str(v_lo)
    hi_s = fmt(v_hi) if callable(fmt) else str(v_hi)
    if left_closed:
      rb = "]" if i == n - 1 else ")"
      result.append(f"[{lo_s}, {hi_s}{rb}")
    else:
      lb = "[" if i == 0 else "("
      result.append(f"{lb}{lo_s}, {hi_s}]")
  return result


def _timedelta_to_physical_representation(td: _dt.timedelta, dtype: pl.DataType) -> int:
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
  for i, (a, b) in enumerate(zip(bounds, bounds[1:], strict=False)):
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


def _get_categories(lf: pl.LazyFrame, col_ref: pl.Expr, name: str, dtype: pl.DataType) -> list[str]:
  """Ordered category list: Enum uses its defined order; String/Categorical sort alphabetically."""
  if isinstance(dtype, pl.Enum):
    return dtype.categories.to_list()
  return lf.select(col_ref.drop_nulls().unique().sort()).collect()[name].to_list()


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
  for i, (a, b) in enumerate(zip(bounds, bounds[1:], strict=False)):
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


def _cut_enum_expr(
  col_ref: pl.Expr,
  name: str,
  breaks_phys: list[int],
  labels: list[str],
  left_closed: bool,
  return_struct: bool,
  lo_phys: int,
  hi_phys: int,
  categories: list[str],
) -> pl.Expr:
  """Like _cut_physical_representation_expr but for string/enum columns.

  Struct bounds are category name strings rather than numeric values.
  """
  bounds_phys = [lo_phys] + breaks_phys + [hi_phys]
  enum_dtype = pl.Enum(categories)
  cat_expr = (
    col_ref
    .cast(enum_dtype)
    .to_physical()
    .cut([float(b) for b in breaks_phys], labels=labels, left_closed=left_closed)
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
    ).alias(name)
  return cat_expr.alias(name)


def _enum_null_result(name: str, labels: Sequence[str] | None, return_struct: bool) -> pl.Expr:
  """Fallback for empty/all-null categorical columns."""
  labs = list(labels) if labels is not None else ["[-∞, ∞)"]
  if return_struct:
    return pl.struct(lo=pl.lit(None, dtype=pl.String), hi=pl.lit(None, dtype=pl.String)).alias(name)
  return pl.lit(None).cast(pl.Enum(labs)).alias(name)


# ── cut helpers ───────────────────────────────────────────────────────────────


def _brk_n(xs: list, n: int, tail: str, left_closed: bool = True) -> list:
  """Interior breakpoints for equal-count bins (santoku brk_n algorithm).

  For left-closed [lo, hi) intervals the break is placed at the first element
  of the next group; for right-closed (lo, hi] intervals it is placed at the
  last element of the current group. Ties are never split across bins.
  """
  breaks = []
  group_starts: list[int] = [0]
  i = 0
  if left_closed:
    while i + n < len(xs):
      next_start = i + n
      while next_start < len(xs) and xs[next_start] == xs[next_start - 1]:
        next_start += 1
      if next_start >= len(xs):
        break
      breaks.append(xs[next_start])
      group_starts.append(next_start)
      i = next_start
  else:
    # Break at the last element of each group; ties spill into current bin.
    while i + n <= len(xs):
      target_val = xs[i + n - 1]
      next_start = i + n
      while next_start < len(xs) and xs[next_start] == target_val:
        next_start += 1
      if next_start >= len(xs):
        break
      breaks.append(target_val)
      group_starts.append(next_start)
      i = next_start
  if tail == "merge" and breaks:
    if len(xs) - group_starts[-1] < n:
      breaks.pop()
  return breaks


def _cut_expr(
  col_ref: pl.Expr,
  name: str,
  breaks: list[float],
  labels: Sequence[str],
  left_closed: bool,
  return_struct: bool,
  lo: float = float("-inf"),
  hi: float = float("inf"),
  discrete: bool = False,
) -> pl.Expr:
  bounds = [lo] + breaks + [hi]
  n = len(bounds) - 1
  labs = list(labels)
  cat_expr = col_ref.cut(breaks, labels=labs, left_closed=left_closed)
  if return_struct:
    idx = cat_expr.to_physical()
    if discrete:
      lo_vals: list[float] = []
      hi_vals: list[float] = []
      for i, (a, b) in enumerate(zip(bounds, bounds[1:], strict=False)):
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
    return pl.struct(lo=lo_lit.gather(idx), hi=hi_lit.gather(idx)).alias(name)
  return cat_expr.alias(name)


def _cut_physical_representation_expr(
  col_ref: pl.Expr,
  name: str,
  breaks_phys: list[int],
  labels: list[str],
  left_closed: bool,
  return_struct: bool,
  lo_phys: int,
  hi_phys: int,
  dtype: pl.DataType,
) -> pl.Expr:
  """Like _cut_expr but operates on the physical (integer) representation of the column."""
  bounds_phys = [lo_phys] + breaks_phys + [hi_phys]
  cat_expr = (
    col_ref
    .to_physical()
    .cut([float(b) for b in breaks_phys], labels=labels, left_closed=left_closed)
  )
  if return_struct:
    idx = cat_expr.to_physical()
    lo_series = pl.Series(bounds_phys[:-1], dtype=pl.Int64).cast(dtype)
    hi_series = pl.Series(bounds_phys[1:], dtype=pl.Int64).cast(dtype)
    return pl.struct(
      lo=pl.lit(lo_series).gather(idx),
      hi=pl.lit(hi_series).gather(idx),
    ).alias(name)
  return cat_expr.alias(name)


# ── labeled cut wrappers ──────────────────────────────────────────────────────
# Each combines auto-label generation with the cut expression, eliminating the
# repeated (if labels is not None else _make_xxx(...)) + _cut_xxx(...) pattern.


def _labeled_physical_representation_cut(
  col_ref: pl.Expr,
  name: str,
  breaks_phys: list[int],
  lo_phys: int,
  hi_phys: int,
  dtype: pl.DataType,
  labels: Sequence[str] | None,
  left_closed: bool,
  fmt: Callable | None,
  return_struct: bool,
) -> pl.Expr:
  labs = (
    list(labels)
    if labels is not None
    else _make_physical_representation_labels(
      breaks_phys, left_closed, lo_phys, hi_phys, dtype, fmt
    )
  )
  return _cut_physical_representation_expr(
    col_ref, name, breaks_phys, labs, left_closed, return_struct, lo_phys, hi_phys, dtype
  )


def _labeled_enum_cut(
  col_ref: pl.Expr,
  name: str,
  breaks_phys: list[int],
  lo_phys: int,
  hi_phys: int,
  categories: list[str],
  labels: Sequence[str] | None,
  left_closed: bool,
  fmt: Callable | None,
  return_struct: bool,
) -> pl.Expr:
  labs = (
    list(labels)
    if labels is not None
    else _make_enum_labels(breaks_phys, left_closed, lo_phys, hi_phys, categories, fmt)
  )
  return _cut_enum_expr(
    col_ref, name, breaks_phys, labs, left_closed, return_struct, lo_phys, hi_phys, categories
  )


def _labeled_num_cut(
  col_ref: pl.Expr,
  name: str,
  breaks: list[float],
  lo: float,
  hi: float,
  dtype: pl.DataType,
  labels: Sequence[str] | None,
  left_closed: bool,
  fmt: str | Callable[[float], str],
  return_struct: bool,
) -> pl.Expr:
  if labels is not None:
    labs = list(labels)
  elif _is_integer(dtype):
    labs = _make_int_labels(breaks, left_closed, lo, hi, fmt)
  else:
    labs = _make_labels(breaks, left_closed, fmt, lo=lo, hi=hi)
  return _cut_expr(
    col_ref, name, breaks, labs, left_closed, return_struct,
    lo=lo, hi=hi, discrete=_is_integer(dtype)
  )


# ── quantile break helpers ────────────────────────────────────────────────────
# Deduplicate quantile values collected into a DataFrame with columns __q0__, __q1__, ...


def _deduplicate_int_breaks(q_df: pl.DataFrame, probs: list[float]) -> list[int]:
  """Unique integer break values from a quantile result DataFrame, preserving order."""
  seen: set[int] = set()
  breaks: list[int] = []
  for i in range(len(probs)):
    v = q_df[f"__q{i}__"][0]
    if v is not None:
      vi = int(v)
      if vi not in seen:
        seen.add(vi)
        breaks.append(vi)
  return breaks


def _deduplicate_float_breaks(
  q_df: pl.DataFrame, probs: list[float]
) -> tuple[list[float], list[float]]:
  """Unique float break values and their probabilities from a quantile result DataFrame."""
  seen: set[float] = set()
  breaks: list[float] = []
  kept_probs: list[float] = []
  for i, p in enumerate(probs):
    v = q_df[f"__q{i}__"][0]
    if v is not None:
      vf = float(v)
      if vf not in seen:
        seen.add(vf)
        breaks.append(vf)
        kept_probs.append(p)
  return breaks, kept_probs


# ── lazy frame query helpers ──────────────────────────────────────────────────


def _numerical_extremes(dtype: pl.DataType) -> tuple[float, float]:
  """Extend-to-infinity bounds: (0, ∞) for unsigned integers, (-∞, ∞) otherwise."""
  return (0.0 if _is_unsigned(dtype) else float("-inf")), float("inf")


def _min_max(lf: pl.LazyFrame, expr: pl.Expr) -> tuple[Any, Any]:
  """Collect min and max of an expression in one pass."""
  df = lf.select(expr.min().alias("__mn__"), expr.max().alias("__mx__")).collect()
  return df["__mn__"][0], df["__mx__"][0]


def _quantile_breaks_physical_representation(
  lf: pl.LazyFrame, probs: list[float], phys_expr: pl.Expr
) -> tuple[list[int], Any, Any]:
  """Collect deduplicated integer quantile breaks plus (min, max) from a physical expression."""
  q_df = lf.select(
    [phys_expr.quantile(p, interpolation="nearest").alias(f"__q{i}__") for i, p in enumerate(probs)]
    + [phys_expr.min().alias("__mn__"), phys_expr.max().alias("__mx__")]
  ).collect()
  return _deduplicate_int_breaks(q_df, probs), q_df["__mn__"][0], q_df["__mx__"][0]


def _quantile_breaks_float(
  lf: pl.LazyFrame, probs: list[float], col_ref: pl.Expr
) -> tuple[list[float], list[float], Any, Any]:
  """Collect deduplicated float quantile breaks, kept probabilities, and (min, max)."""
  q_df = lf.select(
    [
      col_ref.quantile(p, interpolation="linear").alias(f"__q{i}__")
      for i, p in enumerate(probs)
    ]
    + [col_ref.min().alias("__mn__"), col_ref.max().alias("__mx__")]
  ).collect()
  breaks, kept = _deduplicate_float_breaks(q_df, probs)
  return breaks, kept, q_df["__mn__"][0], q_df["__mx__"][0]



# ── chop ──────────────────────────────────────────────────────────────────────


def _chop_numeric(
  *,
  lf: pl.LazyFrame,
  name: str,
  col_ref: pl.Expr,
  dtype: pl.DataType,
  breaks: list,
  labels: Sequence[str] | None,
  left_closed: bool,
  fmt: str | Callable | None,
  extend: bool,
  return_struct: bool,
) -> pl.Expr:
  actual_fmt = fmt if fmt is not None else "g"
  if extend:
    label_lo, label_hi = _numerical_extremes(dtype)
  else:
    raw_lo, raw_hi = _min_max(lf, col_ref)
    if raw_lo is None or raw_hi is None:
      label_lo, label_hi = float("-inf"), float("inf")
    else:
      label_lo, label_hi = float(raw_lo), float(raw_hi)
  return _labeled_num_cut(
    col_ref, name, breaks, label_lo, label_hi, dtype, labels, left_closed, actual_fmt, return_struct
  )


def _chop_categorical(
  *,
  lf: pl.LazyFrame,
  name: str,
  col_ref: pl.Expr,
  dtype: pl.DataType,
  breaks: list,
  labels: Sequence[str] | None,
  left_closed: bool,
  fmt: str | Callable | None,
  extend: bool,
  return_struct: bool,
) -> pl.Expr:
  del extend
  if isinstance(dtype, pl.Enum):
    categories = dtype.categories.to_list()
    invalid = [b for b in breaks if b not in categories]
    if invalid:
      raise ValueError(
        f"Break value(s) {invalid!r} not in Enum categories of column '{name}'. "
        f"Valid categories: {categories}"
      )
  else:
    # String/Categorical: form the union of observed values and break values so
    # that a break can partition on a value not present in the data.
    observed = lf.select(col_ref.drop_nulls().unique().sort()).collect()[name].to_list()
    if not observed:
      return _enum_null_result(name, labels, return_struct)
    categories = sorted(set(observed) | set(breaks))
  breaks_phys = sorted(categories.index(b) for b in breaks)
  phys_expr = col_ref.cast(pl.Enum(categories)).to_physical()
  mn_p, mx_p = _min_max(lf, phys_expr)
  if mn_p is None or mx_p is None:
    return _enum_null_result(name, labels, return_struct)
  lo_phys = int(mn_p)
  # Clamp hi_phys upward so a break beyond all observed values is still within bounds.
  hi_phys = max(int(mx_p), breaks_phys[-1] if breaks_phys else int(mx_p))
  return _labeled_enum_cut(
    col_ref, name, breaks_phys, lo_phys, hi_phys, categories,
    labels, left_closed, fmt, return_struct
  )


def _chop_temporal(
  *,
  lf: pl.LazyFrame,
  name: str,
  col_ref: pl.Expr,
  dtype: pl.DataType,
  breaks: list,
  labels: Sequence[str] | None,
  left_closed: bool,
  fmt: str | Callable | None,
  extend: bool,
  return_struct: bool,
) -> pl.Expr:
  del extend
  phys_breaks = [int(pl.Series([b]).cast(dtype).to_physical()[0]) for b in breaks]
  raw_lo, raw_hi = _min_max(lf, col_ref)
  if raw_lo is None or raw_hi is None:
    lo_phys = phys_breaks[0] if phys_breaks else 0
    hi_phys = phys_breaks[-1] if phys_breaks else 0
  else:
    lo_phys = int(pl.Series([raw_lo]).cast(dtype).to_physical()[0])
    hi_phys = int(pl.Series([raw_hi]).cast(dtype).to_physical()[0])
  return _labeled_physical_representation_cut(
    col_ref, name, phys_breaks, lo_phys, hi_phys, dtype, labels, left_closed, fmt, return_struct
  )


# ── width ─────────────────────────────────────────────────────────────────────


def _width_temporal(
  *,
  lf: pl.LazyFrame,
  name: str,
  col_ref: pl.Expr,
  dtype: pl.DataType,
  size: _dt.timedelta | Any,
  start: Any | None,
  labels: Sequence[str] | None,
  left_closed: bool,
  fmt: str | Callable | None,
  extend: bool,
  return_struct: bool,
) -> pl.Expr:
  del extend
  if not isinstance(size, _dt.timedelta):
    raise TypeError(
      f"Column '{name}' has temporal dtype {dtype}; "
      f"'size' must be a datetime.timedelta, got {type(size).__name__}"
    )
  size_phys = _timedelta_to_physical_representation(size, dtype)
  raw_lo, raw_hi = _min_max(lf, col_ref)
  if raw_lo is None or raw_hi is None:
    return _cut_physical_representation_expr(
      col_ref, name, [], ["[-∞, ∞)"], left_closed, return_struct, 0, 0, dtype
    )
  lo_phys = (
    int(pl.Series([start]).cast(dtype).to_physical()[0])
    if start is not None
    else int(pl.Series([raw_lo]).cast(dtype).to_physical()[0])
  )
  hi_phys = int(pl.Series([raw_hi]).cast(dtype).to_physical()[0])
  n_bins = max(1, math.ceil((hi_phys - lo_phys) / size_phys))
  breaks_phys = [lo_phys + size_phys * i for i in range(1, n_bins)]
  label_hi_phys = lo_phys + n_bins * size_phys
  return _labeled_physical_representation_cut(
    col_ref, name, breaks_phys, lo_phys, label_hi_phys, dtype,
    labels, left_closed, fmt, return_struct
  )


def _width_categorical(
  *,
  lf: pl.LazyFrame,
  name: str,
  col_ref: pl.Expr,
  dtype: pl.DataType,
  size: int | Any,
  start: Any | None,
  labels: Sequence[str] | None,
  left_closed: bool,
  fmt: str | Callable | None,
  extend: bool,
  return_struct: bool,
) -> pl.Expr:
  if not isinstance(size, int):
    raise TypeError(
      f"Column '{name}' has categorical dtype {dtype}; "
      f"'size' must be an int (number of categories), got {type(size).__name__}"
    )
  categories = _get_categories(lf, col_ref, name, dtype)
  if not categories:
    return _enum_null_result(name, labels, return_struct)
  phys_expr = col_ref.cast(pl.Enum(categories)).to_physical()
  mn_p, mx_p = _min_max(lf, phys_expr)
  if mn_p is None or mx_p is None:
    return _enum_null_result(name, labels, return_struct)
  lo_phys = categories.index(start) if start is not None else int(mn_p)
  hi_data = int(mx_p)
  n_bins = max(1, math.ceil((hi_data - lo_phys) / size))
  breaks_phys = [lo_phys + size * i for i in range(1, n_bins)]
  if extend:
    label_lo_phys, label_hi_phys = 0, len(categories) - 1
  else:
    label_lo_phys = lo_phys
    label_hi_phys = min(lo_phys + n_bins * size, len(categories) - 1)
  return _labeled_enum_cut(
    col_ref,
    name,
    breaks_phys,
    label_lo_phys,
    label_hi_phys,
    categories,
    labels,
    left_closed,
    fmt,
    return_struct,
  )


def _width_numeric(
  *,
  lf: pl.LazyFrame,
  name: str,
  col_ref: pl.Expr,
  dtype: pl.DataType,
  size: float | Any,
  start: Any | None,
  labels: Sequence[str] | None,
  left_closed: bool,
  fmt: str | Callable | None,
  extend: bool,
  return_struct: bool,
) -> pl.Expr:
  actual_fmt = fmt if fmt is not None else "g"
  if isinstance(size, _dt.timedelta):
    raise TypeError(
      f"Column '{name}' has numeric dtype {dtype}; 'size' must be a number, "
      f"not a timedelta. Use a timedelta only for temporal columns."
    )
  finite = col_ref.filter(col_ref.is_finite())
  raw_lo_f, raw_hi_f = _min_max(lf, finite)
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
  return _labeled_num_cut(
    col_ref, name, breaks_list, label_lo, label_hi, dtype, labels, left_closed, actual_fmt,
    return_struct
  )


# ── n_elements ────────────────────────────────────────────────────────────────


def _n_elements_temporal(
  *,
  lf: pl.LazyFrame,
  name: str,
  col_ref: pl.Expr,
  dtype: pl.DataType,
  n: int,
  tail: Literal["split", "merge"],
  labels: Sequence[str] | None,
  left_closed: bool,
  fmt: str | Callable | None,
  extend: bool,
  return_struct: bool,
) -> pl.Expr:
  del fmt, extend
  xs_phys = lf.select(col_ref.drop_nulls().sort().to_physical()).collect()[name].to_list()
  if not xs_phys:
    labs = list(labels) if labels is not None else ["[-∞, ∞)"]
    return _cut_physical_representation_expr(
      col_ref, name, [], labs, left_closed, return_struct, 0, 0, dtype
    )
  breaks_phys = _brk_n(xs_phys, n, tail, left_closed)
  lo_phys, hi_phys = xs_phys[0], xs_phys[-1]
  return _labeled_physical_representation_cut(
    col_ref, name, breaks_phys, lo_phys, hi_phys, dtype, labels, left_closed, None, return_struct
  )


def _n_elements_categorical(
  *,
  lf: pl.LazyFrame,
  name: str,
  col_ref: pl.Expr,
  dtype: pl.DataType,
  n: int,
  tail: Literal["split", "merge"],
  labels: Sequence[str] | None,
  left_closed: bool,
  fmt: str | Callable | None,
  extend: bool,
  return_struct: bool,
) -> pl.Expr:
  del fmt
  categories = _get_categories(lf, col_ref, name, dtype)
  xs_phys = (
    lf.select(col_ref.cast(pl.Enum(categories)).to_physical().drop_nulls().sort())
    .collect()[name]
    .to_list()
  )
  if not xs_phys:
    return _enum_null_result(name, labels, return_struct)
  breaks_phys = _brk_n(xs_phys, n, tail, left_closed)
  lo_phys = 0 if extend else xs_phys[0]
  hi_phys = len(categories) - 1 if extend else xs_phys[-1]
  return _labeled_enum_cut(
    col_ref, name, breaks_phys, lo_phys, hi_phys, categories, labels, left_closed, None,
    return_struct
  )


def _n_elements_numeric(
  *,
  lf: pl.LazyFrame,
  name: str,
  col_ref: pl.Expr,
  dtype: pl.DataType,
  n: int,
  tail: Literal["split", "merge"],
  labels: Sequence[str] | None,
  left_closed: bool,
  fmt: str | Callable | None,
  extend: bool,
  return_struct: bool,
) -> pl.Expr:
  xs = lf.select(col_ref.drop_nulls().sort()).collect()[name].to_list()
  xs_f = [float(v) for v in xs]
  breaks_list = _brk_n(xs_f, n, tail, left_closed)
  if not xs_f:
    label_lo, label_hi = float("-inf"), float("inf")
  elif extend:
    label_lo, label_hi = _numerical_extremes(dtype)
  else:
    label_lo, label_hi = xs_f[0], xs_f[-1]
  return _labeled_num_cut(
    col_ref, name, breaks_list, label_lo, label_hi, dtype, labels, left_closed, fmt, return_struct
  )


# ── quantiles ─────────────────────────────────────────────────────────────────


def _quantiles_temporal(
  *,
  lf: pl.LazyFrame,
  name: str,
  col_ref: pl.Expr,
  dtype: pl.DataType,
  probs: list[float],
  labels: Sequence[str] | None,
  left_closed: bool,
  fmt: str | Callable | None,
  raw: bool,
  extend: bool,
  return_struct: bool,
) -> pl.Expr:
  del raw, extend
  breaks_phys, mn_p, mx_p = _quantile_breaks_physical_representation(
    lf, probs, col_ref.to_physical()
  )
  lo_phys = int(mn_p) if mn_p is not None else 0
  hi_phys = int(mx_p) if mx_p is not None else 0
  return _labeled_physical_representation_cut(
    col_ref, name, breaks_phys, lo_phys, hi_phys, dtype, labels, left_closed, fmt, return_struct
  )


def _quantiles_categorical(
  *,
  lf: pl.LazyFrame,
  name: str,
  col_ref: pl.Expr,
  dtype: pl.DataType,
  probs: list[float],
  labels: Sequence[str] | None,
  left_closed: bool,
  fmt: str | Callable | None,
  raw: bool,
  extend: bool,
  return_struct: bool,
) -> pl.Expr:
  del raw
  categories = _get_categories(lf, col_ref, name, dtype)
  if not categories:
    return _enum_null_result(name, labels, return_struct)
  phys_expr = col_ref.cast(pl.Enum(categories)).to_physical()
  breaks_phys, mn_p, mx_p = _quantile_breaks_physical_representation(lf, probs, phys_expr)
  if mn_p is None or mx_p is None:
    return _enum_null_result(name, labels, return_struct)
  lo_phys = 0 if extend else int(mn_p)
  hi_phys = len(categories) - 1 if extend else int(mx_p)
  return _labeled_enum_cut(
    col_ref, name, breaks_phys, lo_phys, hi_phys, categories,
    labels, left_closed, fmt, return_struct
  )


def _quantiles_numeric(
  *,
  lf: pl.LazyFrame,
  name: str,
  col_ref: pl.Expr,
  dtype: pl.DataType,
  probs: list[float],
  labels: Sequence[str] | None,
  left_closed: bool,
  fmt: str | Callable | None,
  raw: bool,
  extend: bool,
  return_struct: bool,
) -> pl.Expr:
  actual_fmt = fmt if fmt is not None else ("g" if raw else ".0%")
  breaks_list, kept_probs, mn, mx = _quantile_breaks_float(lf, probs, col_ref)
  if mn is None or mx is None:
    bound_lo, bound_hi = float("-inf"), float("inf")
  elif extend:
    bound_lo, bound_hi = _numerical_extremes(dtype)
  else:
    bound_lo, bound_hi = float(mn), float(mx)
  if raw:
    return _labeled_num_cut(
      col_ref, name, breaks_list, bound_lo, bound_hi, dtype, labels, left_closed, actual_fmt,
      return_struct
    )
  auto_labels = (
    list(labels)
    if labels is not None
    else _make_quantile_labels(kept_probs, left_closed, actual_fmt)
  )
  return _cut_expr(
    col_ref,
    name,
    breaks_list,
    auto_labels,
    left_closed,
    return_struct,
    lo=bound_lo,
    hi=bound_hi,
    discrete=_is_integer(dtype),
  )


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

    Returns an Enum-typed column whose category names are the bin labels.
    Integer columns use fully-closed `[a, b]` notation; single-element bins
    are written as `{x}`.

    Args:
      breaks: Interior breakpoints; sorted automatically. Accepts numeric, string, or
              temporal Python values (datetime, date, timedelta, time).
      labels: Category labels (must be len(breaks) + 1). Auto-generated if omitted.
      left_closed: If True (default), intervals are [lo, hi); otherwise (lo, hi].
      fmt: Formatter for auto-generated labels. For numeric, a format-spec string
           (e.g. ".2f") or callable. For temporal, a callable or None (uses str()).
      extend: For numeric only — if True (default), outermost labels extend to
              -∞/+∞. For unsigned integers: 0/+∞. If False, uses data min/max.
              Temporal breaks always use data bounds regardless of this setting.
      return_struct: If True, return a struct {lo, hi} instead of just the label.

    Examples:
      scores = polarstation.make_example_data("scores")
      scores.ps.with_columns(
          pl.col("score").ps_chop.chop([40, 70], fmt=".0f").alias("grade")
      )
    """
    breaks = sorted(b for b in breaks)

    all_breaks_numeric = all(isinstance(b, (int, float)) for b in breaks)
    all_breaks_strings = all(isinstance(b, str) for b in breaks)
    all_breaks_temporal = all(isinstance(b, _TEMPORAL_PYTHON_TYPES) for b in breaks)

    def _observed_break_type() -> str:
      if not breaks:
        return "unknown"
      t = type(breaks[0])
      if issubclass(t, (int, float)):
        return "numeric"
      if issubclass(t, str):
        return "string"
      if issubclass(t, _TEMPORAL_PYTHON_TYPES):
        return "temporal"
      return t.__name__

    def make_type_error(name, dtype, required_break_type):
      return TypeError(
        f"Column '{name}' has dtype {dtype}; expected {required_break_type} breaks "
        f"but got {_observed_break_type()} breaks."
      )

    def impl(*, lf, name, col_ref, dtype, **kw):
      _assert_choppable_dtype(name, dtype)
      if _is_temporal(dtype) and not all_breaks_temporal:
        raise make_type_error(name, dtype, "temporal")
      elif _is_temporal(dtype):
        return _chop_temporal(lf=lf, name=name, col_ref=col_ref, dtype=dtype, **kw)
      elif _is_categorical(dtype) and not all_breaks_strings:
        raise make_type_error(name, dtype, "string")
      elif _is_categorical(dtype):
        return _chop_categorical(lf=lf, name=name, col_ref=col_ref, dtype=dtype, **kw)
      elif not all_breaks_numeric:
        raise make_type_error(name, dtype, "numeric")
      else:
        return _chop_numeric(lf=lf, name=name, col_ref=col_ref, dtype=dtype, **kw)

    return FrameExpr(
      self._expr,
      resolve_across_columns(
        self._expr,
        impl,
        breaks=breaks,
        labels=labels,
        left_closed=left_closed,
        fmt=fmt,
        extend=extend,
        return_struct=return_struct,
      ),
    )

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

    Returns an Enum-typed column whose category names are the bin labels.

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

    Examples:
      scores = polarstation.make_example_data("scores")
      scores.ps.with_columns(pl.col("score").ps_chop.width(25).alias("band"))
    """

    def impl(*, lf, name, col_ref, dtype, **kw):
      _assert_choppable_dtype(name, dtype)
      if _is_temporal(dtype):
        return _width_temporal(lf=lf, name=name, col_ref=col_ref, dtype=dtype, **kw)
      if _is_categorical(dtype):
        return _width_categorical(lf=lf, name=name, col_ref=col_ref, dtype=dtype, **kw)
      return _width_numeric(lf=lf, name=name, col_ref=col_ref, dtype=dtype, **kw)

    return FrameExpr(
      self._expr,
      resolve_across_columns(
        self._expr,
        impl,
        size=size,
        start=start,
        labels=labels,
        left_closed=left_closed,
        fmt=fmt,
        extend=extend,
        return_struct=return_struct,
      ),
    )

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

    Returns an Enum-typed column whose category names are the bin labels.
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

    Examples:
      scores = polarstation.make_example_data("scores")
      scores.ps.with_columns(pl.col("score").ps_chop.n_elements(3).alias("tercile"))
    """

    def impl(*, lf, name, col_ref, dtype, **kw):
      _assert_choppable_dtype(name, dtype)
      if _is_temporal(dtype):
        return _n_elements_temporal(lf=lf, name=name, col_ref=col_ref, dtype=dtype, **kw)
      if _is_categorical(dtype):
        return _n_elements_categorical(lf=lf, name=name, col_ref=col_ref, dtype=dtype, **kw)
      return _n_elements_numeric(lf=lf, name=name, col_ref=col_ref, dtype=dtype, **kw)

    return FrameExpr(
      self._expr,
      resolve_across_columns(
        self._expr,
        impl,
        n=n,
        tail=tail,
        labels=labels,
        left_closed=left_closed,
        fmt=fmt,
        extend=extend,
        return_struct=return_struct,
      ),
    )

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

    Returns an Enum-typed column whose category names are the bin labels.

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

    Examples:
      scores = polarstation.make_example_data("scores")
      scores.ps.with_columns(pl.col("score").ps_chop.n_groups(3).alias("tertile"))
    """
    return self.quantiles(
      [i / k for i in range(1, k)],
      labels=labels,
      left_closed=left_closed,
      fmt=fmt,
      raw=raw,
      extend=extend,
      return_struct=return_struct,
    )

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

    Returns an Enum-typed column whose category names are the bin labels.

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

    Examples:
      scores = polarstation.make_example_data("scores")
      scores.ps.with_columns(
          pl.col("score").ps_chop.quantiles([0.25, 0.75]).alias("iqr_group")
      )
    """

    def impl(*, lf, name, col_ref, dtype, **kw):
      _assert_choppable_dtype(name, dtype)
      if _is_temporal(dtype):
        return _quantiles_temporal(lf=lf, name=name, col_ref=col_ref, dtype=dtype, **kw)
      if _is_categorical(dtype):
        return _quantiles_categorical(lf=lf, name=name, col_ref=col_ref, dtype=dtype, **kw)
      return _quantiles_numeric(lf=lf, name=name, col_ref=col_ref, dtype=dtype, **kw)

    probs = sorted(float(p) for p in probs)
    return FrameExpr(
      self._expr,
      resolve_across_columns(
        self._expr,
        impl,
        probs=probs,
        labels=labels,
        left_closed=left_closed,
        fmt=fmt,
        raw=raw,
        extend=extend,
        return_struct=return_struct,
      ),
    )
