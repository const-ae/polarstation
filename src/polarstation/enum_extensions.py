from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal

import polars as pl

from polarstation.frame_expr import FrameExpr


def _make_resolver(expr: pl.Expr, categories: Sequence[str] | None, make_null: list[str]):
  def resolver(df: pl.DataFrame) -> list[pl.Expr]:
    cols = df.select(expr).columns
    result = []
    for col in cols:
      cats = list(categories) if categories is not None else None
      if cats is None:
        s = df[col].cast(pl.String)
        if make_null:
          s = s.set(s.is_in(make_null), None)
        cats = s.drop_nulls().unique(maintain_order=True).to_list()
      str_col = pl.col(col).cast(pl.String)
      if make_null:
        str_col = pl.when(str_col.is_in(make_null)).then(None).otherwise(str_col)
      result.append(str_col.cast(pl.Enum(cats)).alias(col))
    return result

  return resolver


def _lump_resolver(expr: pl.Expr, n: int, other_label: str):
  def resolver(df: pl.DataFrame) -> list[pl.Expr]:
    cols = df.select(expr).columns
    result = []
    for col in cols:
      counts = df.group_by(col).len().sort("len", descending=True)
      non_null_counts = counts.filter(pl.col(col).is_not_null())
      top_n = non_null_counts.head(n)[col].cast(pl.String).to_list()
      has_other = len(top_n) < non_null_counts.height
      new_cats = top_n + ([other_label] if has_other and other_label not in top_n else [])
      if has_other:
        col_expr = (
          pl.when(pl.col(col).is_null())
          .then(None)
          .when(pl.col(col).cast(pl.String).is_in(top_n))
          .then(pl.col(col).cast(pl.String))
          .otherwise(pl.lit(other_label))
          .cast(pl.Enum(new_cats))
          .alias(col)
        )
      else:
        col_expr = pl.col(col).cast(pl.Enum(new_cats)).alias(col)
      result.append(col_expr)
    return result

  return resolver


def _relabel_resolver(
  expr: pl.Expr, mapping: Mapping[str, str] | Callable[[str], str], strict: bool
):
  def resolver(df: pl.DataFrame) -> list[pl.Expr]:
    cols = df.select(expr).columns
    result = []
    for col in cols:
      old_cats = df[col].dtype.categories.to_list()
      if callable(mapping):
        new_cats = [mapping(c) for c in old_cats]
      else:
        if strict:
          unknown = set(mapping.keys()) - set(old_cats)
          if unknown:
            raise ValueError(f"relabel strict=True: keys not in categories: {sorted(unknown)!r}")
        new_cats = [mapping.get(c, c) for c in old_cats]
      result.append(
        pl.col(col).cast(pl.String).replace(old_cats, new_cats).cast(pl.Enum(new_cats)).alias(col)
      )
    return result

  return resolver


def _reorder_resolver(
  expr: pl.Expr,
  bys: list[pl.Expr],
  agg: Callable,
  descending: bool | Sequence[bool],
  nulls_last: bool | Sequence[bool],
  missing: Literal["drop", "last", "first"],
):
  by_names = [b.meta.output_name() for b in bys]

  def resolver(df: pl.DataFrame) -> list[pl.Expr]:
    desc = [descending] * len(bys) if isinstance(descending, bool) else list(descending)
    nl = [nulls_last] * len(bys) if isinstance(nulls_last, bool) else list(nulls_last)

    cols = df.select(expr).columns
    result = []
    for col in cols:
      order_df = df.group_by(col).agg(agg(b).alias(b.meta.output_name()) for b in bys)
      has_null_agg = pl.any_horizontal(pl.col(n).is_null() for n in by_names)
      complete = order_df.filter(~has_null_agg).sort(by_names, descending=desc, nulls_last=nl)
      incomplete = order_df.filter(has_null_agg)

      if missing == "drop":
        ordered = complete
      elif missing == "last":
        ordered = pl.concat([complete, incomplete])
      else:  # "first"
        ordered = pl.concat([incomplete, complete])

      order = ordered[col].drop_nulls().cast(pl.String).to_list()
      result.append(pl.col(col).cast(pl.Enum(order)).alias(col))
    return result

  return resolver


@pl.api.register_expr_namespace("ps_enum")
class PolarstationEnumExpression:
  def __init__(self, expr: pl.Expr) -> None:
    self._expr = expr

  def make(
    self,
    categories: Sequence[str] | None = None,
    make_null: Sequence[str] | str = (),
  ) -> FrameExpr:
    """Cast a string column to Enum, optionally deriving categories from the data.

    Args:
      categories: Fixed set of allowed values. If omitted, derived from the data.
      make_null: Values to replace with null before casting.
    """
    if isinstance(make_null, str):
      make_null = [make_null]
    return FrameExpr(self._expr, _make_resolver(self._expr, categories, list(make_null)))

  def relabel(
    self,
    mapping: Mapping[str, str] | Callable[[str], str],
    strict: bool = True,
  ) -> FrameExpr:
    """Rename categories, leaving any not present in the mapping unchanged.

    Args:
      mapping: A dict of old → new names, or a callable applied to each category name.
      strict: If True (default), raise if any dict key is not an existing category.
    """
    return FrameExpr(self._expr, _relabel_resolver(self._expr, mapping, strict))

  def lump(self, n: int = 5, other_label: str = "Other") -> FrameExpr:
    """Collapse all but the top-n most frequent categories into `other_label`.

    Args:
      n: Number of categories to keep.
      other_label: Label for the collapsed category.
    """
    return FrameExpr(self._expr, _lump_resolver(self._expr, n, other_label))

  def reorder(
    self,
    by: pl.Expr | Iterable[pl.Expr],
    agg: Callable[[pl.Expr], pl.Expr] = pl.Expr.median,
    descending: bool | Sequence[bool] = False,
    nulls_last: bool | Sequence[bool] = False,
    missing: Literal["drop", "last", "first"] = "drop",
  ) -> FrameExpr:
    """Reorder categories by an aggregation of one or more columns within each group.

    Args:
      by: Column(s) to aggregate per category for ordering.
      agg: Aggregation applied to each `by` column (default: median).
      descending: Sort descending. A single bool applies to all columns.
      nulls_last: Place null aggregates last. A single bool applies to all columns.
      missing: How to handle categories whose aggregate is null —
               'drop' excludes them, 'last' appends them, 'first' prepends them.
    """
    bys = [by] if isinstance(by, pl.Expr) else list(by)
    return FrameExpr(
      self._expr, _reorder_resolver(self._expr, bys, agg, descending, nulls_last, missing)
    )
