from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal

import polars as pl

from polarstation.frame_expr import FrameExpr
from polarstation.typing import IntoExpr, _into_expr


def _make_resolver(expr: pl.Expr, categories: Sequence[str] | None, make_null: list[str]):
  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    cols = lf.select(expr).collect_schema().names()
    result = []
    for col in cols:
      cats = list(categories) if categories is not None else None
      if cats is None:
        str_expr = pl.col(col).cast(pl.String)
        if make_null:
          str_expr = pl.when(str_expr.is_in(make_null)).then(None).otherwise(str_expr)
        cats = (
          lf.select(str_expr.drop_nulls().unique(maintain_order=True).alias(col))
          .collect()[col]
          .to_list()
        )
      str_col = pl.col(col).cast(pl.String)
      if make_null:
        str_col = pl.when(str_col.is_in(make_null)).then(None).otherwise(str_col)
      result.append(str_col.cast(pl.Enum(cats)).alias(col))
    return result

  return resolver


def _lump_resolver(expr: pl.Expr, n: int, other_label: str):
  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    cols = lf.select(expr).collect_schema().names()
    result = []
    for col in cols:
      counts = lf.group_by(col).len().sort("len", descending=True).collect()
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
  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    cols = lf.select(expr).collect_schema().names()
    result = []
    for col in cols:
      old_cats = lf.collect_schema()[col].categories.to_list()
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


def _missing_to_category_resolver(expr: pl.Expr, name: str):
  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    cols = lf.select(expr).collect_schema().names()
    result = []
    for col in cols:
      old_cats = lf.collect_schema()[col].categories.to_list()
      if name in old_cats:
        raise ValueError(f"missing_to_category: {name!r} is already a category")
      new_cats = old_cats + [name]
      result.append(
        pl.when(pl.col(col).is_null())
        .then(pl.lit(name))
        .otherwise(pl.col(col).cast(pl.String))
        .cast(pl.Enum(new_cats))
        .alias(col)
      )
    return result

  return resolver


def _category_to_missing_resolver(expr: pl.Expr, name: str):
  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    cols = lf.select(expr).collect_schema().names()
    result = []
    for col in cols:
      old_cats = lf.collect_schema()[col].categories.to_list()
      if name not in old_cats:
        raise ValueError(f"category_to_missing: {name!r} is not a category")
      new_cats = [c for c in old_cats if c != name]
      result.append(
        pl.when(pl.col(col).cast(pl.String) == name)
        .then(None)
        .otherwise(pl.col(col).cast(pl.String))
        .cast(pl.Enum(new_cats))
        .alias(col)
      )
    return result

  return resolver


def _set_categories_resolver(expr: pl.Expr, categories: Sequence[str]):
  new_cats = list(categories)

  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    cols = lf.select(expr).collect_schema().names()
    return [
      pl.col(col).cast(pl.String).cast(pl.Enum(new_cats), strict=False).alias(col)
      for col in cols
    ]

  return resolver


def _drop_unused_resolver(expr: pl.Expr):
  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    cols = lf.select(expr).collect_schema().names()
    result = []
    for col in cols:
      old_cats = lf.collect_schema()[col].categories.to_list()
      used = set(lf.select(pl.col(col).drop_nulls().unique()).collect()[col].to_list())
      new_cats = [c for c in old_cats if c in used]  # preserves original order
      result.append(pl.col(col).cast(pl.Enum(new_cats)).alias(col))
    return result

  return resolver


def _add_categories_resolver(expr: pl.Expr, categories: Sequence[str], after: int | float):
  new_cats_to_add = list(categories)

  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    cols = lf.select(expr).collect_schema().names()
    result = []
    for col in cols:
      old_cats = lf.collect_schema()[col].categories.to_list()
      pos = len(old_cats) if after >= len(old_cats) else max(0, int(after) + 1)
      new_cats = old_cats[:pos] + new_cats_to_add + old_cats[pos:]
      result.append(pl.col(col).cast(pl.Enum(new_cats)).alias(col))
    return result

  return resolver


def _rev_resolver(expr: pl.Expr):
  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    cols = lf.select(expr).collect_schema().names()
    return [
      pl.col(col).cast(pl.Enum(lf.collect_schema()[col].categories.to_list()[::-1])).alias(col)
      for col in cols
    ]

  return resolver


def _reorder_resolver(
  expr: pl.Expr,
  bys: list[pl.Expr],
  agg: Callable,
  descending: bool | Sequence[bool],
  nulls_last: bool | Sequence[bool],
  missing: Literal["drop", "last", "first"],
):
  # Use stable internal names to avoid collision when a by-expr targets the group-by column.
  _tmp = [f"__by_{i}__" for i in range(len(bys))]

  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    desc = [descending] * len(bys) if isinstance(descending, bool) else list(descending)
    nl = [nulls_last] * len(bys) if isinstance(nulls_last, bool) else list(nulls_last)

    cols = lf.select(expr).collect_schema().names()
    result = []
    for col in cols:
      order_df = lf.group_by(col).agg(agg(b).alias(t) for b, t in zip(bys, _tmp)).collect()
      has_null_agg = pl.any_horizontal(pl.col(t).is_null() for t in _tmp)
      complete = order_df.filter(~has_null_agg).sort(_tmp, descending=desc, nulls_last=nl)
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

  def missing_to_category(self, name: str) -> FrameExpr:
    """Convert null values into a new category `name`, appended at the end.

    Args:
      name: Label for the new category. Raises if it already exists.
    """
    return FrameExpr(self._expr, _missing_to_category_resolver(self._expr, name))

  def category_to_missing(self, name: str) -> FrameExpr:
    """Convert all occurrences of category `name` to null and remove it from the Enum.

    Args:
      name: Category to nullify. Raises if it is not a current category.
    """
    return FrameExpr(self._expr, _category_to_missing_resolver(self._expr, name))

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

  def set_categories(self, categories: Sequence[str]) -> FrameExpr:
    """Set the exact category list. Values not in `categories` become null.

    Args:
      categories: The new ordered category list.
    """
    return FrameExpr(self._expr, _set_categories_resolver(self._expr, categories))

  def drop_unused(self) -> FrameExpr:
    """Remove categories that don't appear in the data, preserving order."""
    return FrameExpr(self._expr, _drop_unused_resolver(self._expr))

  def add_categories(
    self, categories: Sequence[str], after: int | float = float("inf")
  ) -> FrameExpr:
    """Insert new categories without changing any values.

    Args:
      categories: New category labels to add.
      after: Insert after this 0-based index. Defaults to appending at the end.
    """
    return FrameExpr(self._expr, _add_categories_resolver(self._expr, categories, after))

  def lump(self, n: int = 5, other_label: str = "Other") -> FrameExpr:
    """Collapse all but the top-n most frequent categories into `other_label`.

    Args:
      n: Number of categories to keep.
      other_label: Label for the collapsed category.
    """
    return FrameExpr(self._expr, _lump_resolver(self._expr, n, other_label))

  def rev(self) -> FrameExpr:
    """Reverse the order of categories."""
    return FrameExpr(self._expr, _rev_resolver(self._expr))

  def infreq(self, descending: bool = False) -> FrameExpr:
    """Reorder categories by frequency, most frequent first.

    Args:
      descending: If True, least frequent first instead.
    """
    ordered = self.reorder(self._expr, agg=pl.Expr.len, descending=False)
    return ordered if descending else ordered.ps_enum.rev()

  def reorder(
    self,
    by: IntoExpr | Iterable[IntoExpr],
    agg: Callable[[pl.Expr], pl.Expr] = pl.Expr.median,
    descending: bool | Sequence[bool] = False,
    nulls_last: bool | Sequence[bool] = False,
    missing: Literal["drop", "last", "first"] = "drop",
  ) -> FrameExpr:
    """Reorder categories by an aggregation of one or more columns within each group.

    Args:
      by: Column(s) to aggregate per category for ordering. Strings are treated as column names.
      agg: Aggregation applied to each `by` column (default: median).
      descending: Sort descending. A single bool applies to all columns.
      nulls_last: Place null aggregates last. A single bool applies to all columns.
      missing: How to handle categories whose aggregate is null —
               'drop' excludes them, 'last' appends them, 'first' prepends them.
    """
    bys = [_into_expr(by)] if isinstance(by, (pl.Expr, str)) else [_into_expr(b) for b in by]
    return FrameExpr(
      self._expr, _reorder_resolver(self._expr, bys, agg, descending, nulls_last, missing)
    )
