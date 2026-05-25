from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal

import polars as pl

from polarstation.frame_expr import FrameExpr
from polarstation.typing import IntoExpr, _into_expr


def _make_resolver(expr: pl.Expr, categories: Sequence[str] | None, make_null: list[str]):
  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    col_schema = lf.select(expr).collect_schema()
    result = []
    for name in col_schema.names():
      cats = list(categories) if categories is not None else None
      if cats is None:
        str_expr = pl.col(name).cast(pl.String)
        if make_null:
          str_expr = pl.when(str_expr.is_in(make_null)).then(None).otherwise(str_expr)
        cats = (
          lf.select(str_expr.drop_nulls().unique().sort().alias(name)).collect()[name].to_list()
        )
      str_col = pl.col(name).cast(pl.String)
      if make_null:
        str_col = pl.when(str_col.is_in(make_null)).then(None).otherwise(str_col)
      result.append(str_col.cast(pl.Enum(cats)).alias(name))
    return result

  return resolver


def _lump_resolver(expr: pl.Expr, n: int, other_label: str, lump_fn: Callable | None):
  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    col_schema = lf.select(expr).collect_schema()
    result = []
    for name in col_schema.names():
      counts = lf.group_by(name).len(name='n').sort("n", descending=True).collect()
      non_null_counts = counts.filter(pl.col(name).is_not_null())
      if lump_fn is not None:
        mask = lump_fn(non_null_counts)
        lump_bool = mask.to_list() if hasattr(mask, "to_list") else list(mask)
        all_cats_str = non_null_counts[name].cast(pl.String).to_list()
        lump_set = {c for c, v in zip(all_cats_str, lump_bool) if v}
        top_n_set = set(all_cats_str) - lump_set
      else:
        top_n_set = set(non_null_counts.head(n)[name].cast(pl.String).to_list())
      has_other = len(top_n_set) < non_null_counts.height
      # Preserve original category order; Other is always appended last.
      original_cats = col_schema[name].categories.to_list()
      kept_cats = [c for c in original_cats if c in top_n_set]
      new_cats = kept_cats + ([other_label] if has_other and other_label not in top_n_set else [])
      if has_other:
        col_expr = (
          pl.when(pl.col(name).is_null())
          .then(None)
          .when(pl.col(name).cast(pl.String).is_in(top_n_set))
          .then(pl.col(name).cast(pl.String))
          .otherwise(pl.lit(other_label))
          .cast(pl.Enum(new_cats))
          .alias(name)
        )
      else:
        col_expr = pl.col(name).cast(pl.Enum(new_cats)).alias(name)
      result.append(col_expr)
    return result

  return resolver


def _relabel_resolver(
  expr: pl.Expr, mapping: Mapping[str, str] | Callable[[str], str], strict: bool
):
  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    col_schema = lf.select(expr).collect_schema()
    result = []
    for name in col_schema.names():
      old_cats = col_schema[name].categories.to_list()
      if callable(mapping):
        new_cats = [mapping(c) for c in old_cats]
      else:
        if strict:
          unknown = set(mapping.keys()) - set(old_cats)
          if unknown:
            raise ValueError(f"relabel strict=True: keys not in categories: {sorted(unknown)!r}")
        new_cats = [mapping.get(c, c) for c in old_cats]
      result.append(
        pl.col(name).cast(pl.String).replace(old_cats, new_cats).cast(pl.Enum(new_cats)).alias(name)
      )
    return result

  return resolver


def _missing_to_category_resolver(expr: pl.Expr, category_name: str):
  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    col_schema = lf.select(expr).collect_schema()
    result = []
    for name in col_schema.names():
      old_cats = col_schema[name].categories.to_list()
      new_cats = old_cats if category_name in old_cats else old_cats + [category_name]
      result.append(
        pl.when(pl.col(name).is_null())
        .then(pl.lit(category_name))
        .otherwise(pl.col(name).cast(pl.String))
        .cast(pl.Enum(new_cats))
        .alias(name)
      )
    return result

  return resolver


def _category_to_missing_resolver(expr: pl.Expr, names: list[str]):
  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    col_schema = lf.select(expr).collect_schema()
    result = []
    for name in col_schema.names():
      old_cats = col_schema[name].categories.to_list()
      unknown = set(names) - set(old_cats)
      if unknown:
        raise ValueError(f"category_to_missing: {sorted(unknown)!r} are not categories")
      names_set = set(names)
      new_cats = [c for c in old_cats if c not in names_set]
      result.append(
        pl.when(pl.col(name).cast(pl.String).is_in(names))
        .then(None)
        .otherwise(pl.col(name).cast(pl.String))
        .cast(pl.Enum(new_cats))
        .alias(name)
      )
    return result

  return resolver


def _set_categories_resolver(expr: pl.Expr, categories: Sequence[str]):
  new_cats = list(categories)

  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    col_schema = lf.select(expr).collect_schema()
    return [
      pl.col(name).cast(pl.String).cast(pl.Enum(new_cats), strict=False).alias(name)
      for name in col_schema.names()
    ]

  return resolver


def _drop_unused_resolver(expr: pl.Expr):
  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    col_schema = lf.select(expr).collect_schema()
    result = []
    for name in col_schema.names():
      old_cats = col_schema[name].categories.to_list()
      used = set(lf.select(pl.col(name).drop_nulls().unique()).collect()[name].to_list())
      new_cats = [c for c in old_cats if c in used]  # preserves original order
      result.append(pl.col(name).cast(pl.Enum(new_cats)).alias(name))
    return result

  return resolver


def _add_categories_resolver(expr: pl.Expr, categories: Sequence[str], after: int | float):
  new_cats_to_add = list(categories)

  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    col_schema = lf.select(expr).collect_schema()
    result = []
    for name in col_schema.names():
      old_cats = col_schema[name].categories.to_list()
      pos = len(old_cats) if after >= len(old_cats) else max(0, int(after) + 1)
      new_cats = old_cats[:pos] + new_cats_to_add + old_cats[pos:]
      result.append(pl.col(name).cast(pl.Enum(new_cats)).alias(name))
    return result

  return resolver


def _rev_resolver(expr: pl.Expr):
  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    col_schema = lf.select(expr).collect_schema()
    return [
      pl.col(name).cast(pl.Enum(col_schema[name].categories.to_list()[::-1])).alias(name)
      for name in col_schema.names()
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

    col_schema = lf.select(expr).collect_schema()
    result = []
    for name in col_schema.names():
      order_df = (
        lf.group_by(name).agg(agg(b).alias(t) for b, t in zip(bys, _tmp, strict=False)).collect()
      )
      has_null_agg = pl.any_horizontal(pl.col(t).is_null() for t in _tmp)
      complete = order_df.filter(~has_null_agg).sort(_tmp, descending=desc, nulls_last=nl)
      incomplete = order_df.filter(has_null_agg)

      if missing == "drop":
        ordered = complete
      elif missing == "last":
        ordered = pl.concat([complete, incomplete])
      else:  # "first"
        ordered = pl.concat([incomplete, complete])

      order = ordered[name].drop_nulls().cast(pl.String).to_list()
      result.append(pl.col(name).cast(pl.Enum(order)).alias(name))
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
      categories: Fixed set of allowed values. If omitted, derived from the data as the
        unique values in alphabetical order.
      make_null: Values to replace with null before casting.

    Examples:
      ```{python}
      animals = polarstation.make_example_data("animals")
      animals.ps.with_columns(pl.col("animal").ps_enum.make())
      ```

      ```{python}
      pl.DataFrame({"x": ["a", "b", "?"]}).ps.with_columns(
          pl.col("x").ps_enum.make(categories=["a", "b", "z"], make_null="?")
      )['x'].dtype
      ```
    """
    if isinstance(make_null, str):
      make_null = [make_null]
    return FrameExpr(self._expr, _make_resolver(self._expr, categories, list(make_null)))

  def missing_to_category(self, name: str) -> FrameExpr:
    """Convert null values into a new category `name`, appended at the end.

    If `name` is already a category, null values are mapped to the existing
    category without modifying the category list.

    Args:
      name: Label for the category to assign to null values.

    Examples:
      animals = polarstation.make_example_data("animals")
      animals.ps.with_columns(
          new_animals = pl.col("animal").ps_enum.make().ps_enum.missing_to_category("unknown")
      )
    """
    return FrameExpr(self._expr, _missing_to_category_resolver(self._expr, name))

  def category_to_missing(self, name: str | Sequence[str]) -> FrameExpr:
    """Convert all occurrences of one or more categories to null and remove them from the Enum.

    Args:
      name: Category name(s) to nullify. Raises if any are not current categories.

    Examples:
      animals = polarstation.make_example_data("animals")
      animals.ps.with_columns(
          new_animals =  pl.col("animal").ps_enum.make().ps_enum.category_to_missing("bird")
      )
    """
    names = [name] if isinstance(name, str) else list(name)
    return FrameExpr(self._expr, _category_to_missing_resolver(self._expr, names))

  def relabel(
    self,
    mapping: Mapping[str, str] | Callable[[str], str],
    strict: bool = True,
  ) -> FrameExpr:
    """Rename categories, leaving any not present in the mapping unchanged.

    Args:
      mapping: A dict of old → new names, or a callable applied to each category name.
      strict: If True (default), raise if any dict key is not an existing category.

    Examples:
      ```{python}
      animals = polarstation.make_example_data("animals")
      animals.ps.with_columns(
          pl.col("animal").ps_enum.make().ps_enum.relabel({"bird": "Bird", "cow": "Cow"})
      )
      ```

      ```{python}
      animals.ps.with_columns(
          pl.col("animal").ps_enum.make().ps_enum.relabel(str.upper)
      )
      ```
    """
    return FrameExpr(self._expr, _relabel_resolver(self._expr, mapping, strict))

  def set_categories(self, categories: Sequence[str]) -> FrameExpr:
    """Set the exact category list. Values not in `categories` become null.

    Args:
      categories: The new ordered category list.

    Examples:
      animals = polarstation.make_example_data("animals")
      animals.ps.with_columns(
          pl.col("animal").ps_enum.make().ps_enum.set_categories(["cow", "dog"])
      )
    """
    return FrameExpr(self._expr, _set_categories_resolver(self._expr, categories))

  def drop_unused(self) -> FrameExpr:
    """Remove categories that don't appear in the data, preserving order.

    Examples:
      df = pl.DataFrame(
          {'x': pl.Series('x', ['bird', 'bird'], dtype=pl.Enum(['fish', 'bird', 'cat']))}
      )
      df.ps.with_columns(pl.col("x").ps_enum.drop_unused())["x"].dtype
    """
    return FrameExpr(self._expr, _drop_unused_resolver(self._expr))

  def add_categories(
    self, categories: Sequence[str], after: int | float = float("inf")
  ) -> FrameExpr:
    """Insert new categories without changing any values.

    Args:
      categories: New category labels to add.
      after: Insert after this 0-based index. Defaults to appending at the end.

    Examples:
      animals = polarstation.make_example_data("animals")
      animals.ps.with_columns(
          pl.col("animal").ps_enum.make().ps_enum.add_categories(["rabbit"], after=1)
      )["animal"].dtype
    """
    return FrameExpr(self._expr, _add_categories_resolver(self._expr, categories, after))

  def lump(
    self,
    n: int = 5,
    other_label: str = "Other",
    lump_fn: Callable[[pl.DataFrame], "Iterable[bool]"] | None = None,
  ) -> FrameExpr:
    """Collapse infrequent categories into `other_label`.

    By default keeps the top-`n` most frequent categories and collapses the rest.
    Pass `lump_fn` to use a custom rule instead (in which case `n` is ignored).

    The order of the categories remains unchanged with `other_label` appended at the end.

    Args:
      n: Number of categories to keep (ignored when `lump_fn` is provided).
      other_label: Label for the collapsed category.
      lump_fn: Optional callable that receives the non-null counts DataFrame
        (columns: category column + ``"n"``, sorted by frequency descending)
        and returns a boolean sequence where ``True`` marks categories to collapse.

    Examples:
      animals = polarstation.make_example_data("animals")
      animals.ps.with_columns(
          pl.col("animal").ps_enum.make().ps_enum.lump(n=1)
      )
    """
    return FrameExpr(self._expr, _lump_resolver(self._expr, n, other_label, lump_fn))

  def rev(self) -> FrameExpr:
    """Reverse the order of categories.

    Examples:
      animals = polarstation.make_example_data("animals")
      animals.ps.with_columns(
          pl.col("animal").ps_enum.make().ps_enum.rev()
      )["animal"].dtype
    """
    return FrameExpr(self._expr, _rev_resolver(self._expr))

  def infreq(self, descending: bool = False) -> FrameExpr:
    """Reorder categories by frequency, most frequent first.

    Args:
      descending: If True, least frequent first instead.

    Examples:
      animals = polarstation.make_example_data("animals")
      animals.ps.with_columns(
          pl.col("animal").ps_enum.make().ps_enum.infreq()
      )["animal"].dtype
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

    Examples:
      animals = polarstation.make_example_data("animals")
      animals.ps.with_columns(
          pl.col("animal").ps_enum.make().ps_enum.reorder("weight", agg=pl.Expr.mean)
      )["animal"].dtype
    """
    bys = [_into_expr(by)] if isinstance(by, (pl.Expr, str)) else [_into_expr(b) for b in by]
    return FrameExpr(
      self._expr, _reorder_resolver(self._expr, bys, agg, descending, nulls_last, missing)
    )
