from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal

import polars as pl

from polarstation.frame_expr import FrameExpr, resolve_across_columns
from polarstation.typing import IntoExpr, _into_expr


def _require_enum(name: str, dtype) -> None:
  if not isinstance(dtype, pl.Enum):
    raise TypeError(
      f"column {name!r} has dtype {dtype}, expected Enum. Call .ps_enum.make() first."
    )


def _get_cats(lf: pl.LazyFrame, col_ref: pl.Expr, name: str, dtype) -> list[str]:
  """Return the category list, deriving it from col_ref for String/Categorical."""
  if isinstance(dtype, pl.Enum):
    return dtype.categories.to_list()
  return lf.select(col_ref.cast(pl.String).drop_nulls().unique().sort()).collect()[name].to_list()


def _make_impl(*, lf, name, col_ref, dtype, categories, make_null):
  cats = list(categories) if categories is not None else None
  if cats is None:
    if make_null:
      filtered = pl.when(col_ref.cast(pl.String).is_in(make_null)).then(None).otherwise(col_ref)
    else:
      filtered = col_ref
    cats = (
      lf.select(filtered.drop_nulls().unique().sort().cast(pl.String).alias(name))
      .collect()[name]
      .to_list()
    )
  str_col = col_ref.cast(pl.String)
  if make_null:
    str_col = pl.when(str_col.is_in(make_null)).then(None).otherwise(str_col)
  return str_col.cast(pl.Enum(cats)).alias(name)


def _lump_impl(*, lf, name, col_ref, dtype, n, other_label, lump_fn):
  counts = lf.group_by(col_ref).len(name="n").sort("n", descending=True).collect()
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
  original_cats = _get_cats(lf, col_ref, name, dtype)
  kept_cats = [c for c in original_cats if c in top_n_set]
  new_cats = kept_cats + ([other_label] if has_other and other_label not in top_n_set else [])
  if has_other:
    return (
      pl.when(col_ref.is_null())
      .then(None)
      .when(col_ref.cast(pl.String).is_in(top_n_set))
      .then(col_ref.cast(pl.String))
      .otherwise(pl.lit(other_label))
      .cast(pl.Enum(new_cats))
      .alias(name)
    )
  return col_ref.cast(pl.Enum(new_cats)).alias(name)


def _rename_impl(*, lf, name, col_ref, dtype, mapping, strict):
  old_cats = _get_cats(lf, col_ref, name, dtype)
  if callable(mapping):
    new_cats_full = [mapping(c) for c in old_cats]
  else:
    if strict:
      unknown = set(mapping.keys()) - set(old_cats)
      if unknown:
        raise ValueError(f"rename strict=True: keys not in categories: {sorted(unknown)!r}")
    new_cats_full = [mapping.get(c, c) for c in old_cats]
  seen: set[str] = set()
  new_cats = [c for c in new_cats_full if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]
  return (
    col_ref.cast(pl.String).replace(old_cats, new_cats_full).cast(pl.Enum(new_cats)).alias(name)
  )


def _missing_to_category_impl(*, lf, name, col_ref, dtype, category_name):
  _require_enum(name, dtype)
  old_cats = dtype.categories.to_list()
  new_cats = old_cats if category_name in old_cats else old_cats + [category_name]
  return (
    pl.when(col_ref.is_null())
    .then(pl.lit(category_name))
    .otherwise(col_ref.cast(pl.String))
    .cast(pl.Enum(new_cats))
    .alias(name)
  )


def _category_to_missing_impl(*, lf, name, col_ref, dtype, names):
  _require_enum(name, dtype)
  old_cats = dtype.categories.to_list()
  unknown = set(names) - set(old_cats)
  if unknown:
    raise ValueError(f"category_to_missing: {sorted(unknown)!r} are not categories")
  names_set = set(names)
  new_cats = [c for c in old_cats if c not in names_set]
  return (
    pl.when(col_ref.cast(pl.String).is_in(names))
    .then(None)
    .otherwise(col_ref.cast(pl.String))
    .cast(pl.Enum(new_cats))
    .alias(name)
  )


def _set_categories_impl(*, lf, name, col_ref, dtype, new_cats):
  _require_enum(name, dtype)
  return col_ref.cast(pl.String).cast(pl.Enum(new_cats), strict=False).alias(name)


def _drop_unused_impl(*, lf, name, col_ref, dtype):
  _require_enum(name, dtype)
  old_cats = dtype.categories.to_list()
  used = set(lf.select(col_ref.drop_nulls().unique()).collect()[name].to_list())
  new_cats = [c for c in old_cats if c in used]
  return col_ref.cast(pl.Enum(new_cats)).alias(name)


def _add_categories_impl(*, lf, name, col_ref, dtype, new_cats_to_add, before):
  _require_enum(name, dtype)
  old_cats = dtype.categories.to_list()
  if before is None:
    pos = len(old_cats)
  elif before >= 0:
    pos = min(before, len(old_cats))
  else:
    pos = max(0, len(old_cats) + before)
  new_cats = old_cats[:pos] + new_cats_to_add + old_cats[pos:]
  return col_ref.cast(pl.Enum(new_cats)).alias(name)


def _move_impl(*, lf, name, col_ref, dtype, levels, before):
  _require_enum(name, dtype)
  old_cats = dtype.categories.to_list()
  unknown = set(levels) - set(old_cats)
  if unknown:
    raise ValueError(f"move: {sorted(unknown)!r} are not existing categories")
  levels_set = set(levels)
  rest = [c for c in old_cats if c not in levels_set]
  if before is None:
    pos = len(rest)
  elif before >= 0:
    pos = min(before, len(rest))
  else:
    pos = max(0, len(rest) + before)
  new_cats = rest[:pos] + list(levels) + rest[pos:]
  return col_ref.cast(pl.Enum(new_cats)).alias(name)


def _rev_impl(*, lf, name, col_ref, dtype):
  _require_enum(name, dtype)
  return col_ref.cast(pl.Enum(dtype.categories.to_list()[::-1])).alias(name)


def _reorder_impl(*, lf, name, col_ref, dtype, bys, _tmp, agg, desc, nl, missing):
  order_df = (
    lf.group_by(col_ref).agg(agg(b).alias(t) for b, t in zip(bys, _tmp, strict=False)).collect()
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
  return col_ref.cast(pl.Enum(order)).alias(name)


@pl.api.register_expr_namespace("ps_enum")
class PolarstationEnumExpression:
  def __init__(self, expr: pl.Expr) -> None:
    self._expr = expr

  def make(
    self,
    categories: Sequence[str] | None = None,
    make_null: Sequence[str] | str = (),
  ) -> FrameExpr:
    """Cast a column to Enum, optionally deriving categories from the data.

    When categories are derived from the data, they are sorted by the column's native dtype
    before being cast to string. This means integers sort numerically (1, 2, 10), dates
    chronologically, and strings alphabetically — rather than all sorting lexicographically.

    Args:
      categories: Fixed set of allowed values. If omitted, derived from the data as the
        unique values sorted by native dtype order.
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
    return FrameExpr(
      self._expr,
      resolve_across_columns(
        self._expr, _make_impl, categories=categories, make_null=list(make_null)
      ),
    )

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
    return FrameExpr(
      self._expr,
      resolve_across_columns(self._expr, _missing_to_category_impl, category_name=name),
    )

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
    return FrameExpr(
      self._expr,
      resolve_across_columns(self._expr, _category_to_missing_impl, names=names),
    )

  def rename(
    self,
    mapping: Mapping[str, str] | Callable[[str], str],
    strict: bool = True,
  ) -> FrameExpr:
    """Rename categories, leaving any not present in the mapping unchanged.

    The function also accepts a String or Categorical column, in addition to Enum, in which
    case `ps_enum.make()` is called first.

    Args:
      mapping: A dict of old → new names, or a callable applied to each category name.
      strict: If True (default), raise if any dict key is not an existing category.

    Examples:
      ```{python}
      animals = polarstation.make_example_data("animals")
      animals.ps.with_columns(
          pl.col("animal").ps_enum.make().ps_enum.rename({"bird": "Bird", "cow": "Cow"})
      )
      ```

      ```{python}
      animals.ps.with_columns(
          pl.col("animal").ps_enum.make().ps_enum.rename(str.upper)
      )
      ```
    """
    return FrameExpr(
      self._expr,
      resolve_across_columns(self._expr, _rename_impl, mapping=mapping, strict=strict),
    )

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
    return FrameExpr(
      self._expr,
      resolve_across_columns(self._expr, _set_categories_impl, new_cats=list(categories)),
    )

  def unify(self) -> FrameExpr:
    """Give all matched Enum columns the same category set — the union of all their levels.

    Categories are ordered by first appearance across columns (left to right). Values are
    never changed; only the dtype gains the extra categories.

    Requires all matched columns to already be Enum. Call ``.ps_enum.make()`` first if needed.

    Examples:
      df = pl.DataFrame({
          'x': pl.Series(['a', 'b'], dtype=pl.Enum(['a', 'b'])),
          'y': pl.Series(['b', 'c'], dtype=pl.Enum(['b', 'c'])),
      })
      df.ps.with_columns(pl.col('x', 'y').ps_enum.unify())['x'].dtype
    """
    expr = self._expr

    def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
      col_schema = lf.select(expr).collect_schema()
      seen: set[str] = set()
      union: list[str] = []
      for col_name in col_schema.names():
        dtype = col_schema[col_name]
        _require_enum(col_name, dtype)
        for c in dtype.categories.to_list():
          if c not in seen:
            seen.add(c)
            union.append(c)
      return [
        pl.struct(expr).struct.field(col_name).cast(pl.Enum(union)).alias(col_name)
        for col_name in col_schema.names()
      ]

    return FrameExpr(expr, resolver)

  def drop_unused(self) -> FrameExpr:
    """Remove categories that don't appear in the data, preserving order.

    Examples:
      df = pl.DataFrame(
          {'x': pl.Series('x', ['bird', 'bird'], dtype=pl.Enum(['fish', 'bird', 'cat']))}
      )
      df.ps.with_columns(pl.col("x").ps_enum.drop_unused())["x"].dtype
    """
    return FrameExpr(self._expr, resolve_across_columns(self._expr, _drop_unused_impl))

  def add_categories(
    self, categories: Sequence[str], before: int | None = None
  ) -> FrameExpr:
    """Insert new categories without changing any values.

    Args:
      categories: New category labels to add.
      before: Insert before this 0-based index of the existing categories.
        ``None`` (default) appends at the end. Negative indices count from the end.
        Any value ≥ len(categories) is equivalent to ``None`` (end).

    Examples:
      animals = polarstation.make_example_data("animals")
      animals.ps.with_columns(
          pl.col("animal").ps_enum.make().ps_enum.add_categories(["rabbit"], before=1)
      )["animal"].dtype
    """
    return FrameExpr(
      self._expr,
      resolve_across_columns(
        self._expr, _add_categories_impl, new_cats_to_add=list(categories), before=before
      ),
    )

  def move(self, *levels: str, before: int | None = 0) -> FrameExpr:
    """Move specified categories to a given position, keeping all others in their relative order.

    Args:
      *levels: Category names to move, in the order they should appear at the destination.
      before: Insert before this 0-based index of the remaining categories.
        ``0`` (default) moves to the front. ``None`` appends at the end.
        Negative indices count from the end of the remaining categories.
        Any value ≥ len(remaining) is equivalent to ``None`` (end).

    Examples:
      animals = polarstation.make_example_data("animals")
      animals.ps.with_columns(
          pl.col("animal").ps_enum.make().ps_enum.move("dog")
      )["animal"].dtype
    """
    return FrameExpr(
      self._expr,
      resolve_across_columns(self._expr, _move_impl, levels=levels, before=before),
    )

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

    The function also accepts a String or Categorical column, in addition to Enum, in which
    case `ps_enum.make()` is called first.

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
    return FrameExpr(
      self._expr,
      resolve_across_columns(
        self._expr, _lump_impl, n=n, other_label=other_label, lump_fn=lump_fn
      ),
    )

  def rev(self) -> FrameExpr:
    """Reverse the order of categories.

    Examples:
      animals = polarstation.make_example_data("animals")
      animals.ps.with_columns(
          pl.col("animal").ps_enum.make().ps_enum.rev()
      )["animal"].dtype
    """
    return FrameExpr(self._expr, resolve_across_columns(self._expr, _rev_impl))

  def infreq(self, descending: bool = False) -> FrameExpr:
    """Reorder categories by frequency, most frequent first.

    The function also accepts a String or Categorical column, in addition to Enum, in which
    case `ps_enum.make()` is called first.

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

    The function also accepts a String or Categorical column, in addition to Enum, in which
    case `ps_enum.make()` is called first.

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
    _tmp = [f"__by_{i}__" for i in range(len(bys))]
    desc = [descending] * len(bys) if isinstance(descending, bool) else list(descending)
    nl = [nulls_last] * len(bys) if isinstance(nulls_last, bool) else list(nulls_last)
    return FrameExpr(
      self._expr,
      resolve_across_columns(
        self._expr,
        _reorder_impl,
        bys=bys,
        _tmp=_tmp,
        agg=agg,
        desc=desc,
        nl=nl,
        missing=missing,
      ),
    )
