# polarstation


<!-- DO NOT EDIT README.md — it is generated from README.qmd.
     To update: quarto render README.qmd --to gfm -->

Tidy helper functions for [Polars](https://pola.rs), inspired by the R
[tidyverse](https://tidyverse.org/).

## Installation

``` bash
pip install polarstation
```

or with uv:

``` bash
uv add polarstation
```

## Quick start

``` python
import polars as pl
import polarstation   # registers extension functions for polars

df = pl.DataFrame({
    "animal": ["cat", "dog", None, "bird", "dog" , "bird", "bird"],
    "weight": [4.2, 8.1, 7.5, 0.5, 0.6, 0.4, None],
})

df.ps.with_columns(
    pl.col("animal").ps_enum.make().ps_enum.infreq()
)
```

    shape: (7, 2)
    ┌────────┬────────┐
    │ animal ┆ weight │
    │ ---    ┆ ---    │
    │ enum   ┆ f64    │
    ╞════════╪════════╡
    │ cat    ┆ 4.2    │
    │ dog    ┆ 8.1    │
    │ null   ┆ 7.5    │
    │ bird   ┆ 0.5    │
    │ dog    ┆ 0.6    │
    │ bird   ┆ 0.4    │
    │ bird   ┆ null   │
    └────────┴────────┘

`ps.with_columns` is a drop-in replacement for `with_columns` from
polars that can handle some additional use-case like functions that need
to peek at the full data for evaluation. It works efficiently
identically on both `DataFrame` and `LazyFrame`.

------------------------------------------------------------------------

## `ps_enum` — Enum column helpers

These function must be executed from within `ps.with_columns`.

``` python
animals = pl.DataFrame({
    "animal": ["cat", "dog", None, "bird", "bird", "bird", ],
    "weight": [4.2, 8.1, 7.5, 0.5, 0.12, None],
})
```

### `make(categories=None, make_null=())`

Cast a string column to `pl.Enum`, deriving the category list from the
data when `categories` is omitted. Pass `make_null` to treat specific
strings as `null`.

``` python
animals.ps.with_columns(pl.col("animal").ps_enum.make())
```

    shape: (6, 2)
    ┌────────┬────────┐
    │ animal ┆ weight │
    │ ---    ┆ ---    │
    │ enum   ┆ f64    │
    ╞════════╪════════╡
    │ cat    ┆ 4.2    │
    │ dog    ┆ 8.1    │
    │ null   ┆ 7.5    │
    │ bird   ┆ 0.5    │
    │ bird   ┆ 0.12   │
    │ bird   ┆ null   │
    └────────┴────────┘

``` python
# fixed category list + treat "?" as null
pl.DataFrame({"x": ["a", "b", "?"]}).ps.with_columns(
    pl.col("x").ps_enum.make(categories=["a", "b", "z"], make_null="?")
)['x'].dtype
```

    Enum(categories=['a', 'b', 'z'])

### `lump(n=5, other_label="Other")`

Keep the `n` most frequent categories; collapse the rest into
`other_label`.

``` python
animals.ps.with_columns(
    pl.col("animal").ps_enum.make().ps_enum.lump(n=2)
)
```

    shape: (6, 2)
    ┌────────┬────────┐
    │ animal ┆ weight │
    │ ---    ┆ ---    │
    │ enum   ┆ f64    │
    ╞════════╪════════╡
    │ cat    ┆ 4.2    │
    │ Other  ┆ 8.1    │
    │ null   ┆ 7.5    │
    │ bird   ┆ 0.5    │
    │ bird   ┆ 0.12   │
    │ bird   ┆ null   │
    └────────┴────────┘

### `relabel(mapping, strict=True)`

Rename categories. Pass a `dict` or a callable. With `strict=True`
(default), raises if any dict key is not an existing category.

``` python
animals.ps.with_columns(
    pl.col("animal").ps_enum.make().ps_enum.relabel({"bird": "Bird", "cat": "Cat"})
)
```

    shape: (6, 2)
    ┌────────┬────────┐
    │ animal ┆ weight │
    │ ---    ┆ ---    │
    │ enum   ┆ f64    │
    ╞════════╪════════╡
    │ Cat    ┆ 4.2    │
    │ dog    ┆ 8.1    │
    │ null   ┆ 7.5    │
    │ Bird   ┆ 0.5    │
    │ Bird   ┆ 0.12   │
    │ Bird   ┆ null   │
    └────────┴────────┘

``` python
# callable form
animals.ps.with_columns(
    pl.col("animal").ps_enum.make().ps_enum.relabel(str.upper)
)
```

    shape: (6, 2)
    ┌────────┬────────┐
    │ animal ┆ weight │
    │ ---    ┆ ---    │
    │ enum   ┆ f64    │
    ╞════════╪════════╡
    │ CAT    ┆ 4.2    │
    │ DOG    ┆ 8.1    │
    │ null   ┆ 7.5    │
    │ BIRD   ┆ 0.5    │
    │ BIRD   ┆ 0.12   │
    │ BIRD   ┆ null   │
    └────────┴────────┘

### `rev()`

Reverse the order of categories.

``` python
animals.ps.with_columns(
    pl.col("animal").ps_enum.make().ps_enum.rev()
)["animal"].dtype
```

    Enum(categories=['bird', 'dog', 'cat'])

### `infreq(descending=False)`

Sort categories by frequency, most frequent first. Pass
`descending=True` for least frequent first.

``` python
animals.ps.with_columns(
    pl.col("animal").ps_enum.make().ps_enum.infreq()
)["animal"].dtype
```

    Enum(categories=['bird', 'dog', 'cat'])

### `reorder(by, agg=pl.Expr.median, descending=False, nulls_last=False, missing="drop")`

Sort categories by an aggregation of another column within each group.
`by` accepts a column name (string) or expression.

``` python
animals.ps.with_columns(
    pl.col("animal").ps_enum.make().ps_enum.reorder("weight", agg=pl.Expr.mean)
)["animal"].dtype
```

    Enum(categories=['bird', 'cat', 'dog'])

`missing` controls what happens to categories whose aggregate is `null`:
`"drop"` (default) removes them from the Enum; `"last"` / `"first"` keep
them.

### `set_categories(categories)`

Replace the category list entirely. Values not present in `categories`
become `null`.

``` python
animals.ps.with_columns(
    pl.col("animal").ps_enum.make().ps_enum.set_categories(["cat", "dog"])
)
```

    shape: (6, 2)
    ┌────────┬────────┐
    │ animal ┆ weight │
    │ ---    ┆ ---    │
    │ enum   ┆ f64    │
    ╞════════╪════════╡
    │ cat    ┆ 4.2    │
    │ dog    ┆ 8.1    │
    │ null   ┆ 7.5    │
    │ null   ┆ 0.5    │
    │ null   ┆ 0.12   │
    │ null   ┆ null   │
    └────────┴────────┘

### `add_categories(categories, after=inf)`

Insert new categories without changing any values. `after` is a 0-based
index; defaults to appending at the end.

``` python
animals.ps.with_columns(
    pl.col("animal").ps_enum.make().ps_enum.add_categories(["rabbit"], after=1)
)["animal"].dtype
```

    Enum(categories=['cat', 'dog', 'rabbit', 'bird'])

### `drop_unused()`

Remove categories that don’t appear in the data, preserving order.

``` python
(pl.DataFrame({'x':  pl.Series('x', ['bird', 'bird'], dtype=pl.Enum(['fish', 'bird', 'cat']))})
    .ps.with_columns(pl.col("x").ps_enum.drop_unused())
)["x"].dtype
```

    Enum(categories=['bird'])

### `missing_to_category(name)` / `category_to_missing(name)`

Convert between `null` and a named category.

``` python
with_na = animals.ps.with_columns(
    pl.col("animal").ps_enum.make().ps_enum.missing_to_category("unknown")
)
print(with_na["animal"].to_list())

back = with_na.ps.with_columns(
    pl.col("animal").ps_enum.category_to_missing("unknown")
)
print(back["animal"].null_count(), "nulls restored")
```

    ['cat', 'dog', 'unknown', 'bird', 'bird', 'bird']
    1 nulls restored

------------------------------------------------------------------------

## `ps_str` — String column helpers

### `count(pattern="")`

Count non-overlapping regex matches per string.

``` python
pl.DataFrame({"x": ["hello world", "foo bar baz", ""]}).select(
    pl.col("x").ps_str.count(r"\b\w+\b").alias("word_count")
)
```

    shape: (3, 1)
    ┌────────────┐
    │ word_count │
    │ ---        │
    │ u32        │
    ╞════════════╡
    │ 2          │
    │ 3          │
    │ 0          │
    └────────────┘

### `wrap(width=80, ...)`

Word-wrap each string to at most `width` characters per line.

``` python
pl.DataFrame({"x": ["A long sentence that exceeds the column width."]}).select(
    pl.col("x").ps_str.wrap(width=25)
)
```

    shape: (1, 1)
    ┌──────────────────────┐
    │ x                    │
    │ ---                  │
    │ str                  │
    ╞══════════════════════╡
    │ A long sentence that │
    │ exceeds t…           │
    └──────────────────────┘

### `trunc(width=5, side="right", placeholder="…")`

Truncate each string to fit within `width` characters. `side` can be
`"right"` (default), `"left"`, or `"center"`.

``` python
pl.DataFrame({"x": ["short", "a much longer string"]}).select(
    pl.col("x").ps_str.trunc(width=10)
)
```

    shape: (2, 1)
    ┌────────────┐
    │ x          │
    │ ---        │
    │ str        │
    ╞════════════╡
    │ short      │
    │ a much lo… │
    └────────────┘

------------------------------------------------------------------------

## `ps.apply` — Escape hatch

When you need the full DataFrame context to build an expression, use
`expr.ps.apply`. The callable receives the `LazyFrame` and the resolved
column name.

``` python
def center_scale(lf: pl.LazyFrame, col: str) -> pl.Expr:
    stats = lf.select(pl.col(col).mean().alias("m"), pl.col(col).std().alias("s")).collect()
    m, s = stats["m"][0], stats["s"][0]
    return ((pl.col(col) - m) / s).alias(col)

pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).ps.with_columns(
    pl.col("x").ps.apply(center_scale)
)
```

    /Users/ahlmanne/prog/python/polarstation/src/polarstation/expr_extensions.py:22: PerformanceWarning: Determining the column names of a LazyFrame requires resolving its schema, which is a potentially expensive operation. Use `LazyFrame.collect_schema().names()` to get the column names without this warning.
      return [fn(df, col) for col in df.select(expr).columns]

    shape: (5, 1)
    ┌───────────┐
    │ x         │
    │ ---       │
    │ f64       │
    ╞═══════════╡
    │ -1.264911 │
    │ -0.632456 │
    │ 0.0       │
    │ 0.632456  │
    │ 1.264911  │
    └───────────┘

## Details

The key idea is `FrameExpr` — an expression that needs a peek at the
data (schema or a small aggregation) before it resolves into a regular
Polars expression. This unlocks operations like deriving Enum categories
from the data, lumping rare levels, or reordering factor levels by a
summary statistic, while keeping the rest of your pipeline lazy.

------------------------------------------------------------------------

## License

MIT
