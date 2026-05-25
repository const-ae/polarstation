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
    "animal": ["dog", "dog", None, "bird", "cow" , "bird", "bird"],
    "weight": [12.2, 8.1, 7.5, 0.5, 460, 0.4, None],
}).ps.with_columns(
    pl.col("animal").ps_enum.make().ps_enum.reorder(by='weight')
)

print(df)

print(df['animal'].dtype)
```

    shape: (7, 2)
    ┌────────┬────────┐
    │ animal ┆ weight │
    │ ---    ┆ ---    │
    │ enum   ┆ f64    │
    ╞════════╪════════╡
    │ dog    ┆ 12.2   │
    │ dog    ┆ 8.1    │
    │ null   ┆ 7.5    │
    │ bird   ┆ 0.5    │
    │ cow    ┆ 460.0  │
    │ bird   ┆ 0.4    │
    │ bird   ┆ null   │
    └────────┴────────┘
    Enum(categories=['bird', 'dog', 'cow'])

`ps.with_columns` is a drop-in replacement for `with_columns` from
polars that can handle some additional use-case like functions that need
to peek at the full data for evaluation. It works efficiently
identically on both `DataFrame` and `LazyFrame`.

------------------------------------------------------------------------

## `ps_enum` — Enum column helpers

These function must be executed from within `ps.with_columns`.

``` python
animals = pl.DataFrame({
    "animal": ["dog", None, "bird", "cow" , "bird"],
    "weight": [12.2, 7.5, 0.5, 460, None],
})
```

### `make(categories=None, make_null=())`

Cast a string column to `pl.Enum`, deriving the category list from the
data when `categories` is omitted. Pass `make_null` to treat specific
strings as `null`.

``` python
animals.ps.with_columns(pl.col("animal").ps_enum.make())
```

    shape: (5, 2)
    ┌────────┬────────┐
    │ animal ┆ weight │
    │ ---    ┆ ---    │
    │ enum   ┆ f64    │
    ╞════════╪════════╡
    │ dog    ┆ 12.2   │
    │ null   ┆ 7.5    │
    │ bird   ┆ 0.5    │
    │ cow    ┆ 460.0  │
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

    shape: (5, 2)
    ┌────────┬────────┐
    │ animal ┆ weight │
    │ ---    ┆ ---    │
    │ enum   ┆ f64    │
    ╞════════╪════════╡
    │ dog    ┆ 12.2   │
    │ null   ┆ 7.5    │
    │ bird   ┆ 0.5    │
    │ Other  ┆ 460.0  │
    │ bird   ┆ null   │
    └────────┴────────┘

### `relabel(mapping, strict=True)`

Rename categories. Pass a `dict` or a callable. With `strict=True`
(default), raises if any dict key is not an existing category.

``` python
animals.ps.with_columns(
    pl.col("animal").ps_enum.make().ps_enum.relabel({"bird": "Bird", "cow": "Cow"})
)
```

    shape: (5, 2)
    ┌────────┬────────┐
    │ animal ┆ weight │
    │ ---    ┆ ---    │
    │ enum   ┆ f64    │
    ╞════════╪════════╡
    │ dog    ┆ 12.2   │
    │ null   ┆ 7.5    │
    │ Bird   ┆ 0.5    │
    │ Cow    ┆ 460.0  │
    │ Bird   ┆ null   │
    └────────┴────────┘

``` python
# callable form
animals.ps.with_columns(
    pl.col("animal").ps_enum.make().ps_enum.relabel(str.upper)
)
```

    shape: (5, 2)
    ┌────────┬────────┐
    │ animal ┆ weight │
    │ ---    ┆ ---    │
    │ enum   ┆ f64    │
    ╞════════╪════════╡
    │ DOG    ┆ 12.2   │
    │ null   ┆ 7.5    │
    │ BIRD   ┆ 0.5    │
    │ COW    ┆ 460.0  │
    │ BIRD   ┆ null   │
    └────────┴────────┘

### `rev()`

Reverse the order of categories.

``` python
animals.ps.with_columns(
    pl.col("animal").ps_enum.make().ps_enum.rev()
)["animal"].dtype
```

    Enum(categories=['dog', 'cow', 'bird'])

### `infreq(descending=False)`

Sort categories by frequency, most frequent first. Pass
`descending=True` for least frequent first.

``` python
animals.ps.with_columns(
    pl.col("animal").ps_enum.make().ps_enum.infreq()
)["animal"].dtype
```

    Enum(categories=['bird', 'dog', 'cow'])

### `reorder(by, agg=pl.Expr.median, descending=False, nulls_last=False, missing="drop")`

Sort categories by an aggregation of another column within each group.
`by` accepts a column name (string) or expression.

``` python
animals.ps.with_columns(
    pl.col("animal").ps_enum.make().ps_enum.reorder("weight", agg=pl.Expr.mean)
)["animal"].dtype
```

    Enum(categories=['bird', 'dog', 'cow'])

`missing` controls what happens to categories whose aggregate is `null`:
`"drop"` (default) removes them from the Enum; `"last"` / `"first"` keep
them.

### `set_categories(categories)`

Replace the category list entirely. Values not present in `categories`
become `null`.

``` python
animals.ps.with_columns(
    pl.col("animal").ps_enum.make().ps_enum.set_categories(["cow", "dog"])
)
```

    shape: (5, 2)
    ┌────────┬────────┐
    │ animal ┆ weight │
    │ ---    ┆ ---    │
    │ enum   ┆ f64    │
    ╞════════╪════════╡
    │ dog    ┆ 12.2   │
    │ null   ┆ 7.5    │
    │ null   ┆ 0.5    │
    │ cow    ┆ 460.0  │
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

    Enum(categories=['bird', 'cow', 'rabbit', 'dog'])

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
print(back["animal"].to_list())
```

    ['dog', 'unknown', 'bird', 'cow', 'bird']
    ['dog', None, 'bird', 'cow', 'bird']

------------------------------------------------------------------------

## `ps_str` — String column helpers

### `count(pattern="")`

Count non-overlapping regex matches per string. This just a wrapper
around `pl.Expr.str.count_matches()` and will probably deleted soon.

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

## `ps_chop` — Bin a column into intervals

These functions must be executed from within `ps.with_columns`. They
return an `Enum`-typed column whose category names are the bin labels.

``` python
scores = pl.DataFrame({"score": [12, 45, 67, 89, 95, 23, 78]})
```

### `chop(breaks, left_closed=True, extend=True, fmt=None, labels=None)`

Cut at explicit breakpoints.

``` python
scores.ps.with_columns(
    pl.col("score").ps_chop.chop([40, 70], fmt=".0f").alias("grade")
)
```

    shape: (7, 2)
    ┌───────┬──────────┐
    │ score ┆ grade    │
    │ ---   ┆ ---      │
    │ i64   ┆ enum     │
    ╞═══════╪══════════╡
    │ 12    ┆ (-∞, 39] │
    │ 45    ┆ [40, 69] │
    │ 67    ┆ [40, 69] │
    │ 89    ┆ [70, +∞) │
    │ 95    ┆ [70, +∞) │
    │ 23    ┆ (-∞, 39] │
    │ 78    ┆ [70, +∞) │
    └───────┴──────────┘

Integer columns use fully-closed `[a, b]` notation; single-element bins
are written as `{x}`.

### `width(size, start=None, left_closed=True, extend=False, fmt=None, labels=None)`

Cut into equal-width bins.

``` python
scores.ps.with_columns(
    pl.col("score").ps_chop.width(25).alias("band")
)
```

    shape: (7, 2)
    ┌───────┬──────────┐
    │ score ┆ band     │
    │ ---   ┆ ---      │
    │ i64   ┆ enum     │
    ╞═══════╪══════════╡
    │ 12    ┆ [12, 36] │
    │ 45    ┆ [37, 61] │
    │ 67    ┆ [62, 86] │
    │ 89    ┆ [87, 95] │
    │ 95    ┆ [87, 95] │
    │ 23    ┆ [12, 36] │
    │ 78    ┆ [62, 86] │
    └───────┴──────────┘

For temporal columns `size` must be a `datetime.timedelta`.

### `n_elements(n, tail="split", left_closed=True, extend=False, fmt="g", labels=None)`

Cut into groups of `n` observations (sorted order). Ties are never split
— the boundary advances to the next distinct value. `tail="merge"`
absorbs a short final group into the preceding one.

``` python
scores.ps.with_columns(
    pl.col("score").ps_chop.n_elements(3).alias("tercile")
)
```

    shape: (7, 2)
    ┌───────┬──────────┐
    │ score ┆ tercile  │
    │ ---   ┆ ---      │
    │ i64   ┆ enum     │
    ╞═══════╪══════════╡
    │ 12    ┆ [12, 66] │
    │ 45    ┆ [12, 66] │
    │ 67    ┆ [67, 94] │
    │ 89    ┆ [67, 94] │
    │ 95    ┆ {95}     │
    │ 23    ┆ [12, 66] │
    │ 78    ┆ [67, 94] │
    └───────┴──────────┘

### `n_groups(k, raw=True, left_closed=True, extend=False, fmt=None, labels=None)`

Cut into `k` equal-count groups using quantile boundaries. With
`raw=False` the labels show percentages instead of actual values.

``` python
scores.ps.with_columns(
    pl.col("score").ps_chop.n_groups(3).alias("tertile")
)
```

    shape: (7, 2)
    ┌───────┬──────────┐
    │ score ┆ tertile  │
    │ ---   ┆ ---      │
    │ i64   ┆ enum     │
    ╞═══════╪══════════╡
    │ 12    ┆ [12, 44] │
    │ 45    ┆ [45, 77] │
    │ 67    ┆ [45, 77] │
    │ 89    ┆ [78, 95] │
    │ 95    ┆ [78, 95] │
    │ 23    ┆ [12, 44] │
    │ 78    ┆ [78, 95] │
    └───────┴──────────┘

### `quantiles(probs, raw=False, left_closed=True, extend=False, fmt=None, labels=None)`

Cut at specific quantile probabilities. Default labels show percentages;
pass `raw=True` for actual cut values.

``` python
scores.ps.with_columns(
    pl.col("score").ps_chop.quantiles([0.25, 0.75]).alias("iqr_group")
)
```

    shape: (7, 2)
    ┌───────┬─────────────┐
    │ score ┆ iqr_group   │
    │ ---   ┆ ---         │
    │ i64   ┆ enum        │
    ╞═══════╪═════════════╡
    │ 12    ┆ [0%, 25%)   │
    │ 45    ┆ [25%, 75%)  │
    │ 67    ┆ [25%, 75%)  │
    │ 89    ┆ [75%, 100%] │
    │ 95    ┆ [75%, 100%] │
    │ 23    ┆ [0%, 25%)   │
    │ 78    ┆ [25%, 75%)  │
    └───────┴─────────────┘

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

### How FrameExpr stays efficient

`ps.with_columns` resolves each `FrameExpr` in two phases. First it runs
a small aggregation (e.g. `unique().sort()` to discover categories)
against the *current* lazy plan — so any preceding `.filter()` or
`.select()` is already embedded and Polars’ predicate/projection
pushdown keeps the peek cheap. Then it uses the result to build a
concrete `pl.Expr` (e.g. `.cast(pl.Enum(["a", "b", "c"]))`) that goes
back into the lazy plan and executes normally.

``` python
# Only the filtered rows are scanned for category discovery;
# the cast itself remains lazy.
lf = pl.scan_parquet("events.parquet")
result = (
    lf.filter(pl.col("country") == "DE")
      .ps.with_columns(pl.col("status").ps_enum.make())
      .filter(pl.col("status") == "active")
      .collect()
)
```

See the `FrameExpr` docstring for the full explanation, including when
the peek is larger and notes on parallel evaluation.

------------------------------------------------------------------------

## License

MIT
