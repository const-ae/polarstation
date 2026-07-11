# v0.2.2 (2026-07-11)

* Add `ps.F`, `ps.B`, `ps.E` for calling arbitrary functions (numpy, scipy, plain Python, ...)
  directly on expressions, in place of hand-rolled `pl.struct(...).map_batches(...)`/
  `map_elements(...)`.
* Add `ps.format`, `ps.fmt_col`, and `Expr.ps_str.format` — an `Expr`-aware `str.format`. Any
  `{field:spec}` whose value is a `pl.Expr` is formatted per row via Python's own `format()`, so
  the full format-spec mini-language works without reimplementing it. `ps.fmt_col(...)` lets a
  column be embedded directly inside a real f-string without touching `pl.Expr.__format__` itself.
  `Expr.ps_str.format` additionally unpacks a Struct column's fields as named template arguments.
* `FrameExpr` (and therefore any `ps_enum`/`ps_chop`/`ps.apply`/`ps.F` expression) now supports
  `.over(*by)`, a group-aware equivalent of `pl.Expr.over()`.
* Fix `ps_enum.infreq()`/`ps_enum.reorder()` raising a `DuplicateError` when applied to multiple
  columns at once (e.g. `pl.col('a', 'b').ps_enum.infreq()`). `reorder()`'s `by` now also raises a
  clear `ValueError` (instead of crashing) if a `by` expression resolves to more than one column.

# v0.2.1 (2026-07-01)

* Add `ps_enum.to_level()` and refactor enum handling to avoid unnecessary `to_physical()` calls.
* Fix `ps_chop.chop()` on older Polars versions.

# v0.2.0 (2026-06-04)

* Rename `ps_enum.relabel` to `ps_enum.rename`.
* Add `ps_enum.move()` and `ps_enum.unify()`.
* `ps_enum.rename()` can now merge multiple factor levels into one.
* `ps_enum.make()` uses natural sort order for discovered categories instead of lexical order.
* Refactor column resolution to consistently alias inside the resolver across all `ps_enum`/
  `ps_chop` functions.

# v0.1.5 (2026-05-30)

* Clarify which `ps_enum` functions accept plain `String` columns (not just `Enum`/`Categorical`).

# v0.1.4 (2026-05-26)

* Add `ps.select`, the `FrameExpr`-aware counterpart of `ps.with_columns` for `.select()`.
* Relax the minimum supported Python version to 3.10.

# v0.1.0–0.1.3 (2026-05-25)

* Initial release: `ps_enum` (Enum creation, reordering, lumping rare levels, replacing missing
  values, ...), `ps_chop` (binning numeric/temporal/categorical columns, inspired by R's
  [santoku](https://hughjonesd.github.io/santoku/)), `ps.apply` for custom functions that need to
  peek at the full `LazyFrame`, and the underlying `FrameExpr`/two-phase peek-then-resolve design.
* Documentation site and package metadata fixes.
