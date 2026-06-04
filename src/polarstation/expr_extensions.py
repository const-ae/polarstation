from collections.abc import Callable

import polars as pl

from polarstation.frame_expr import FrameExpr, resolve_across_columns


@pl.api.register_expr_namespace("ps")
class PolarstationExpression:
  def __init__(self, expr: pl.Expr) -> None:
    self._expr = expr

  def apply(self, fn: Callable[[pl.LazyFrame, pl.Expr], pl.Expr]) -> FrameExpr:
    """Apply a custom function with full LazyFrame context.

    Args:
      fn: Called as fn(lf, col_ref) → pl.Expr for each matched column.
          col_ref evaluates to the column's values in lf; it works correctly
          for any expression shape, including transforms and when/then/otherwise.
          Use col_ref.meta.output_name() when the string column name is needed.

    Examples:
      ```{python}
      def center_scale(lf: pl.LazyFrame, col_ref: pl.Expr) -> pl.Expr:
          stats = lf.select(
              col_ref.mean().alias("m"), col_ref.std().alias("s")
          ).collect()
          m, s = stats["m"][0], stats["s"][0]
          return (col_ref - m) / s

      pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).ps.with_columns(
          pl.col("x").ps.apply(center_scale)
      )
      ```

      ```{python}
      import math

      df = pl.DataFrame({
          "doc_id": [1, 1, 2, 2, 2],
          "term": ["cat", "dog", "cat", "cat", "bird"],
      })

      def idf(lf: pl.LazyFrame, col_ref: pl.Expr) -> pl.Expr:
          col_name = col_ref.meta.output_name()
          n = lf.select(pl.col("doc_id").n_unique()).collect().item()
          freq = lf.group_by(col_ref).agg(
              pl.col("doc_id").n_unique().alias("n")
          ).collect()
          scores = {r[col_name]: math.log(n / r["n"]) for r in freq.iter_rows(named=True)}
          return col_ref.replace_strict(
              list(scores), list(scores.values()), return_dtype=pl.Float64
          )

      df.ps.with_columns(pl.col("term").ps.apply(idf).alias("idf"))
      ```
    """
    expr = self._expr

    def handler(*, lf, name, col_ref, dtype):
      return fn(lf, col_ref).alias(name)

    return FrameExpr(expr, resolve_across_columns(expr, handler))
