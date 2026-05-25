from collections.abc import Callable

import polars as pl

from polarstation.frame_expr import FrameExpr


@pl.api.register_expr_namespace("ps")
class PolarstationExpression:
  def __init__(self, expr: pl.Expr) -> None:
    self._expr = expr

  def apply(self, fn: Callable[[pl.LazyFrame, str], pl.Expr]) -> FrameExpr:
    """Apply a custom function with full LazyFrame context.

    Args:
      fn: Called as fn(lf, col_name) → pl.Expr for each matched column.

    Examples:
      ```{python}
      def center_scale(lf: pl.LazyFrame, col: str) -> pl.Expr:
          stats = lf.select(
              pl.col(col).mean().alias("m"), pl.col(col).std().alias("s")
          ).collect()
          m, s = stats["m"][0], stats["s"][0]
          return ((pl.col(col) - m) / s).alias(col)

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

      def idf(lf: pl.LazyFrame, col: str) -> pl.Expr:
          n = lf.select(pl.col("doc_id").n_unique()).collect().item()
          freq = lf.group_by(col).agg(
              pl.col("doc_id").n_unique().alias("n")
          ).collect()
          scores = {r[col]: math.log(n / r["n"]) for r in freq.iter_rows(named=True)}
          return pl.col(col).replace_strict(
              list(scores), list(scores.values()), return_dtype=pl.Float64
          )

      df.ps.with_columns(pl.col("term").ps.apply(idf).alias("idf"))
      ```
    """
    expr = self._expr

    def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
      return [fn(lf, col) for col in lf.select(expr).collect_schema().names()]

    return FrameExpr(expr, resolver)
