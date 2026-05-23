from collections.abc import Callable

import polars as pl

from polarstation.frame_expr import FrameExpr


@pl.api.register_expr_namespace("ps")
class PolarstationExpression:
  def __init__(self, expr: pl.Expr) -> None:
    self._expr = expr

  def apply(self, fn: Callable[[pl.DataFrame, str], pl.Expr]) -> FrameExpr:
    """Apply a custom function with full DataFrame context.

    Args:
      fn: Called as fn(df, col_name) → pl.Expr for each matched column.
    """
    expr = self._expr

    def resolver(df: pl.DataFrame) -> list[pl.Expr]:
      return [fn(df, col) for col in df.select(expr).columns]

    return FrameExpr(expr, resolver)
