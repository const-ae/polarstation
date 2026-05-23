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
    """
    expr = self._expr

    def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
      return [fn(lf, col) for col in lf.select(expr).collect_schema().names()]

    return FrameExpr(expr, resolver)
