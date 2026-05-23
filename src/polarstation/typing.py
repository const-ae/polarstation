import polars as pl

# Mirrors polars.typing.IntoExpr (not yet public in polars 1.x).
# Strings are treated as column names (equivalent to pl.col(name)).
IntoExpr = pl.Expr | str | int | float | bool | pl.Series


def _into_expr(x: IntoExpr) -> pl.Expr:
  if isinstance(x, pl.Expr):
    return x
  if isinstance(x, str):
    return pl.col(x)
  return pl.lit(x)
