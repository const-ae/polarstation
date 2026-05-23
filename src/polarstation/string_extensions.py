import textwrap

import polars as pl


@pl.api.register_expr_namespace("ps_str")
class PolarstationStringExpression:
  def __init__(self, expr: pl.Expr) -> None:
    self._expr = expr

  def count(self, pattern="") -> pl.Expr:
    return self._expr.str.count_matches(pattern)

  def wrap(
    self,
    width: int = 80,
    initial_indent: int = 0,
    subsequent_indent: int = 0,
    break_on_hyphens: bool = True,
    **kwargs,
  ) -> pl.Expr:
    initial_indent_str = " " * initial_indent
    subsequent_indent_str = " " * subsequent_indent
    return self._expr.map_elements(
      lambda s: textwrap.fill(
        s,
        width=width,
        initial_indent=initial_indent_str,
        subsequent_indent=subsequent_indent_str,
        break_on_hyphens=break_on_hyphens,
        **kwargs,
      ),
      return_dtype=pl.String,
    )

  def shorten(
    self,
    width: int = 5,
    placeholder: str = "…",
    **kwargs,
  ) -> pl.Expr:
    return self._expr.map_elements(
      lambda s: textwrap.shorten(s, width=width, placeholder=placeholder, **kwargs),
      return_dtype=pl.String,
    )
