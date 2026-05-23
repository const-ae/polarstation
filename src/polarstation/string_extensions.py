import textwrap

import polars as pl


@pl.api.register_expr_namespace("ps_str")
class PolarstationStringExpression:
  def __init__(self, expr: pl.Expr) -> None:
    self._expr = expr

  def count(self, pattern="") -> pl.Expr:
    """Count non-overlapping regex matches in each string."""
    return self._expr.str.count_matches(pattern)

  def wrap(
    self,
    width: int = 80,
    initial_indent: int = 0,
    subsequent_indent: int = 0,
    break_on_hyphens: bool = True,
    **kwargs,
  ) -> pl.Expr:
    """Wrap each string to at most `width` characters per line.

    Args:
      width: Maximum line length.
      initial_indent: Number of spaces prepended to the first line.
      subsequent_indent: Number of spaces prepended to every subsequent line.
      break_on_hyphens: Allow breaks at hyphens in compound words.
      **kwargs: Forwarded to `textwrap.fill`.
    """
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
    """Truncate each string to fit within `width` characters.

    Collapses whitespace and appends `placeholder` when the text is cut.

    Args:
      width: Maximum length of the result, including the placeholder.
      placeholder: String appended when the text is truncated.
      **kwargs: Forwarded to `textwrap.shorten`.
    """
    return self._expr.map_elements(
      lambda s: textwrap.shorten(s, width=width, placeholder=placeholder, **kwargs),
      return_dtype=pl.String,
    )
