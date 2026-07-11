import math
import textwrap
from typing import Literal

import polars as pl

from polarstation.frame_expr import FrameExpr, resolve_across_columns
from polarstation.func_extensions import format as ps_format


@pl.api.register_expr_namespace("ps_str")
class PolarstationStringExpression:
  def __init__(self, expr: pl.Expr) -> None:
    self._expr = expr

  def format(self, template: str) -> FrameExpr:
    """Format this column's values into ``template``, the way ``str.format`` formats a value.

    For a plain (non-Struct) column, this is a one-argument shorthand for
    ``ps.format(template, self)`` — ``template`` contains exactly one ``{...}`` field,
    referring to this expression's own values. For a Struct column, each field is
    instead unpacked as a named argument, keyed by its field name — ``template`` can
    then reference each one by name, the same way ``ps.format(template, **fields)`` would.

    Examples:
      ```{python}
      pl.DataFrame({"err": [0.5, 1.25, 12.0]}).ps.with_columns(
          msg=pl.col("err").ps_str.format("error={:.2f}")
      )
      ```

      ```{python}
      pl.DataFrame({"x": [1, 2], "y": [3.0, 4.0]}).ps.with_columns(
          msg=pl.struct(a="x", b="y").ps_str.format("a={a}, b={b:.1f}")
      )
      ```
    """
    expr = self._expr

    def handler(*, lf, name, col_ref, dtype):
      if isinstance(dtype, pl.Struct):
        fields = {f.name: col_ref.struct.field(f.name) for f in dtype.fields}
        return ps_format(template, **fields)
      return ps_format(template, col_ref)

    return FrameExpr(expr, resolve_across_columns(expr, handler))

  def count(self, pattern="") -> pl.Expr:
    r"""Count non-overlapping regex matches in each string.

    Deprecated: thin wrapper around `pl.Expr.str.count_matches`; likely to be removed.

    Examples:
      pl.DataFrame({"x": ["hello world", "foo bar baz", ""]}).select(
          pl.col("x").ps_str.count(r"\b\w+\b").alias("word_count")
      )
    """
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

    Examples:
      text = pl.DataFrame({"x": ["A long sentence that exceeds the column width."]}).select(
          pl.col("x").ps_str.wrap(width=25)
      )['x'].to_list()
      text
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

  def trunc(
    self,
    width: int = 5,
    side: Literal["right", "left", "center"] = "right",
    placeholder: str = "…",
  ) -> pl.Expr:
    """Truncate each string to fit within `width` characters.

    Collapses whitespace and appends `placeholder` when the text is cut.

    Args:
      width: Maximum length of the result, including the placeholder.
      side: Which side to truncate — 'right' (default), 'left', or 'center'.
      placeholder: String inserted where the text is cut.

    Examples:
      pl.DataFrame({"x": ["short", "a much longer string"]}).select(
          pl.col("x").ps_str.trunc(width=10)
      )
    """
    placeholder_width = len(placeholder)
    if placeholder_width > width:
      raise ValueError(
        f"placeholder width ({placeholder_width}) is larger than maximum width ({width})"
      )
    free_width = width - placeholder_width

    str_len = self._expr.str.len_chars()
    too_long = self._expr.is_not_null() & (str_len > width)
    match side:
      case "right":
        str_mod = self._expr.str.slice(0, free_width) + placeholder
      case "left":
        str_mod = placeholder + self._expr.str.slice(str_len - free_width)
      case "center":
        str_mod = (
          self._expr.str.slice(0, math.ceil(free_width / 2))
          + placeholder
          + self._expr.str.slice(str_len - math.floor(free_width / 2))
        )
      case _:
        raise ValueError(f"Unknown 'side={side}' specification. It has to be right|left|center")

    return (
      pl.when(too_long).then(str_mod).otherwise(self._expr).alias(self._expr.meta.output_name())
    )
