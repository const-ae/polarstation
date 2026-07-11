"""Tests for ps.format, ps.fmt_col, and Expr.ps_str.format — Expr-aware str.format()."""

import polars as pl
import pytest

import polarstation as ps  # noqa: F401 — registers namespaces

# ── ps.format — explicit args/kwargs ──────────────────────────────────────────


def test_format_positional_expr_arg():
  df = pl.DataFrame({"err": [0.5, 1.25, 12.0]})
  r = df.with_columns(msg=ps.format("error={:.2f}", pl.col("err")))
  assert r["msg"].to_list() == ["error=0.50", "error=1.25", "error=12.00"]


def test_format_named_expr_kwarg():
  df = pl.DataFrame({"err": [0.5, 1.0]})
  r = df.with_columns(msg=ps.format("error={e:.1%}", e=pl.col("err")))
  assert r["msg"].to_list() == ["error=50.0%", "error=100.0%"]


def test_format_multiple_expr_args():
  df = pl.DataFrame({"err": [0.5, 1.25], "n": [3, 12]})
  r = df.with_columns(msg=ps.format("error={:.2f} (n={})", pl.col("err"), pl.col("n")))
  assert r["msg"].to_list() == ["error=0.50 (n=3)", "error=1.25 (n=12)"]


def test_format_plain_value_is_formatted_once_not_per_row():
  """A non-Expr argument is a literal, formatted immediately — same for every row."""
  df = pl.DataFrame({"err": [0.5, 1.25]})
  r = df.with_columns(msg=ps.format("error={:.2f} unit={}", pl.col("err"), "USD"))
  assert r["msg"].to_list() == ["error=0.50 unit=USD", "error=1.25 unit=USD"]


def test_format_escaped_braces():
  df = pl.DataFrame({"err": [0.5]})
  r = df.with_columns(msg=ps.format("{{literal}} err={:.2f}", pl.col("err")))
  assert r["msg"].to_list() == ["{literal} err=0.50"]


def test_format_no_fields_pure_literal():
  df = pl.DataFrame({"x": [1, 2]})
  r = df.with_columns(msg=ps.format("just text"))
  assert r["msg"].to_list() == ["just text", "just text"]


def test_format_only_plain_values_no_expr():
  df = pl.DataFrame({"x": [1]})
  r = df.with_columns(msg=ps.format("n={}", 5))
  assert r["msg"].to_list() == ["n=5"]


# ── Expr.ps_str.format — single-column fluent form ────────────────────────────


def test_ps_str_format_basic():
  df = pl.DataFrame({"err": [0.5, 1.25, 12.0]})
  r = df.ps.with_columns(msg=pl.col("err").ps_str.format("error={:.2f}"))
  assert r["msg"].to_list() == ["error=0.50", "error=1.25", "error=12.00"]


def test_ps_str_format_equivalent_to_format():
  df = pl.DataFrame({"err": [0.5, 1.25]})
  a = df.ps.with_columns(msg=pl.col("err").ps_str.format("val: {:.1f}"))
  b = df.with_columns(msg=ps.format("val: {:.1f}", pl.col("err")))
  assert a["msg"].to_list() == b["msg"].to_list()


# ── Expr.ps_str.format — Struct column, named fields ──────────────────────────


def test_ps_str_format_struct_named_fields():
  df = pl.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
  r = df.ps.with_columns(msg=pl.struct(a="x", b="y").ps_str.format("a={a}, b={b:.1f}"))
  assert r["msg"].to_list() == ["a=1, b=3.0", "a=2, b=4.0"]


def test_ps_str_format_struct_existing_column():
  """Works the same for a struct column that already exists, not just pl.struct(...) inline."""
  df = pl.DataFrame({"x": [1, 2], "y": [3.0, 4.0]}).with_columns(s=pl.struct(a="x", b="y"))
  r = df.ps.with_columns(msg=pl.col("s").ps_str.format("a={a}, b={b:.1f}"))
  assert r["msg"].to_list() == ["a=1, b=3.0", "a=2, b=4.0"]


# ── ps.fmt_col — embedding in a real f-string ─────────────────────────────────


def test_fmt_col_string_name():
  df = pl.DataFrame({"err": [0.5, 1.25], "n": [3, 12]})
  r = df.with_columns(msg=ps.format(f"error={ps.fmt_col('err'):.2f} (n={ps.fmt_col('n')})"))
  assert r["msg"].to_list() == ["error=0.50 (n=3)", "error=1.25 (n=12)"]


def test_fmt_col_accepts_expr_too():
  df = pl.DataFrame({"err": [0.5, 1.25]})
  r = df.with_columns(msg=ps.format(f"error={ps.fmt_col(pl.col('err') * 2):.2f}"))
  assert r["msg"].to_list() == ["error=1.00", "error=2.50"]


def test_fmt_col_equivalent_to_explicit_args_form():
  df = pl.DataFrame({"err": [0.5, 1.25]})
  a = df.with_columns(msg=ps.format(f"v={ps.fmt_col('err'):.2f}"))
  b = df.with_columns(msg=ps.format("v={:.2f}", pl.col("err")))
  assert a["msg"].to_list() == b["msg"].to_list()


def test_fmt_col_does_not_touch_pl_expr_format():
  """The whole point: pl.Expr.__format__ itself must stay untouched."""
  assert pl.Expr.__format__ is object.__format__
  with pytest.raises(TypeError):
    f"{pl.col('x'):.2f}"


def test_fmt_col_is_stateless_across_repeated_calls():
  """No shared registry — the same field can be embedded and resolved repeatedly."""
  df = pl.DataFrame({"err": [0.5, 1.25]})
  for _ in range(3):
    r = df.with_columns(msg=ps.format(f"v={ps.fmt_col('err'):.2f}"))
    assert r["msg"].to_list() == ["v=0.50", "v=1.25"]
