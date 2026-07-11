"""Tests for ps.F, ps.B, ps.E — calling arbitrary functions from expressions."""

import math

import numpy as np
import polars as pl
import pytest

import polarstation as ps  # noqa: F401 — registers namespaces

# ── ps.F ────────────────────────────────────────────────────────────────────


def test_f_multi_arg_numpy_ufunc():
  df = pl.DataFrame({"y": [1.0, 0.0, -1.0], "x": [1.0, 1.0, 1.0]})
  r = df.ps.with_columns(angle=ps.F(np.arctan2)(pl.col("y"), pl.col("x")))
  assert r["angle"].to_list() == pytest.approx(
    [math.atan2(1.0, 1.0), math.atan2(0.0, 1.0), math.atan2(-1.0, 1.0)]
  )


def test_f_single_arg():
  df = pl.DataFrame({"x": [1.0, 4.0, 9.0]})
  r = df.ps.with_columns(root=ps.F(np.sqrt)(pl.col("x")))
  assert r["root"].to_list() == pytest.approx([1.0, 2.0, 3.0])


def test_f_dtype_matches_real_output():
  """No return_dtype is needed: dtype comes from the real (int) result."""
  df = pl.DataFrame({"a": [1, 2, 3]})
  r = df.ps.with_columns(doubled=ps.F(lambda s: s * 2)(pl.col("a")))
  assert r["doubled"].dtype == pl.Int64
  assert r["doubled"].to_list() == [2, 4, 6]


def test_f_fn_called_exactly_once():
  calls = []

  def fn(s):
    calls.append(len(s))
    return s * 2.0

  df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
  df.ps.with_columns(ps.F(fn)(pl.col("a")))
  assert calls == [3]


def test_f_lazyframe_respects_preceding_filter():
  df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
  r = (
    df.lazy()
    .filter(pl.col("a") > 2.0)
    .ps.with_columns(doubled=ps.F(lambda s: s * 2.0)(pl.col("a")))
    .collect()
  )
  assert r["doubled"].to_list() == pytest.approx([6.0, 8.0])


def test_f_non_expr_arg_passed_through_unchanged():
  """A plain (non-Expr) argument is forwarded to fn as-is, not resolved as a column."""
  df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
  r = df.ps.with_columns(scaled=ps.F(lambda s, factor: s * factor)(pl.col("a"), 10))
  assert r["scaled"].to_list() == pytest.approx([10.0, 20.0, 30.0])


def test_f_kwargs_supported():
  """Expr and non-Expr keyword arguments both work."""
  df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
  r = df.ps.with_columns(scaled=ps.F(lambda s, factor: s * factor)(pl.col("a"), factor=10))
  assert r["scaled"].to_list() == pytest.approx([10.0, 20.0, 30.0])


def test_f_no_expr_args_at_all():
  """fn with no pl.Expr arguments still runs (via the literal-name fallback)."""
  df = pl.DataFrame({"a": [1.0]})
  r = df.ps.with_columns(constant=ps.F(lambda: [42.0])())
  assert r["constant"].to_list() == [42.0]


# ── ps.B ────────────────────────────────────────────────────────────────────


def test_b_multi_arg_numpy_ufunc():
  df = pl.DataFrame({"y": [1.0, 0.0, -1.0], "x": [1.0, 1.0, 1.0]})
  r = df.select(angle=ps.B(np.arctan2, return_dtype=pl.Float64)(pl.col("y"), pl.col("x")))
  assert r["angle"].to_list() == pytest.approx(
    [math.atan2(1.0, 1.0), math.atan2(0.0, 1.0), math.atan2(-1.0, 1.0)]
  )


def test_b_stays_lazy():
  df = pl.DataFrame({"a": [1.0, 2.0]})
  expr = ps.B(lambda s: s * 2.0, return_dtype=pl.Float64)(pl.col("a"))
  assert isinstance(expr, pl.Expr)
  lf = df.lazy().with_columns(doubled=expr)
  assert isinstance(lf, pl.LazyFrame)
  assert lf.collect()["doubled"].to_list() == pytest.approx([2.0, 4.0])


# ── ps.E ────────────────────────────────────────────────────────────────────


def test_e_single_arg_scalar_fn():
  df = pl.DataFrame({"a": [1.0, math.e]})
  r = df.select(r=ps.E(math.log, return_dtype=pl.Float64)(pl.col("a")))
  assert r["r"].to_list() == pytest.approx([0.0, 1.0])


def test_e_multi_arg_scalar_fn():
  df = pl.DataFrame({"y": [1.0, 0.0, -1.0], "x": [1.0, 1.0, 1.0]})
  r = df.select(angle=ps.E(math.atan2, return_dtype=pl.Float64)(pl.col("y"), pl.col("x")))
  assert r["angle"].to_list() == pytest.approx(
    [math.atan2(1.0, 1.0), math.atan2(0.0, 1.0), math.atan2(-1.0, 1.0)]
  )


def test_e_null_argument_reaches_fn_as_none():
  """skip_nulls does not skip individual-null-field structs — fn must handle None itself."""
  df = pl.DataFrame({"a": [1.0, None]})
  with pytest.raises(TypeError):
    df.select(ps.E(math.log, return_dtype=pl.Float64)(pl.col("a")))


def test_e_raises_on_mixed_return_type_instead_of_corrupting():
  """Unlike np.vectorize, map_elements raises rather than silently truncating."""

  def f(a):
    return a if a < 2 else a / 2.0

  df = pl.DataFrame({"a": [1, 2, 3, 4]})
  with pytest.raises(pl.exceptions.SchemaError):
    df.select(ps.E(f)(pl.col("a")))
