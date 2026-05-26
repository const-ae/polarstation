import polars as pl
import pytest

import polarstation  # noqa: F401 — registers ps namespace

BASE = pl.DataFrame({"x": ["a", "b", "b", "c", "c", "c"], "y": [1, 2, 3, 4, 5, 6]})


# ── DataFrame ─────────────────────────────────────────────────────────────────


def test_select_frame_expr():
    r = BASE.ps.select(pl.col("x").ps_enum.make())
    assert r.columns == ["x"]
    assert r["x"].dtype == pl.Enum(["a", "b", "c"])


def test_select_drops_unmentioned_columns():
    r = BASE.ps.select(pl.col("x").ps_enum.make())
    assert "y" not in r.columns


def test_select_plain_expr():
    r = BASE.ps.select(pl.col("x"), pl.col("y") * 2)
    assert r["y"].to_list() == [2, 4, 6, 8, 10, 12]


def test_select_mixed_frame_expr_and_plain():
    r = BASE.ps.select(pl.col("x").ps_enum.make(), pl.col("y"))
    assert r.columns == ["x", "y"]
    assert isinstance(r["x"].dtype, pl.Enum)
    assert r["y"].to_list() == BASE["y"].to_list()


def test_select_named_expr():
    r = BASE.ps.select(animal=pl.col("x").ps_enum.make())
    assert r.columns == ["animal"]
    assert isinstance(r["animal"].dtype, pl.Enum)


def test_select_multiple_frame_exprs():
    df = pl.DataFrame({"x": ["a", "b", "a"], "z": ["p", "p", "q"]})
    r = df.ps.select(pl.col("x").ps_enum.make(), pl.col("z").ps_enum.make())
    assert isinstance(r["x"].dtype, pl.Enum)
    assert isinstance(r["z"].dtype, pl.Enum)
    assert r.columns == ["x", "z"]


def test_select_string_col_passthrough():
    r = BASE.ps.select("x", "y")
    assert r.columns == ["x", "y"]
    assert r.equals(BASE)


def test_select_returns_dataframe():
    r = BASE.ps.select(pl.col("x").ps_enum.make())
    assert isinstance(r, pl.DataFrame)


def test_select_chained_frame_exprs():
    r = BASE.ps.select(pl.col("x").ps_enum.make().ps_enum.infreq())
    assert r["x"].dtype.categories.to_list() == ["c", "b", "a"]


# ── LazyFrame ─────────────────────────────────────────────────────────────────


def test_select_lazyframe_frame_expr():
    r = BASE.lazy().ps.select(pl.col("x").ps_enum.make()).collect()
    assert r.columns == ["x"]
    assert r["x"].dtype == pl.Enum(["a", "b", "c"])


def test_select_lazyframe_drops_unmentioned_columns():
    r = BASE.lazy().ps.select(pl.col("x").ps_enum.make()).collect()
    assert "y" not in r.columns


def test_select_lazyframe_returns_lazyframe():
    result = BASE.lazy().ps.select(pl.col("x").ps_enum.make())
    assert isinstance(result, pl.LazyFrame)


def test_select_lazyframe_mixed():
    r = BASE.lazy().ps.select(pl.col("x").ps_enum.make(), pl.col("y")).collect()
    assert r.columns == ["x", "y"]
    assert isinstance(r["x"].dtype, pl.Enum)


def test_select_lazyframe_plain_expr():
    r = BASE.lazy().ps.select(pl.col("y") * 10).collect()
    assert r["y"].to_list() == [10, 20, 30, 40, 50, 60]


def test_select_preserves_filter_pushdown():
    r = (
        BASE.lazy()
        .filter(pl.col("y") > 3)
        .ps.select(pl.col("x").ps_enum.make())
        .collect()
    )
    # Only rows with y>3: x values are c, c, c → single category
    assert r["x"].dtype == pl.Enum(["c"])
    assert len(r) == 3
