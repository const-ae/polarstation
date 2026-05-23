import pytest
import polars as pl

import polarstation  # noqa: F401 — registers ps_enum namespace

_X = ["a"] + ["b"] * 3 + [None] * 2 + ["c"] * 7
BASE = pl.DataFrame({"x": _X})
# y gives a clear per-group aggregate: a=30, b=10, c=20
ORDER_DF = pl.DataFrame({"x": _X, "y": [30] + [10] * 3 + [20] * 9})
# c has all-null y → its aggregate will be null
NULL_AGG_DF = pl.DataFrame({"x": _X, "y": [30] + [10] * 3 + [None] * 9})


# ── make ──────────────────────────────────────────────────────────────────────


def test_make_derives_cats():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make())
    assert r["x"].dtype == pl.Enum(["a", "b", "c"])


def test_make_explicit_cats():
    r = BASE.ps.with_columns(
        pl.col("x").ps_enum.make(categories=["c", "b", "a"])
    )
    assert r["x"].dtype == pl.Enum(["c", "b", "a"])


def test_make_make_null_str():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make(make_null="b"))
    assert r["x"].to_list().count(None) == 5
    assert "b" not in r["x"].dtype.categories


def test_make_make_null_list():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make(make_null=["a", "b"]))
    assert r["x"].to_list().count(None) == 6
    assert r["x"].dtype == pl.Enum(["c"])


def test_make_invalid_cat_raises():
    with pytest.raises(Exception):
        BASE.ps.with_columns(pl.col("x").ps_enum.make(categories=["a", "b"]))


def test_make_multi_col():
    df = pl.DataFrame({"x": _X, "z": ["p"] * 7 + ["q"] * 6})
    r = df.ps.with_columns(pl.col(pl.String).ps_enum.make())
    assert isinstance(r["x"].dtype, pl.Enum)
    assert isinstance(r["z"].dtype, pl.Enum)


# ── other ─────────────────────────────────────────────────────────────────────


def test_lump_basic():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.lump(n=2))
    assert r["x"].dtype == pl.Enum(["c", "b", "Other"])
    assert "a" not in r["x"].to_list()
    assert r["x"].to_list().count("Other") == 1


def test_lump_custom_label():
    r = BASE.ps.with_columns(
        pl.col("x").ps_enum.make().ps_enum.lump(n=1, other_label="Misc")
    )
    assert "Misc" in r["x"].dtype.categories
    assert "Other" not in r["x"].dtype.categories


def test_lump_no_collapse():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.lump(n=10))
    assert "Other" not in r["x"].dtype.categories


# ── reorder ───────────────────────────────────────────────────────────────────


def test_reorder_ascending():
    # agg median: b=10, c=20, a=30 → asc: b, c, a
    r = ORDER_DF.ps.with_columns(
        pl.col("x").ps_enum.make().ps_enum.reorder(pl.col("y"))
    )
    assert r["x"].dtype.categories.to_list() == ["b", "c", "a"]


def test_reorder_descending():
    r = ORDER_DF.ps.with_columns(
        pl.col("x").ps_enum.make().ps_enum.reorder(pl.col("y"), descending=True)
    )
    assert r["x"].dtype.categories.to_list() == ["a", "c", "b"]


def test_reorder_custom_agg():
    r = ORDER_DF.ps.with_columns(
        pl.col("x").ps_enum.make().ps_enum.reorder(pl.col("y"), agg=pl.Expr.mean)
    )
    assert r["x"].dtype.categories.to_list() == ["b", "c", "a"]


def test_reorder_multi_by():
    # y breaks into two tiers; z breaks the tie within y=10
    df = pl.DataFrame({
        "x": ["a", "b", "b", "c"],
        "y": [10, 10, 10, 20],
        "z": [5, 1, 1, 3],
    })
    # a: y=10, z=5 / b: y=10, z=1 → b before a; c: y=20
    r = df.ps.with_columns(
        pl.col("x").ps_enum.make().ps_enum.reorder([pl.col("y"), pl.col("z")])
    )
    assert r["x"].dtype.categories.to_list() == ["b", "a", "c"]


def test_reorder_missing_last():
    r = NULL_AGG_DF.ps.with_columns(
        pl.col("x").ps_enum.make().ps_enum.reorder(pl.col("y"), missing="last")
    )
    assert r["x"].dtype.categories.to_list() == ["b", "a", "c"]


def test_reorder_missing_first():
    r = NULL_AGG_DF.ps.with_columns(
        pl.col("x").ps_enum.make().ps_enum.reorder(pl.col("y"), missing="first")
    )
    assert r["x"].dtype.categories.to_list() == ["c", "b", "a"]


def test_reorder_missing_drop_excludes_cat():
    # null out c first so dropping it from cats doesn't break the cast
    r = NULL_AGG_DF.ps.with_columns(
        pl.col("x").ps_enum.make(make_null="c")
        .ps_enum.reorder(pl.col("y"), missing="drop")
    )
    assert r["x"].dtype.categories.to_list() == ["b", "a"]


def test_reorder_missing_drop_raises_without_make_null():
    # c still in data but dropped from cats → cast error
    with pytest.raises(Exception):
        NULL_AGG_DF.ps.with_columns(
            pl.col("x").ps_enum.make().ps_enum.reorder(pl.col("y"), missing="drop")
        )