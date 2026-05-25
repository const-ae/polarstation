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


# ── set_categories / drop_unused / add_categories ────────────────────────────


def test_set_categories_reorders():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.set_categories(["c", "b", "a"]))
    assert r["x"].dtype.categories.to_list() == ["c", "b", "a"]


def test_set_categories_drops_to_null():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.set_categories(["a", "c"]))
    assert "b" not in r["x"].dtype.categories
    assert r["x"].null_count() == 5  # 3 b's + 2 original nulls


def test_set_categories_preserves_nulls():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.set_categories(["a", "b", "c"]))
    assert r["x"].null_count() == 2


def test_drop_unused_removes_empty_cats():
    r = BASE.ps.with_columns(
        pl.col("x").ps_enum.make(categories=["a", "b", "c", "d"]).ps_enum.drop_unused()
    )
    assert r["x"].dtype.categories.to_list() == ["a", "b", "c"]


def test_drop_unused_preserves_order():
    r = BASE.ps.with_columns(
        pl.col("x").ps_enum.make(categories=["c", "d", "b", "a"]).ps_enum.drop_unused()
    )
    assert r["x"].dtype.categories.to_list() == ["c", "b", "a"]


def test_add_categories_appends_by_default():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.add_categories(["d", "e"]))
    assert r["x"].dtype.categories.to_list() == ["a", "b", "c", "d", "e"]


def test_add_categories_after_index():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.add_categories(["d"], after=0))
    assert r["x"].dtype.categories.to_list() == ["a", "d", "b", "c"]


def test_add_categories_no_value_change():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.add_categories(["z"]))
    assert r["x"].null_count() == 2
    assert "z" not in r["x"].drop_nulls().to_list()


# ── missing_to_category / category_to_missing ────────────────────────────────


def test_missing_to_category_basic():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.missing_to_category("NA"))
    assert "NA" in r["x"].dtype.categories
    assert r["x"].null_count() == 0
    assert r["x"].to_list().count("NA") == 2


def test_missing_to_category_appended_last():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.missing_to_category("NA"))
    assert r["x"].dtype.categories.to_list()[-1] == "NA"


def test_missing_to_category_existing_category():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.missing_to_category("a"))
    assert r["x"].dtype.categories.to_list() == ["a", "b", "c"]  # unchanged
    assert r["x"].null_count() == 0
    assert (r["x"] == "a").sum() == BASE["x"].is_null().sum() + (BASE["x"] == "a").sum()


def test_category_to_missing_basic():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.category_to_missing("b"))
    assert "b" not in r["x"].dtype.categories
    assert r["x"].null_count() == 5  # 3 b's + 2 original nulls


def test_category_to_missing_raises_if_absent():
    with pytest.raises(ValueError):
        BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.category_to_missing("z"))


def test_missing_to_category_roundtrip():
    r = BASE.ps.with_columns(
        pl.col("x").ps_enum.make().ps_enum.missing_to_category("NA").ps_enum.category_to_missing("NA")
    )
    assert r["x"].null_count() == 2
    assert "NA" not in r["x"].dtype.categories


# ── relabel ───────────────────────────────────────────────────────────────────


def test_relabel_dict():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.relabel({"a": "A", "c": "C"}))
    assert r["x"].dtype.categories.to_list() == ["A", "b", "C"]
    assert r["x"].drop_nulls().to_list().count("A") == 1
    assert r["x"].drop_nulls().to_list().count("C") == 7


def test_relabel_callable():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.relabel(str.upper))
    assert r["x"].dtype.categories.to_list() == ["A", "B", "C"]


def test_relabel_preserves_nulls():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.relabel({"a": "A"}))
    assert r["x"].null_count() == 2


def test_relabel_partial_mapping():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.relabel({"b": "bee"}))
    assert "b" not in r["x"].dtype.categories
    assert "a" in r["x"].dtype.categories
    assert "c" in r["x"].dtype.categories


def test_relabel_strict_raises_on_unknown_key():
    with pytest.raises(ValueError, match="strict"):
        BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.relabel({"z": "Z"}))


def test_relabel_strict_false_ignores_unknown_key():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.relabel({"z": "Z"}, strict=False))
    assert r["x"].dtype.categories.to_list() == ["a", "b", "c"]


# ── other ─────────────────────────────────────────────────────────────────────


def test_lump_basic():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.lump(n=2))
    assert r["x"].dtype == pl.Enum(["b", "c", "Other"])
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


def test_rev():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.rev())
    assert r["x"].dtype.categories.to_list() == ["c", "b", "a"]


def test_rev_double():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.rev().ps_enum.rev())
    assert r["x"].dtype.categories.to_list() == ["a", "b", "c"]


def test_infreq_default():
    # c=7, b=3, a=1 → most frequent first
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.infreq())
    assert r["x"].dtype.categories.to_list() == ["c", "b", "a"]


def test_infreq_ascending():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.infreq(descending=True))
    assert r["x"].dtype.categories.to_list() == ["a", "b", "c"]


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