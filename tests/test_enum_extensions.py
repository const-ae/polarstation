import polars as pl
import pytest

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


def test_make_integer_sorts_numerically():
    r = pl.DataFrame({"x": [1, 2, 10, 20]}).ps.with_columns(pl.col("x").ps_enum.make())
    assert r["x"].dtype.categories.to_list() == ["1", "2", "10", "20"]


def test_make_date_sorts_chronologically():
    import datetime
    df = pl.DataFrame({"x": [datetime.date(2020, 1, 1), datetime.date(2019, 6, 1), datetime.date(2021, 3, 15)]})
    r = df.ps.with_columns(pl.col("x").ps_enum.make())
    assert r["x"].dtype.categories.to_list() == ["2019-06-01", "2020-01-01", "2021-03-15"]


# ── unify ─────────────────────────────────────────────────────────────────────


def test_unify_basic():
    df = pl.DataFrame({
        "x": pl.Series(["a", "b"], dtype=pl.Enum(["a", "b"])),
        "y": pl.Series(["b", "c"], dtype=pl.Enum(["b", "c"])),
    })
    r = df.ps.with_columns(pl.col("x", "y").ps_enum.unify())
    assert r["x"].dtype == pl.Enum(["a", "b", "c"])
    assert r["y"].dtype == pl.Enum(["a", "b", "c"])


def test_unify_preserves_values():
    df = pl.DataFrame({
        "x": pl.Series(["a", "b"], dtype=pl.Enum(["a", "b"])),
        "y": pl.Series(["b", "c"], dtype=pl.Enum(["b", "c"])),
    })
    r = df.ps.with_columns(pl.col("x", "y").ps_enum.unify())
    assert r["x"].to_list() == ["a", "b"]
    assert r["y"].to_list() == ["b", "c"]


def test_unify_first_seen_ordering():
    df = pl.DataFrame({
        "x": pl.Series(["a"], dtype=pl.Enum(["c", "a"])),
        "y": pl.Series(["b"], dtype=pl.Enum(["b", "a"])),
    })
    r = df.ps.with_columns(pl.col("x", "y").ps_enum.unify())
    assert r["x"].dtype.categories.to_list() == ["c", "a", "b"]


def test_unify_already_same():
    df = pl.DataFrame({
        "x": pl.Series(["a"], dtype=pl.Enum(["a", "b"])),
        "y": pl.Series(["b"], dtype=pl.Enum(["a", "b"])),
    })
    r = df.ps.with_columns(pl.col("x", "y").ps_enum.unify())
    assert r["x"].dtype == r["y"].dtype == pl.Enum(["a", "b"])


def test_unify_single_column_unchanged():
    df = pl.DataFrame({"x": pl.Series(["a", "b"], dtype=pl.Enum(["a", "b"]))})
    r = df.ps.with_columns(pl.col("x").ps_enum.unify())
    assert r["x"].dtype == pl.Enum(["a", "b"])


def test_unify_requires_enum():
    df = pl.DataFrame({"x": ["a", "b"]})
    with pytest.raises(TypeError):
        df.ps.with_columns(pl.col("x").ps_enum.unify())


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


def test_add_categories_before_index():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.add_categories(["d"], before=1))
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


# ── rename ───────────────────────────────────────────────────────────────────


def test_rename_dict():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.rename({"a": "A", "c": "C"}))
    assert r["x"].dtype.categories.to_list() == ["A", "b", "C"]
    assert r["x"].drop_nulls().to_list().count("A") == 1
    assert r["x"].drop_nulls().to_list().count("C") == 7


def test_rename_callable():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.rename(str.upper))
    assert r["x"].dtype.categories.to_list() == ["A", "B", "C"]


def test_rename_preserves_nulls():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.rename({"a": "A"}))
    assert r["x"].null_count() == 2


def test_rename_partial_mapping():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.rename({"b": "bee"}))
    assert "b" not in r["x"].dtype.categories
    assert "a" in r["x"].dtype.categories
    assert "c" in r["x"].dtype.categories


def test_rename_strict_raises_on_unknown_key():
    with pytest.raises(ValueError, match="strict"):
        BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.rename({"z": "Z"}))


def test_rename_strict_false_ignores_unknown_key():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.rename({"z": "Z"}, strict=False))
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


# ── move ──────────────────────────────────────────────────────────────────────


def test_move_to_front():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.move("c"))
    assert r["x"].dtype.categories.to_list() == ["c", "a", "b"]


def test_move_to_end():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.move("a", before=None))
    assert r["x"].dtype.categories.to_list() == ["b", "c", "a"]


def test_move_before_index():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.move("a", before=1))
    assert r["x"].dtype.categories.to_list() == ["b", "a", "c"]


def test_move_negative_index():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.move("a", before=-1))
    assert r["x"].dtype.categories.to_list() == ["b", "a", "c"]


def test_move_large_index_is_end():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.move("a", before=99))
    assert r["x"].dtype.categories.to_list() == ["b", "c", "a"]


def test_move_multiple():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.move("c", "a"))
    assert r["x"].dtype.categories.to_list() == ["c", "a", "b"]


def test_move_preserves_values():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.move("c"))
    assert r["x"].to_list() == BASE["x"].cast(pl.Enum(["a", "b", "c"])).to_list()


def test_move_preserves_nulls():
    r = BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.move("c"))
    assert r["x"].null_count() == 2


def test_move_unknown_raises():
    with pytest.raises(ValueError, match="move"):
        BASE.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.move("z"))


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

# ── auto-make from String / Categorical ──────────────────────────────────────


_STR = pl.DataFrame({"x": ["a", "b", "b", "c", "c", "c"], "y": [1, 2, 3, 4, 5, 6]})
_CAT = pl.DataFrame({"x": pl.Series(["a", "b", "b", "c", "c", "c"], dtype=pl.Categorical)})


def test_lump_string_input():
    r = _STR.ps.with_columns(pl.col("x").ps_enum.lump(n=2))
    assert r["x"].dtype == pl.Enum(["b", "c", "Other"])


def test_lump_categorical_input():
    r = _CAT.ps.with_columns(pl.col("x").ps_enum.lump(n=2))
    assert r["x"].dtype == pl.Enum(["b", "c", "Other"])


def test_rename_string_input():
    r = _STR.ps.with_columns(pl.col("x").ps_enum.rename({"a": "A"}))
    assert r["x"].dtype == pl.Enum(["A", "b", "c"])
    assert r["x"].drop_nulls().to_list().count("A") == 1


def test_rename_categorical_input():
    r = _CAT.ps.with_columns(pl.col("x").ps_enum.rename(str.upper))
    assert r["x"].dtype == pl.Enum(["A", "B", "C"])


def test_reorder_string_input():
    r = _STR.ps.with_columns(pl.col("x").ps_enum.reorder("y"))
    assert r["x"].dtype.categories.to_list() == ["a", "b", "c"]


def test_infreq_string_input():
    r = _STR.ps.with_columns(pl.col("x").ps_enum.infreq())
    assert r["x"].dtype.categories.to_list() == ["c", "b", "a"]


def test_infreq_categorical_input():
    r = _CAT.ps.with_columns(pl.col("x").ps_enum.infreq())
    assert r["x"].dtype.categories.to_list() == ["c", "b", "a"]
