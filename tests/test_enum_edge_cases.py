import polars as pl

import polarstation  # noqa: F401

EMPTY = pl.DataFrame({"x": pl.Series([], dtype=pl.String)})
ALL_NULL = pl.DataFrame({"x": pl.Series([None, None, None], dtype=pl.String)})
NO_MATCH = pl.DataFrame({"n": [1, 2, 3]})  # no String columns
BASE = pl.DataFrame({"x": ["a", "b", "b", "c", "c", "c"]})


# ── empty DataFrame ───────────────────────────────────────────────────────────


def test_make_empty_df():
    r = EMPTY.ps.with_columns(pl.col("x").ps_enum.make())
    assert r["x"].dtype == pl.Enum([])
    assert len(r) == 0


def test_lump_empty_df():
    r = EMPTY.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.lump(n=2))
    assert isinstance(r["x"].dtype, pl.Enum)
    assert len(r) == 0


def test_reorder_empty_df():
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.String), "y": pl.Series([], dtype=pl.Float64)})
    r = df.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.reorder(pl.col("y")))
    assert isinstance(r["x"].dtype, pl.Enum)
    assert len(r) == 0


def test_infreq_empty_df():
    r = EMPTY.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.infreq())
    assert isinstance(r["x"].dtype, pl.Enum)


def test_rev_empty_df():
    r = EMPTY.ps.with_columns(pl.col("x").ps_enum.make().ps_enum.rev())
    assert r["x"].dtype == pl.Enum([])


def test_drop_unused_empty_df():
    r = EMPTY.ps.with_columns(pl.col("x").ps_enum.make(categories=["a", "b"]).ps_enum.drop_unused())
    assert r["x"].dtype == pl.Enum([])


# ── all-null column ───────────────────────────────────────────────────────────


def test_make_all_null():
    r = ALL_NULL.ps.with_columns(pl.col("x").ps_enum.make())
    assert r["x"].dtype == pl.Enum([])
    assert r["x"].null_count() == 3


def test_make_all_null_explicit_cats():
    r = ALL_NULL.ps.with_columns(pl.col("x").ps_enum.make(categories=["a", "b"]))
    assert r["x"].dtype == pl.Enum(["a", "b"])
    assert r["x"].null_count() == 3


def test_lump_all_null():
    r = ALL_NULL.ps.with_columns(pl.col("x").ps_enum.make(categories=["a", "b"]).ps_enum.lump(n=1))
    assert r["x"].null_count() == 3
    assert "Other" not in r["x"].dtype.categories


def test_reorder_all_null():
    df = pl.DataFrame({
        "x": pl.Series([None, None], dtype=pl.String),
        "y": [1.0, 2.0],
    })
    r = df.ps.with_columns(
        pl.col("x").ps_enum.make(categories=["a", "b"]).ps_enum.reorder(pl.col("y"), missing="last")
    )
    assert r["x"].null_count() == 2


def test_missing_to_category_all_null():
    r = ALL_NULL.ps.with_columns(
        pl.col("x").ps_enum.make(categories=["a"]).ps_enum.missing_to_category("NA")
    )
    assert r["x"].null_count() == 0
    assert r["x"].to_list() == ["NA", "NA", "NA"]


def test_drop_unused_all_null():
    r = ALL_NULL.ps.with_columns(
        pl.col("x").ps_enum.make(categories=["a", "b"]).ps_enum.drop_unused()
    )
    assert r["x"].dtype == pl.Enum([])
    assert r["x"].null_count() == 3


# ── no matching columns ───────────────────────────────────────────────────────


def test_no_matching_columns_is_noop():
    r = NO_MATCH.ps.with_columns(pl.col(pl.String).ps_enum.make())
    assert r.schema == NO_MATCH.schema
    assert r.equals(NO_MATCH)


# ── LazyFrame ─────────────────────────────────────────────────────────────────


def test_lazyframe_make():
    r = BASE.lazy().ps.with_columns(pl.col("x").ps_enum.make()).collect()
    assert r["x"].dtype == pl.Enum(["a", "b", "c"])


def test_lazyframe_chained():
    r = BASE.lazy().ps.with_columns(
        pl.col("x").ps_enum.make().ps_enum.infreq()
    ).collect()
    assert r["x"].dtype.categories.to_list() == ["c", "b", "a"]


def test_lazyframe_returns_lazyframe():
    result = BASE.lazy().ps.with_columns(pl.col("x").ps_enum.make())
    assert isinstance(result, pl.LazyFrame)


def test_lazyframe_plain_expr():
    r = BASE.lazy().ps.with_columns(pl.col("x").str.to_uppercase()).collect()
    assert r["x"].to_list() == ["A", "B", "B", "C", "C", "C"]
