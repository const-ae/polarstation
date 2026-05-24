import datetime
import math

import polars as pl
import pytest

import polarstation  # noqa: F401 — registers ps_chop namespace

DF10 = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
DF_TIES = pl.DataFrame({"x": [1.0, 2.0, 2.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
DF_NULL = pl.DataFrame({"x": [1.0, None, 3.0, 4.0, 5.0]})
DF_EMPTY = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64)})
DF_ALL_NULL = pl.DataFrame({"x": pl.Series([None, None, None], dtype=pl.Float64)})


def col(df, expr):
    return df.ps.with_columns(expr)["x"]


# ── chop ──────────────────────────────────────────────────────────────────────


def test_chop_labels_left_closed():
    # extend=True (default): outermost labels show ±∞
    s = col(DF10, pl.col("x").ps_chop.chop([3.0, 7.0]))
    assert s.to_list()[:2] == ["[-∞, 3)", "[-∞, 3)"]
    assert s.to_list()[2] == "[3, 7)"
    assert s.to_list()[-1] == "[7, ∞)"


def test_chop_labels_no_extend():
    # extend=False: outermost labels use data min/max
    s = col(DF10, pl.col("x").ps_chop.chop([3.0, 7.0], extend=False))
    assert s.to_list()[:2] == ["[1, 3)", "[1, 3)"]
    assert s.to_list()[2] == "[3, 7)"
    assert s.to_list()[-1] == "[7, 10]"


def test_chop_labels_right_closed():
    s = col(DF10, pl.col("x").ps_chop.chop([3.0, 7.0], left_closed=False))
    assert s.to_list()[0] == "(-∞, 3]"
    assert s.to_list()[3] == "(3, 7]"
    assert s.to_list()[-1] == "(7, ∞]"


def test_chop_custom_labels():
    s = col(DF10, pl.col("x").ps_chop.chop([5.0], labels=["low", "high"]))
    assert set(s.to_list()) == {"low", "high"}
    assert s[0] == "low"
    assert s[-1] == "high"


def test_chop_fmt():
    s = col(DF10, pl.col("x").ps_chop.chop([3.5], fmt=".1f"))
    assert "[3.5, ∞)" in s.to_list()


def test_chop_null_passthrough():
    s = col(DF_NULL, pl.col("x").ps_chop.chop([2.0]))
    assert s[1] is None


def test_chop_include_breaks():
    # extend=True (default): outer bounds are ±inf
    s = col(DF10, pl.col("x").ps_chop.chop([3.0, 7.0], return_struct=True))
    assert s.dtype == pl.Struct({"lo": pl.Float64, "hi": pl.Float64})
    rows = s.to_list()
    assert rows[0] == {"lo": float("-inf"), "hi": 3.0}
    assert rows[2] == {"lo": 3.0, "hi": 7.0}
    assert rows[-1] == {"lo": 7.0, "hi": float("inf")}


def test_chop_include_breaks_no_extend():
    # extend=False: outer bounds are data min/max
    s = col(DF10, pl.col("x").ps_chop.chop([3.0, 7.0], extend=False, return_struct=True))
    assert s.to_list()[0] == {"lo": 1.0, "hi": 3.0}
    assert s.to_list()[-1] == {"lo": 7.0, "hi": 10.0}


def test_chop_include_breaks_null():
    s = col(DF_NULL, pl.col("x").ps_chop.chop([2.0], return_struct=True))
    assert s[1] == {"lo": None, "hi": None}


# ── width ─────────────────────────────────────────────────────────────────────


def test_width_labels():
    s = col(DF10, pl.col("x").ps_chop.width(3.0))
    assert s[0] == "[1, 4)"
    assert s[3] == "[4, 7)"
    assert s[-1] == "[7, 10]"


def test_width_labels_extend():
    s = col(DF10, pl.col("x").ps_chop.width(3.0, extend=True))
    assert s[0] == "[-∞, 4)"
    assert s[-1] == "[7, ∞)"


def test_width_custom_start():
    s = col(DF10, pl.col("x").ps_chop.width(3.0, start=0.0))
    assert s[0] == "[0, 3)"   # 1.0 → first bin
    assert s[2] == "[3, 6)"   # 3.0 is at the break, goes into second bin (left_closed)
    assert s[3] == "[3, 6)"   # 4.0 → second bin


def test_width_right_closed():
    s = col(DF10, pl.col("x").ps_chop.width(3.0, left_closed=False))
    assert s[0] == "[1, 4]"   # 1.0 → first bin, special "[" for finite lo
    assert s[3] == "[1, 4]"   # 4.0 → also first bin (right-closed, 4 ≤ 4)
    assert s[4] == "(4, 7]"   # 5.0 → second bin


def test_width_null_passthrough():
    s = col(DF_NULL, pl.col("x").ps_chop.width(2.0))
    assert s[1] is None


def test_width_include_breaks():
    s = col(DF10, pl.col("x").ps_chop.width(3.0, return_struct=True))
    assert s.dtype == pl.Struct({"lo": pl.Float64, "hi": pl.Float64})
    assert s[0] == {"lo": 1.0, "hi": 4.0}
    assert s[3] == {"lo": 4.0, "hi": 7.0}
    assert s[-1] == {"lo": 7.0, "hi": 10.0}


def test_width_include_breaks_extend():
    s = col(DF10, pl.col("x").ps_chop.width(3.0, extend=True, return_struct=True))
    assert math.isinf(s[0]["lo"])
    assert math.isinf(s[-1]["hi"])


# ── n_elements ────────────────────────────────────────────────────────────────


def test_n_elements_labels():
    s = col(DF10, pl.col("x").ps_chop.n_elements(4))
    # groups: {1,2,3,4}, {5,6,7,8}, {9,10}
    assert s[0] == "[1, 5)"
    assert s[4] == "[5, 9)"
    assert s[-1] == "[9, 10]"


def test_n_elements_tail_split():
    # n=3 on 10 elements: groups of 3,3,3 + 1 leftover
    s = col(DF10, pl.col("x").ps_chop.n_elements(3, tail="split"))
    assert s[0] == "[1, 4)"
    assert s[3] == "[4, 7)"
    assert s[6] == "[7, 10)"
    assert s[9] == "[10, 10]"


def test_n_elements_tail_merge():
    s = col(DF10, pl.col("x").ps_chop.n_elements(3, tail="merge"))
    assert s[0] == "[1, 4)"
    assert s[3] == "[4, 7)"
    assert s[6] == "[7, 10]"
    assert s[9] == "[7, 10]"


def test_n_elements_ties_not_split():
    # Three 2.0s: boundary must advance past them, landing at 5.0
    s = col(DF_TIES, pl.col("x").ps_chop.n_elements(3))
    # First bin should include all three 2.0s (indices 1-3 in source)
    assert s[1] == s[2] == s[3]  # all 2.0s in same bin


def test_n_elements_extend():
    s = col(DF10, pl.col("x").ps_chop.n_elements(4, extend=True))
    assert s[0].startswith("[-∞")
    assert s[-1].endswith("∞)")


def test_n_elements_null_passthrough():
    s = col(DF_NULL, pl.col("x").ps_chop.n_elements(2))
    assert s[1] is None


def test_n_elements_include_breaks():
    s = col(DF10, pl.col("x").ps_chop.n_elements(4, return_struct=True))
    assert s.dtype == pl.Struct({"lo": pl.Float64, "hi": pl.Float64})
    assert s[0] == {"lo": 1.0, "hi": 5.0}
    assert s[4] == {"lo": 5.0, "hi": 9.0}


# ── n_groups ──────────────────────────────────────────────────────────────────


def test_n_groups_labels_raw():
    s = col(DF10, pl.col("x").ps_chop.n_groups(3))
    assert s[0] == "[1, 4)"
    assert s[3] == "[4, 7)"
    assert s[-1] == "[7, 10]"


def test_n_groups_labels_pct():
    s = col(DF10, pl.col("x").ps_chop.n_groups(3, raw=False))
    assert s[0] == "[0%, 33%)"
    assert s[-1] == "[67%, 100%]"


def test_n_groups_extend():
    s = col(DF10, pl.col("x").ps_chop.n_groups(3, extend=True))
    assert s[0] == "[-∞, 4)"
    assert s[-1] == "[7, ∞)"


def test_n_groups_extend_pct_unaffected():
    # extend has no effect on percentage labels
    s_ext = col(DF10, pl.col("x").ps_chop.n_groups(3, raw=False, extend=True))
    s_no_ext = col(DF10, pl.col("x").ps_chop.n_groups(3, raw=False, extend=False))
    assert s_ext.to_list() == s_no_ext.to_list()


def test_n_groups_null_passthrough():
    s = col(DF_NULL, pl.col("x").ps_chop.n_groups(2))
    assert s[1] is None


def test_n_groups_include_breaks():
    s = col(DF10, pl.col("x").ps_chop.n_groups(3, return_struct=True))
    assert s.dtype == pl.Struct({"lo": pl.Float64, "hi": pl.Float64})
    assert s[0] == {"lo": 1.0, "hi": 4.0}
    assert s[-1] == {"lo": 7.0, "hi": 10.0}


def test_n_groups_include_breaks_extend():
    s = col(DF10, pl.col("x").ps_chop.n_groups(3, return_struct=True, extend=True))
    assert math.isinf(s[0]["lo"])
    assert math.isinf(s[-1]["hi"])


# ── quantiles ─────────────────────────────────────────────────────────────────


def test_quantiles_labels_pct():
    s = col(DF10, pl.col("x").ps_chop.quantiles([0.25, 0.5, 0.75]))
    assert s[0] == "[0%, 25%)"
    assert s[4] == "[25%, 50%)"
    assert s[-1] == "[75%, 100%]"


def test_quantiles_labels_raw():
    s = col(DF10, pl.col("x").ps_chop.quantiles([0.25, 0.5, 0.75], raw=True))
    assert s[0] == "[1, 3.25)"
    assert s[-1] == "[7.75, 10]"


def test_quantiles_extend_raw():
    s = col(DF10, pl.col("x").ps_chop.quantiles([0.25, 0.5, 0.75], raw=True, extend=True))
    assert s[0] == "[-∞, 3.25)"
    assert s[-1] == "[7.75, ∞)"


def test_quantiles_extend_pct_unaffected():
    s_ext = col(DF10, pl.col("x").ps_chop.quantiles([0.5], raw=False, extend=True))
    s_no_ext = col(DF10, pl.col("x").ps_chop.quantiles([0.5], raw=False, extend=False))
    assert s_ext.to_list() == s_no_ext.to_list()


def test_quantiles_right_closed():
    s = col(DF10, pl.col("x").ps_chop.quantiles([0.5], left_closed=False))
    assert s[0] == "[0%, 50%]"
    assert s[-1] == "(50%, 100%]"


def test_quantiles_null_passthrough():
    s = col(DF_NULL, pl.col("x").ps_chop.quantiles([0.5]))
    assert s[1] is None


def test_quantiles_include_breaks_pct():
    s = col(DF10, pl.col("x").ps_chop.quantiles([0.25, 0.5, 0.75], return_struct=True))
    assert s.dtype == pl.Struct({"lo": pl.Float64, "hi": pl.Float64})
    assert s[0] == {"lo": 1.0, "hi": 3.25}
    assert s[-1] == {"lo": 7.75, "hi": 10.0}


def test_quantiles_include_breaks_raw_extend():
    s = col(DF10, pl.col("x").ps_chop.quantiles([0.5], raw=True, extend=True, return_struct=True))
    assert math.isinf(s[0]["lo"])
    assert math.isinf(s[-1]["hi"])


def test_quantiles_duplicate_breaks_deduplicated():
    # All same value → quantile boundaries collapse; we still get a single bin
    df = pl.DataFrame({"x": [5.0, 5.0, 5.0, 5.0]})
    s = col(df, pl.col("x").ps_chop.quantiles([0.25, 0.5, 0.75]))
    assert s.n_unique() == 1


# ── empty and all-null edge cases ─────────────────────────────────────────────


def test_chop_empty():
    assert col(DF_EMPTY, pl.col("x").ps_chop.chop([3.0])).len() == 0


def test_chop_all_null():
    s = col(DF_ALL_NULL, pl.col("x").ps_chop.chop([3.0]))
    assert s.is_null().all()
    assert s.dtype.categories.to_list() == ['[-∞, 3)', '[3, ∞)']


def test_chop_no_extend_empty():
    assert col(DF_EMPTY, pl.col("x").ps_chop.chop([3.0], extend=False)).len() == 0


def test_chop_no_extend_all_null():
    s = col(DF_ALL_NULL, pl.col("x").ps_chop.chop([3.0], extend=False))
    assert s.is_null().all()
    assert s.dtype.categories.to_list() == ['[-∞, 3)', '[3, ∞)']


def test_width_empty():
    s = col(DF_EMPTY, pl.col("x").ps_chop.width(3.0))
    assert s.len() == 0
    assert s.dtype.categories.to_list() == ['[-∞, ∞)']


def test_width_all_null():
    s = col(DF_ALL_NULL, pl.col("x").ps_chop.width(3.0))
    assert s.is_null().all()
    assert s.dtype.categories.to_list() == ['[-∞, ∞)']


def test_n_elements_empty():
    assert col(DF_EMPTY, pl.col("x").ps_chop.n_elements(3)).len() == 0


def test_n_elements_all_null():
    s = col(DF_ALL_NULL, pl.col("x").ps_chop.n_elements(3))
    assert s.is_null().all()
    assert s.dtype.categories.to_list() == ['[-∞, ∞)']


def test_n_groups_empty():
    assert col(DF_EMPTY, pl.col("x").ps_chop.n_groups(3)).len() == 0


def test_n_groups_all_null():
    s = col(DF_ALL_NULL, pl.col("x").ps_chop.n_groups(3))
    assert s.is_null().all()
    assert s.dtype.categories.to_list() == ['[-∞, ∞)']


def test_quantiles_empty():
    assert col(DF_EMPTY, pl.col("x").ps_chop.quantiles([0.25, 0.75])).len() == 0


def test_quantiles_all_null():
    s = col(DF_ALL_NULL, pl.col("x").ps_chop.quantiles([0.25, 0.75]))
    assert s.is_null().all()
    assert s.dtype.categories.to_list() == ["[0%, 100%]"]


# ── ±Inf in data ──────────────────────────────────────────────────────────────


DF_INF = pl.DataFrame({"x": [1.0, 3.0, 5.0, float("inf"), float("-inf")]})
DF_POS_INF = pl.DataFrame(
    {"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, float("inf")]}
)


def test_chop_pos_inf_in_last_bin():
    s = col(DF_INF, pl.col("x").ps_chop.chop([3.0]))
    assert s[3] == "[3, ∞)"


def test_chop_neg_inf_in_first_bin():
    s = col(DF_INF, pl.col("x").ps_chop.chop([3.0]))
    assert s[4] == "[-∞, 3)"


def test_chop_nan_becomes_null():
    df = pl.DataFrame({"x": [1.0, float("nan"), 5.0]})
    s = col(df, pl.col("x").ps_chop.chop([3.0]))
    assert s[1] is None


def test_width_pos_inf_in_last_bin():
    s = col(DF_POS_INF, pl.col("x").ps_chop.width(5.0))
    assert s[-1] == "[5, 10]"
    assert s[-2] == "[5, 10]"


def test_width_pos_inf_does_not_create_extra_bin():
    s = col(DF_POS_INF, pl.col("x").ps_chop.width(5.0))
    assert s.dtype.categories.to_list() == ["[0, 5)", "[5, 10]"]


# ── integer columns (discrete closed labels) ──────────────────────────────────


DF_INT = pl.DataFrame({"x": pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=pl.Int32)})


def test_chop_int_labels_extend():
    s = col(DF_INT, pl.col("x").ps_chop.chop([3, 7]))
    assert s.dtype.categories.to_list() == ["(-∞, 2]", "[3, 6]", "[7, +∞)"]


def test_chop_int_labels_no_extend():
    s = col(DF_INT, pl.col("x").ps_chop.chop([3, 7], extend=False))
    assert s.dtype.categories.to_list() == ["[1, 2]", "[3, 6]", "[7, 10]"]


def test_chop_int_single_element_label():
    df = pl.DataFrame({"x": pl.Series([1, 2, 3], dtype=pl.Int32)})
    s = col(df, pl.col("x").ps_chop.chop([2, 3], extend=False))
    assert s.dtype.categories.to_list() == ["{1}", "{2}", "{3}"]


def test_chop_int_struct():
    s = col(DF_INT, pl.col("x").ps_chop.chop([3, 7], extend=False, return_struct=True))
    assert s[0] == {"lo": 1.0, "hi": 2.0}
    assert s[2] == {"lo": 3.0, "hi": 6.0}
    assert s[-1] == {"lo": 7.0, "hi": 10.0}


def test_chop_int_struct_extend():
    s = col(DF_INT, pl.col("x").ps_chop.chop([3, 7], return_struct=True))
    assert math.isinf(s[0]["lo"])
    assert s[0]["hi"] == 2.0
    assert s[2] == {"lo": 3.0, "hi": 6.0}
    assert math.isinf(s[-1]["hi"])


def test_n_elements_int_labels():
    s = col(DF_INT, pl.col("x").ps_chop.n_elements(3))
    assert s.dtype.categories.to_list() == ["[1, 3]", "[4, 6]", "[7, 9]", "{10}"]


def test_width_int_labels():
    s = col(DF_INT, pl.col("x").ps_chop.width(5))
    assert s.dtype.categories.to_list() == ["[1, 5]", "[6, 10]"]


# ── string / enum columns ─────────────────────────────────────────────────────


DF_STR = pl.DataFrame({"x": ["apple", "banana", "cherry", "date", "elderberry", None]})
DF_ENUM = pl.DataFrame({
    "x": pl.Series(["low", "medium", "medium", "high"], dtype=pl.Enum(["low", "medium", "high"]))
})


def test_chop_str_labels():
    s = col(DF_STR, pl.col("x").ps_chop.chop(["banana", "date"]))
    assert s.dtype.categories.to_list() == ["{apple}", "[banana, cherry]", "[date, elderberry]"]
    assert s[0] == "{apple}"
    assert s[1] == "[banana, cherry]"
    assert s[4] == "[date, elderberry]"
    assert s[5] is None


def test_chop_str_struct():
    s = col(DF_STR, pl.col("x").ps_chop.chop(["banana", "date"], return_struct=True))
    assert s[0] == {"lo": "apple", "hi": "apple"}
    assert s[1] == {"lo": "banana", "hi": "cherry"}
    assert s[4] == {"lo": "date", "hi": "elderberry"}
    assert s[5] == {"lo": None, "hi": None}


def test_chop_enum_preserves_order():
    # Enum with non-alphabetical order: [low, medium, high]; break at "medium"
    s = col(DF_ENUM, pl.col("x").ps_chop.chop(["medium"]))
    assert s.dtype.categories.to_list() == ["{low}", "[medium, high]"]
    assert s[0] == "{low}"
    assert s[1] == "[medium, high]"


def test_chop_str_break_not_in_data_raises():
    with pytest.raises(ValueError, match="not found"):
        col(DF_STR, pl.col("x").ps_chop.chop(["zzz"]))


def test_n_elements_str_labels():
    s = col(DF_STR.drop_nulls(), pl.col("x").ps_chop.n_elements(2))
    assert s.dtype.categories.to_list() == [
        "[apple, banana]", "[cherry, date]", "{elderberry}"
    ]


def test_n_elements_str_single_element():
    df = pl.DataFrame({"x": ["a", "b", "c"]})
    s = col(df, pl.col("x").ps_chop.n_elements(1))
    assert s.dtype.categories.to_list() == ["{a}", "{b}", "{c}"]


def test_n_groups_str():
    df = pl.DataFrame({"x": ["a", "b", "c", "c", "c", "c"]})
    s = col(df, pl.col("x").ps_chop.n_groups(2))
    assert len(s.dtype.categories) == 2


def test_quantiles_str():
    df = pl.DataFrame({"x": ["a", "b", "c", "d"]})
    s = col(df, pl.col("x").ps_chop.quantiles([0.5]))
    assert len(s.dtype.categories) >= 1


def test_width_str():
    df = pl.DataFrame({"x": ["a", "b", "c", "d", "e", "f"]})
    s = col(df, pl.col("x").ps_chop.width(2))
    assert s.dtype.categories.to_list() == ["[a, b]", "[c, d]", "[e, f]"]


def test_str_null_passthrough():
    s = col(DF_STR, pl.col("x").ps_chop.chop(["banana"]))
    assert s[5] is None


# ── temporal columns ──────────────────────────────────────────────────────────


D = datetime.date
DT = datetime.datetime
TD = datetime.timedelta

DF_DATE = pl.DataFrame({"d": [D(2020, 1, 1), D(2020, 4, 1), D(2020, 7, 1), D(2020, 10, 1)]})
DF_DT = pl.DataFrame({"d": [DT(2020, 1, 1), DT(2020, 7, 1), DT(2021, 1, 1), DT(2021, 7, 1)]})
DF_DUR = pl.DataFrame({"d": pl.Series(
    [TD(days=1), TD(days=5), TD(days=10), TD(days=15)], dtype=pl.Duration("us")
)})


def col_t(df, expr):
    name = expr.meta.output_name()
    return df.ps.with_columns(expr)[name]


def test_chop_date_labels():
    s = col_t(DF_DATE, pl.col("d").ps_chop.chop([D(2020, 7, 1)]))
    cats = s.dtype.categories.to_list()
    assert len(cats) == 2
    assert cats[0] == "[2020-01-01, 2020-07-01)"


def test_chop_date_struct_lo():
    s = col_t(DF_DATE, pl.col("d").ps_chop.chop([D(2020, 7, 1)], return_struct=True))
    assert s[0]["lo"] == D(2020, 1, 1)
    assert s[2]["lo"] == D(2020, 7, 1)


def test_chop_date_struct_hi():
    s = col_t(DF_DATE, pl.col("d").ps_chop.chop([D(2020, 7, 1)], return_struct=True))
    # hi of first bin is the break boundary; hi of last bin is data max
    assert s[0]["hi"] == D(2020, 7, 1)
    assert s[-1]["hi"] == D(2020, 10, 1)


def test_chop_datetime_labels():
    s = col_t(DF_DT, pl.col("d").ps_chop.chop([DT(2021, 1, 1)]))
    cats = s.dtype.categories.to_list()
    assert len(cats) == 2
    assert cats[0] == '[2020-01-01 00:00:00, 2021-01-01 00:00:00)'


def test_chop_datetime_struct_lo():
    s = col_t(DF_DT, pl.col("d").ps_chop.chop([DT(2021, 1, 1)], return_struct=True))
    assert s[0]["lo"] == DT(2020, 1, 1)
    assert s[2]["lo"] == DT(2021, 1, 1)


def test_chop_datetime_struct_hi():
    s = col_t(DF_DT, pl.col("d").ps_chop.chop([DT(2021, 1, 1)], return_struct=True))
    # hi of first bin is the break; hi of last bin is data max
    assert s[0]["hi"] == DT(2021, 1, 1)
    assert s[-1]["hi"] == DT(2021, 7, 1)


def test_width_date_timedelta():
    s = col_t(DF_DATE, pl.col("d").ps_chop.width(TD(days=90)))
    cats = s.dtype.categories.to_list()
    assert len(cats) >= 2
    assert cats[0] == "[2020-01-01, 2020-03-31)"


def test_width_duration():
    s = col_t(DF_DUR, pl.col("d").ps_chop.width(TD(days=5)))
    cats = s.dtype.categories.to_list()
    assert len(cats) == 3
    assert cats[0] == '[1 day, 0:00:00, 6 days, 0:00:00)'


def test_n_elements_date():
    s = col_t(DF_DATE, pl.col("d").ps_chop.n_elements(2))
    cats = s.dtype.categories.to_list()
    assert len(cats) == 2
    assert s[0] == cats[0]
    assert s[2] == cats[1]


def test_n_groups_date():
    s = col_t(DF_DATE, pl.col("d").ps_chop.n_groups(2))
    cats = s.dtype.categories.to_list()
    assert len(cats) == 2


def test_quantiles_date():
    s = col_t(DF_DATE, pl.col("d").ps_chop.quantiles([0.5]))
    cats = s.dtype.categories.to_list()
    assert len(cats) >= 1


def test_date_struct_dtype():
    s = col_t(DF_DATE, pl.col("d").ps_chop.chop([D(2020, 7, 1)], return_struct=True))
    assert s.dtype.fields[0].dtype == pl.Date
    assert s.dtype.fields[1].dtype == pl.Date


def test_datetime_struct_dtype():
    s = col_t(DF_DT, pl.col("d").ps_chop.chop([DT(2021, 1, 1)], return_struct=True))
    assert isinstance(s.dtype.fields[0].dtype, pl.Datetime)


def test_date_null_passthrough():
    df = pl.DataFrame({"d": [D(2020, 1, 1), None, D(2020, 7, 1)]})
    s = col_t(df, pl.col("d").ps_chop.chop([D(2020, 4, 1)]))
    assert s[1] is None


# ── n_elements right-closed (left_closed=False) ───────────────────────────────


def test_n_elements_right_closed_equal_groups():
    # break placed at last element of first group, not first of next
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    s = col(df, pl.col("x").ps_chop.n_elements(2, left_closed=False))
    assert s.dtype.categories.to_list() == ["[1, 2]", "(2, 4]"]
    assert s[0] == "[1, 2]"
    assert s[1] == "[1, 2]"
    assert s[2] == "(2, 4]"
    assert s[3] == "(2, 4]"


def test_n_elements_right_closed_ties_not_split():
    # ties on the break value spill into current (right-closed) bin
    df = pl.DataFrame({"x": [1.0, 2.0, 2.0, 2.0, 5.0]})
    s = col(df, pl.col("x").ps_chop.n_elements(2, left_closed=False))
    assert s[1] == s[2] == s[3]  # all 2.0s in the same bin


def test_n_elements_right_closed_int():
    df = pl.DataFrame({"x": pl.Series([1, 2, 3, 4], dtype=pl.Int32)})
    s = col(df, pl.col("x").ps_chop.n_elements(2, left_closed=False))
    assert s.dtype.categories.to_list() == ["[1, 2]", "[3, 4]"]
    assert s[0] == "[1, 2]"
    assert s[1] == "[1, 2]"
    assert s[2] == "[3, 4]"
    assert s[3] == "[3, 4]"


def test_n_elements_right_closed_str():
    df = pl.DataFrame({"x": ["a", "b", "c", "d"]})
    s = col(df, pl.col("x").ps_chop.n_elements(2, left_closed=False))
    assert s.dtype.categories.to_list() == ["[a, b]", "[c, d]"]
    assert s[0] == "[a, b]"
    assert s[1] == "[a, b]"
    assert s[2] == "[c, d]"
    assert s[3] == "[c, d]"


def test_n_elements_right_closed_date():
    s = col_t(DF_DATE, pl.col("d").ps_chop.n_elements(2, left_closed=False))
    assert s.dtype.categories.to_list() == [
        "[2020-01-01, 2020-04-01]",
        "(2020-04-01, 2020-10-01]",
    ]
    assert s[0] == "[2020-01-01, 2020-04-01]"
    assert s[1] == "[2020-01-01, 2020-04-01]"
    assert s[2] == "(2020-04-01, 2020-10-01]"
    assert s[3] == "(2020-04-01, 2020-10-01]"


# ── single-value and boundary edge cases ──────────────────────────────────────


def test_chop_single_value_data():
    # one data point: lands in the bin that contains it
    df = pl.DataFrame({"x": [5.0]})
    s = col(df, pl.col("x").ps_chop.chop([3.0]))
    assert s[0] == "[3, ∞)"


def test_chop_single_value_no_extend():
    df = pl.DataFrame({"x": [5.0]})
    s = col(df, pl.col("x").ps_chop.chop([3.0], extend=False))
    assert s[0] == "[3, 5]"


def test_n_elements_n_larger_than_data():
    # n > len(data) with tail="merge": everything goes in one bin
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    s = col(df, pl.col("x").ps_chop.n_elements(10, tail="merge"))
    assert s.n_unique() == 1


def test_n_elements_n_equals_data_length():
    # n == len(data): all values in one bin
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    s = col(df, pl.col("x").ps_chop.n_elements(3))
    assert s.n_unique() == 1


def test_n_elements_all_same_values():
    # all identical: can't split, should produce a single bin
    df = pl.DataFrame({"x": [7.0, 7.0, 7.0, 7.0]})
    s = col(df, pl.col("x").ps_chop.n_elements(2))
    assert s.n_unique() == 1