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
    s = col(DF10, pl.col("x").ps_chop.chop([3.0, 7.0], include_breaks=True))
    assert s.dtype == pl.Struct({"lo": pl.Float64, "hi": pl.Float64})
    rows = s.to_list()
    assert rows[0] == {"lo": float("-inf"), "hi": 3.0}
    assert rows[2] == {"lo": 3.0, "hi": 7.0}
    assert rows[-1] == {"lo": 7.0, "hi": float("inf")}


def test_chop_include_breaks_no_extend():
    # extend=False: outer bounds are data min/max
    s = col(DF10, pl.col("x").ps_chop.chop([3.0, 7.0], extend=False, include_breaks=True))
    assert s.to_list()[0] == {"lo": 1.0, "hi": 3.0}
    assert s.to_list()[-1] == {"lo": 7.0, "hi": 10.0}


def test_chop_include_breaks_null():
    s = col(DF_NULL, pl.col("x").ps_chop.chop([2.0], include_breaks=True))
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
    s = col(DF10, pl.col("x").ps_chop.width(3.0, include_breaks=True))
    assert s.dtype == pl.Struct({"lo": pl.Float64, "hi": pl.Float64})
    assert s[0] == {"lo": 1.0, "hi": 4.0}
    assert s[3] == {"lo": 4.0, "hi": 7.0}
    assert s[-1] == {"lo": 7.0, "hi": 10.0}


def test_width_include_breaks_extend():
    s = col(DF10, pl.col("x").ps_chop.width(3.0, extend=True, include_breaks=True))
    assert math.isinf(s[0]["lo"])
    assert math.isinf(s[-1]["hi"])


# ── n_elements ────────────────────────────────────────────────────────────────


def test_n_elements_labels():
    s = col(DF10, pl.col("x").ps_chop.n_elements(4))
    # [1..4) → {1,2,3,4}? No: 4 elements = {1,2,3,4} first group, then break at xs[4]=5
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
    s = col(DF10, pl.col("x").ps_chop.n_elements(4, include_breaks=True))
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
    s = col(DF10, pl.col("x").ps_chop.n_groups(3, include_breaks=True))
    assert s.dtype == pl.Struct({"lo": pl.Float64, "hi": pl.Float64})
    assert s[0] == {"lo": 1.0, "hi": 4.0}
    assert s[-1] == {"lo": 7.0, "hi": 10.0}


def test_n_groups_include_breaks_extend():
    s = col(DF10, pl.col("x").ps_chop.n_groups(3, include_breaks=True, extend=True))
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
    s = col(DF10, pl.col("x").ps_chop.quantiles([0.25, 0.5, 0.75], include_breaks=True))
    assert s.dtype == pl.Struct({"lo": pl.Float64, "hi": pl.Float64})
    assert s[0] == {"lo": 1.0, "hi": 3.25}
    assert s[-1] == {"lo": 7.75, "hi": 10.0}


def test_quantiles_include_breaks_raw_extend():
    s = col(DF10, pl.col("x").ps_chop.quantiles([0.5], raw=True, extend=True, include_breaks=True))
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
    # +Inf maps to the last bin
    s = col(DF_INF, pl.col("x").ps_chop.chop([3.0]))
    assert s[3] == "[3, ∞)"


def test_chop_neg_inf_in_first_bin():
    # -Inf maps to the first bin
    s = col(DF_INF, pl.col("x").ps_chop.chop([3.0]))
    assert s[4] == "[-∞, 3)"


def test_chop_nan_becomes_null():
    df = pl.DataFrame({"x": [1.0, float("nan"), 5.0]})
    s = col(df, pl.col("x").ps_chop.chop([3.0]))
    assert s[1] is None


def test_width_pos_inf_in_last_bin():
    # +Inf maps to the last finite bin; bin structure is based on finite data only
    s = col(DF_POS_INF, pl.col("x").ps_chop.width(5.0))
    assert s[-1] == "[5, 10]"   # inf lands in last bin, labeled by finite bounds
    assert s[-2] == "[5, 10]"   # 10.0 is also in last bin


def test_width_pos_inf_does_not_create_extra_bin():
    # Bin structure comes from finite data [0..10]; +Inf should not add an extra bin
    s = col(DF_POS_INF, pl.col("x").ps_chop.width(5.0))
    assert s.dtype.categories.to_list() == ["[0, 5)", "[5, 10]"]
