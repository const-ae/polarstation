"""Tests for FrameExpr.over() — grouped resolution of ps_enum/ps_chop methods.

.over() uses a single mechanism (frame_expr.py's _generic_partition_resolver) for
every method and every composition of them: partition the frame by the .over()
columns, rerun the (unmodified) resolver per partition, concatenate the results back
together. An earlier version of this dispatched to per-method "fast path"
implementations for direct (non-chained) calls, avoiding the eager collect below —
that was reverted after (a) finding a genuine correctness divergence between the fast
path and the ungrouped semantics for an Enum column with an unobserved category, and
(b) benchmarking showing no reliable speed advantage over this single mechanism (some
fast paths were measurably slower). These tests exercise correctness of that one
mechanism: it must match a manual partition_by + per-group call + concat, for both
single methods and composed chains.
"""
from unittest.mock import patch

import polars as pl
import pytest

import polarstation  # noqa: F401 — registers ps_enum/ps_chop namespaces


def _manual_reference(df: pl.DataFrame, by: list[str], build_expr, target_col: str = "v") -> list:
    """Reference implementation: partition by `by`, apply build_expr() per partition,
    concat back, restore order — the same thing FrameExpr.over() does internally, but
    hand-written independently of the implementation under test.
    """
    idx_df = df.with_row_index("__idx__")
    parts = idx_df.partition_by(by, maintain_order=True)
    resolved = []
    for part in parts:
        r = part.ps.with_columns(build_expr())
        resolved.append(r.select("__idx__", pl.col(target_col).cast(pl.String).alias(target_col)))
    combined = pl.concat(resolved, how="vertical_relaxed").sort("__idx__").drop("__idx__")
    return combined[target_col].to_list()


# ── reference-implementation comparisons ───────────────────────────────────────


def test_make_over_matches_manual_partition():
    df = pl.DataFrame({"g": ["x", "x", "x", "y", "y"], "v": ["a", "b", "a", "b", "c"]})
    grouped = df.ps.with_columns(pl.col("v").ps_enum.make().over("g"))
    expected = _manual_reference(df, ["g"], lambda: pl.col("v").ps_enum.make())
    assert grouped["v"].cast(pl.String).to_list() == expected
    assert grouped["v"].dtype == pl.Enum(["a", "b", "c"])


def test_make_over_enum_input_with_unobserved_category_matches_ungrouped():
    # Regression test: an earlier "fast path" (native rank().over()) disagreed with the
    # ungrouped semantics here, since an already-Enum column's category list is a
    # schema property (dtype.categories), not something to be rediscovered per group —
    # group "y" never observes "b", but "c" must still land at its declared position.
    df = pl.DataFrame({
        "g": ["x", "x", "x", "y", "y"],
        "v": pl.Series(["a", "b", "c", "a", "c"], dtype=pl.Enum(["a", "b", "c"])),
    })
    grouped = df.ps.with_columns(pl.col("v").ps_enum.to_level().over("g"))
    expected = _manual_reference(df, ["g"], lambda: pl.col("v").ps_enum.to_level())
    assert grouped["v"].cast(pl.String).to_list() == expected
    assert grouped["v"].to_list() == [0, 1, 2, 0, 2]


def test_lump_over_matches_manual_partition():
    df = pl.DataFrame({
        "g": ["x"] * 6 + ["y"] * 6,
        "v": ["a", "a", "a", "b", "b", "c"] + ["p", "p", "q", "q", "q", "r"],
    })
    grouped = df.ps.with_columns(pl.col("v").ps_enum.lump(n=1).over("g"))
    expected = _manual_reference(df, ["g"], lambda: pl.col("v").ps_enum.lump(n=1))
    assert grouped["v"].cast(pl.String).to_list() == expected


def test_lump_over_null_target_stays_null():
    df = pl.DataFrame({"g": ["x", "x", "y", "y"], "v": ["a", None, "b", None]})
    grouped = df.ps.with_columns(pl.col("v").ps_enum.lump(n=1).over("g"))
    assert grouped["v"].to_list()[1] is None
    assert grouped["v"].to_list()[3] is None


def test_reorder_over_matches_manual_partition():
    # Compares values only (cast to String): the Enum's declared category *order* is a
    # first-appearance union across groups by design, not any single group's own order
    # — chaining .to_level() (below) is how a caller recovers genuine per-group order.
    df = pl.DataFrame({
        "g": ["x", "x", "x", "y", "y"],
        "v": ["a", "b", "a", "b", "c"],
        "w": [1.0, 10.0, 1.0, 1.0, 10.0],
    })
    grouped = df.ps.with_columns(pl.col("v").ps_enum.reorder("w").over("g"))
    expected = _manual_reference(df, ["g"], lambda: pl.col("v").ps_enum.reorder("w"))
    assert grouped["v"].cast(pl.String).to_list() == expected


def test_reorder_to_level_over_matches_manual_partition():
    df = pl.DataFrame({
        "g": ["x", "x", "x", "y", "y"],
        "v": ["a", "b", "a", "b", "c"],
        "w": [1.0, 10.0, 1.0, 1.0, 10.0],
    })
    grouped = df.ps.with_columns(
        pl.col("v").ps_enum.make().ps_enum.reorder("w").ps_enum.to_level().over("g")
    )
    expected_df = pl.concat(
        [
            part.ps.with_columns(
                pl.col("v").ps_enum.make().ps_enum.reorder("w").ps_enum.to_level()
            )
            for part in df.with_row_index("__idx__").partition_by(["g"], maintain_order=True)
        ],
        how="vertical_relaxed",
    ).sort("__idx__")
    assert grouped["v"].to_list() == expected_df["v"].to_list()


def test_infreq_descending_true_over_matches_manual_partition():
    df = pl.DataFrame({"g": ["x", "x", "x", "y", "y"], "v": ["a", "b", "a", "b", "c"]})
    grouped = df.ps.with_columns(pl.col("v").ps_enum.infreq(descending=True).over("g"))
    expected = _manual_reference(df, ["g"], lambda: pl.col("v").ps_enum.infreq(descending=True))
    assert grouped["v"].cast(pl.String).to_list() == expected


@pytest.mark.parametrize("raw", [False, True])
def test_quantiles_over_matches_manual_partition(raw):
    df = pl.DataFrame({
        "g": ["x"] * 5 + ["y"] * 5,
        "v": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0, 300.0, 400.0, 500.0],
    })
    grouped = df.ps.with_columns(pl.col("v").ps_chop.quantiles([0.5], raw=raw).over("g"))
    expected = _manual_reference(df, ["g"], lambda: pl.col("v").ps_chop.quantiles([0.5], raw=raw))
    assert grouped["v"].cast(pl.String).to_list() == expected


@pytest.mark.parametrize("raw", [False, True])
def test_quantiles_over_tie_dedup_matches_manual_partition(raw):
    # Group "x" is low-cardinality enough that some of the 3 probs collapse to the same
    # quantile value.
    df = pl.DataFrame({
        "g": ["x"] * 4 + ["y"] * 10,
        "v": [1.0, 1.0, 1.0, 1.0] + [float(i) for i in range(1, 11)],
    })
    probs = [0.25, 0.5, 0.75]
    grouped = df.ps.with_columns(pl.col("v").ps_chop.quantiles(probs, raw=raw).over("g"))
    expected = _manual_reference(df, ["g"], lambda: pl.col("v").ps_chop.quantiles(probs, raw=raw))
    assert grouped["v"].cast(pl.String).to_list() == expected


def test_quantiles_over_integer_dtype_with_extend():
    df = pl.DataFrame({"g": ["x"] * 4 + ["y"] * 4, "v": [1, 2, 3, 4, 10, 20, 30, 40]})
    grouped = df.ps.with_columns(
        pl.col("v").ps_chop.quantiles([0.5], raw=True, extend=True).over("g")
    )
    expected = _manual_reference(
        df, ["g"], lambda: pl.col("v").ps_chop.quantiles([0.5], raw=True, extend=True)
    )
    assert grouped["v"].cast(pl.String).to_list() == expected


def test_quantiles_over_null_target_stays_null():
    df = pl.DataFrame({"g": ["x", "x", "x", "y", "y"], "v": [1.0, None, 3.0, 10.0, 20.0]})
    grouped = df.ps.with_columns(pl.col("v").ps_chop.quantiles([0.5]).over("g"))
    assert grouped["v"].to_list()[1] is None


def test_quantiles_over_categorical_dtype_works():
    df = pl.DataFrame({"g": ["x", "x", "x", "y", "y"], "v": ["a", "b", "c", "a", "b"]})
    grouped = df.ps.with_columns(pl.col("v").ps_enum.make().ps_chop.quantiles([0.5]).over("g"))
    expected = _manual_reference(
        df, ["g"], lambda: pl.col("v").ps_enum.make().ps_chop.quantiles([0.5])
    )
    assert grouped["v"].cast(pl.String).to_list() == expected


def test_quantiles_over_return_struct_works():
    df = pl.DataFrame({"g": ["x", "x", "x", "y", "y"], "v": [1.0, 2.0, 3.0, 10.0, 20.0]})
    grouped = df.ps.with_columns(
        pl.col("v").ps_chop.quantiles([0.5], return_struct=True).over("g")
    )
    assert len(grouped) == len(df)


def test_quantiles_over_wrong_label_count_raises_like_ungrouped():
    # len(labels) not matching len(probs) + 1 is a genuine user error — it should raise
    # the same way the ungrouped call does, not silently produce a wrong result.
    df = pl.DataFrame({"g": ["x", "x", "y", "y"], "v": [1.0, 2.0, 10.0, 20.0]})
    with pytest.raises(pl.exceptions.ShapeError):
        df.ps.with_columns(
            pl.col("v").ps_chop.quantiles([0.25, 0.5, 0.75], labels=["a", "b"]).over("g")
        )


# ── composed chains ─────────────────────────────────────────────────────────────


def test_composed_chain_infreq_to_level_over():
    df = pl.DataFrame({
        "g": ["x"] * 6 + ["y"] * 6,
        "v": ["a", "a", "a", "b", "b", "c"] + ["p", "q", "q", "q", "r", "r"],
    })
    grouped = df.ps.with_columns(
        level=pl.col("v").ps_enum.make().ps_enum.infreq().ps_enum.to_level().over("g")
    )
    # Within each group, infreq() orders most-frequent-first, so level 0 == the mode.
    x_levels = grouped.filter(pl.col("g") == "x")["level"].to_list()
    y_levels = grouped.filter(pl.col("g") == "y")["level"].to_list()
    assert x_levels == [0, 0, 0, 1, 1, 2]  # "a" (x3) -> 0, "b" (x2) -> 1, "c" (x1) -> 2
    assert y_levels == [2, 0, 0, 0, 1, 1]  # "q" (x3) -> 0, "r" (x2) -> 1, "p" (x1) -> 2


def test_composed_chain_lt_comparison_over():
    df = pl.DataFrame({
        "g": ["x"] * 6 + ["y"] * 6,
        "v": ["a", "a", "a", "b", "b", "c"] + ["p", "q", "q", "q", "r", "r"],
    })
    grouped = df.ps.with_columns(
        keep=(pl.col("v").ps_enum.make().ps_enum.infreq().ps_enum.to_level().lt(1)).over("g")
    )
    assert grouped["keep"].to_list() == [
        True, True, True, False, False, False,
        False, True, True, True, False, False,
    ]


# ── dunder forwarding (bare operators on FrameExpr) ────────────────────────────


def test_bare_lt_operator_matches_lt_method():
    df = pl.DataFrame({
        "g": ["x"] * 6 + ["y"] * 6,
        "v": ["a", "a", "a", "b", "b", "c"] + ["p", "q", "q", "q", "r", "r"],
    })
    via_method = df.ps.with_columns(
        keep=pl.col("v").ps_enum.make().ps_enum.infreq().ps_enum.to_level().lt(1).over("g")
    )["keep"].to_list()
    via_operator = df.ps.with_columns(
        keep=(pl.col("v").ps_enum.make().ps_enum.infreq().ps_enum.to_level() < 1).over("g")
    )["keep"].to_list()
    assert via_operator == via_method


@pytest.mark.parametrize(
    "op",
    [
        lambda e: e < 1,
        lambda e: e <= 1,
        lambda e: e > 1,
        lambda e: e >= 1,
        lambda e: e == 1,
        lambda e: e != 1,
        lambda e: e + 1,
    ],
)
def test_bare_operators_compose_with_over(op):
    df = pl.DataFrame({"g": ["x", "x", "y", "y"], "v": ["a", "b", "a", "b"]})
    level = pl.col("v").ps_enum.to_level()
    result = df.ps.with_columns(out=op(level).over("g"))
    assert len(result) == 4


# ── multi-column by / computed grouping key ────────────────────────────────────


def test_over_multi_column_by_string_args_and_list_agree():
    df = pl.DataFrame({
        "g1": ["x", "x", "y", "y"],
        "g2": [1, 2, 1, 2],
        "v": ["a", "b", "a", "b"],
    })
    a = df.ps.with_columns(pl.col("v").ps_enum.make().over("g1", "g2"))
    b = df.ps.with_columns(pl.col("v").ps_enum.make().over(["g1", "g2"]))
    assert a["v"].to_list() == b["v"].to_list()
    assert a["v"].dtype == b["v"].dtype


def test_over_computed_grouping_key():
    df = pl.DataFrame({"n": [1, 2, 11, 12], "v": ["a", "b", "a", "c"]})
    grouped = df.ps.with_columns(pl.col("v").ps_enum.make().over(pl.col("n") // 10))
    # group 0: n in {1,2} -> v in {a,b}; group 1: n in {11,12} -> v in {a,c}
    assert grouped["v"].dtype == pl.Enum(["a", "b", "c"])
    assert grouped["v"].to_list() == ["a", "b", "a", "c"]


# ── null handling ────────────────────────────────────────────────────────────


def test_over_null_group_key_forms_its_own_partition():
    df = pl.DataFrame({"g": ["x", None, "x", None], "v": ["a", "b", "a", "c"]})
    grouped = df.ps.with_columns(pl.col("v").ps_enum.make().over("g"))
    assert grouped["v"].dtype == pl.Enum(["a", "b", "c"])
    assert grouped["v"].to_list() == ["a", "b", "a", "c"]


def test_over_null_in_target_column_stays_null():
    df = pl.DataFrame({"g": ["x", "x", "y", "y"], "v": ["a", None, "b", None]})
    grouped = df.ps.with_columns(pl.col("v").ps_enum.make().over("g"))
    assert grouped["v"].to_list() == ["a", None, "b", None]


# ── empty / all-null frames ─────────────────────────────────────────────────────


def test_over_empty_frame():
    df = pl.DataFrame({"g": pl.Series([], dtype=pl.String), "v": pl.Series([], dtype=pl.String)})
    grouped = df.ps.with_columns(pl.col("v").ps_enum.make().over("g"))
    assert len(grouped) == 0
    assert grouped["v"].dtype == pl.Enum([])


def test_over_all_null_target():
    df = pl.DataFrame({"g": ["x", "x", "y"], "v": pl.Series([None, None, None], dtype=pl.String)})
    grouped = df.ps.with_columns(pl.col("v").ps_enum.make().over("g"))
    assert grouped["v"].to_list() == [None, None, None]


# ── column-name collision avoidance ──────────────────────────────────────────


def test_over_handles_existing_temp_name_collision():
    df = pl.DataFrame({
        "g": ["x", "x", "y"],
        "v": ["a", "b", "a"],
        "__ps_over_idx__": [9, 9, 9],
        "__ps_over_by_0__": [9, 9, 9],
    })
    grouped = df.ps.with_columns(pl.col("v").ps_enum.make().over("g"))
    assert grouped["v"].dtype == pl.Enum(["a", "b"])
    assert grouped["v"].to_list() == ["a", "b", "a"]


# ── .over() is eager: collects the full frame, unlike ungrouped resolvers ──────


class _CollectTracker:
    def __init__(self):
        self.row_counts: list[int] = []

    def __enter__(self):
        original = pl.LazyFrame.collect
        tracker = self

        def patched(self_lf, *args, **kwargs):
            result = original(self_lf, *args, **kwargs)
            tracker.row_counts.append(result.height)
            return result

        self._patcher = patch.object(pl.LazyFrame, "collect", patched)
        self._patcher.start()
        return self

    def __exit__(self, *args):
        self._patcher.stop()

    @property
    def max_rows(self) -> int:
        return max(self.row_counts) if self.row_counts else 0


def test_over_always_collects_full_frame_documented_tradeoff():
    """Every .over() call — direct or chained — uses the same eager partition
    mechanism: a following .filter() cannot help, because .over() already collected
    everything by the time with_columns() returns. This is documented, expected
    behavior (see FrameExpr.over()'s docstring), not a bug to "optimize away" without
    reintroducing the per-method fast-path risk this design deliberately avoids.
    """
    df = pl.DataFrame({"g": ["x"] * 50 + ["y"] * 50, "v": (["a", "b"] * 50)})
    with _CollectTracker() as t:
        df.ps.with_columns(pl.col("v").ps_enum.make().over("g"))
    assert t.max_rows == 100, f"expected .over() to collect all 100 rows, got {t.row_counts}"


def test_preceding_filter_shrinks_collect():
    df = pl.DataFrame({"g": ["x"] * 50 + ["y"] * 50, "v": (["a", "b"] * 50)})
    with _CollectTracker() as t:
        df.filter(pl.col("g") == "x").ps.with_columns(pl.col("v").ps_enum.make().over("g"))
    assert t.max_rows == 50, f"a preceding filter should shrink the collect, got {t.row_counts}"


def test_following_filter_does_not_shrink_collect():
    """A .filter() applied AFTER ps.with_columns(...).over(...) cannot retroactively
    narrow the eager collect that already happened — documented limitation, not a bug.
    """
    df = pl.DataFrame({"g": ["x"] * 50 + ["y"] * 50, "v": (["a", "b"] * 50)})
    with _CollectTracker() as t:
        df.ps.with_columns(pl.col("v").ps_enum.make().over("g")).filter(pl.col("g") == "x")
    assert t.max_rows == 100, f"expected no benefit from the following filter, got {t.row_counts}"
