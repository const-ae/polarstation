"""
Verify that resolvers never collect the full dataset.
Each resolver should only collect small frames (≤ K rows) for group counts,
unique values, or aggregations. Schema-only resolvers should collect nothing.
"""
from unittest.mock import patch

import polars as pl
import pytest

import polarstation  # noqa: F401

N = 100_000  # rows — large enough that an accidental full scan is obvious
K = 10  # unique category values


@pytest.fixture
def large_lf():
    return (
        pl.select(pl.int_range(0, N, eager=True).alias("_id"))
        .lazy()
        .with_columns(
            (pl.col("_id") % K).cast(pl.String).alias("group"),
            pl.col("_id").cast(pl.Float64).alias("value"),
        )
    )


@pytest.fixture
def large_enum_lf(large_lf):
    # Calls make() once outside any tracker — establishes the Enum type lazily.
    return large_lf.ps.with_columns(pl.col("group").ps_enum.make())


class CollectTracker:
    """Patches LazyFrame.collect to record the height of every collected frame."""

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


# ── operations that do one small targeted collect ─────────────────────────────


def test_make_collects_only_unique_values(large_lf):
    with CollectTracker() as t:
        large_lf.ps.with_columns(pl.col("group").ps_enum.make())
    assert t.max_rows <= K, f"make() collected {t.max_rows} rows, expected ≤ {K}"


def test_lump_collects_only_group_counts(large_enum_lf):
    with CollectTracker() as t:
        large_enum_lf.ps.with_columns(pl.col("group").ps_enum.lump(n=5))
    assert t.max_rows <= K, f"lump() collected {t.max_rows} rows, expected ≤ {K}"


def test_reorder_collects_only_aggregations(large_enum_lf):
    with CollectTracker() as t:
        large_enum_lf.ps.with_columns(pl.col("group").ps_enum.reorder("value"))
    assert t.max_rows <= K, f"reorder() collected {t.max_rows} rows, expected ≤ {K}"


def test_drop_unused_collects_only_unique_values(large_enum_lf):
    with CollectTracker() as t:
        large_enum_lf.ps.with_columns(pl.col("group").ps_enum.drop_unused())
    assert t.max_rows <= K, f"drop_unused() collected {t.max_rows} rows, expected ≤ {K}"


# ── schema-only operations: zero collects ────────────────────────────────────


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(
            lambda lf, cats: lf.ps.with_columns(pl.col("group").ps_enum.rev()),
            id="rev",
        ),
        pytest.param(
            lambda lf, cats: lf.ps.with_columns(
                pl.col("group").ps_enum.relabel({"0": "zero"})
            ),
            id="relabel",
        ),
        pytest.param(
            lambda lf, cats: lf.ps.with_columns(
                pl.col("group").ps_enum.add_categories(["new"])
            ),
            id="add_categories",
        ),
        pytest.param(
            lambda lf, cats: lf.ps.with_columns(
                pl.col("group").ps_enum.set_categories(cats + ["extra"])
            ),
            id="set_categories",
        ),
        pytest.param(
            lambda lf, cats: lf.ps.with_columns(
                pl.col("group").ps_enum.missing_to_category("NA")
            ),
            id="missing_to_category",
        ),
        pytest.param(
            lambda lf, cats: lf.ps.with_columns(
                pl.col("group").ps_enum.category_to_missing("0")
            ),
            id="category_to_missing",
        ),
    ],
)
def test_schema_only_ops_never_collect(op, large_enum_lf):
    cats = [str(i) for i in range(K)]
    with CollectTracker() as t:
        op(large_enum_lf, cats)
    assert t.row_counts == [], f"expected zero collects, got: {t.row_counts}"
