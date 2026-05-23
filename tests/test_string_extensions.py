import polars as pl
import polars.testing as plt

import polarstation as ps  # noqa: F401 — registers ps_str namespace


def test_count():
    df = pl.DataFrame({"s": ["hello", "world", "foo"]})
    result = df.select(pl.col("s").ps_str.count("[aeiou]"))
    plt.assert_series_equal(result["s"], pl.Series("s", [2, 1, 2], dtype=pl.UInt32))


def test_wrap():
    df = pl.DataFrame({"s": ["one two three four five"]})
    result = df.select(pl.col("s").ps_str.wrap(width=10))["s"][0]
    assert all(len(line) <= 10 for line in result.splitlines())

def test_trunc():
    df = pl.DataFrame({"s": ["a long sentence that should be truncated"]})
    result = df.select(pl.col("s").ps_str.trunc(width=3))["s"][0]
    assert result == "a …"

    result2 = df.select(pl.col("s").ps_str.trunc(width=3, placeholder="..."))["s"][0]
    assert result2 == "..."
    
    result3 = df.select(pl.col("s").ps_str.trunc(width=50))["s"][0]
    assert result3 == "a long sentence that should be truncated"

    result4 = df.select(pl.col("s").ps_str.trunc(width=3, side="left"))["s"][0]
    assert result4 == "…ed"

    result5 = df.select(pl.col("s").ps_str.trunc(width=3, side="center"))["s"][0]
    assert result5 == "a…d"