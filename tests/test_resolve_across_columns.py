"""Tests for resolve_across_columns and ps.apply.

Edge cases covered
------------------
- real column reference
- multi-column selector (pl.col(dtype))
- transform with same output name (str.to_uppercase)
- when/then/otherwise with synthetic "literal" output name
- when/then/otherwise with real column name but transformed values
- multi-column transform (each column sees its own transformed values)
- fn can peek lf for aggregation through col_ref
- fn retrieves column name via col_ref.meta.output_name()
- chained ps.apply: second fn sees first fn's output values
- LazyFrame: ps.apply returns LazyFrame and collects correctly
- empty DataFrame: ps.apply returns empty result without error
- keyword (named) alias in ps.with_columns
- filter context: preceding filter is visible to fn via lf
- CSE: map_elements inside expr is called multiple times (Polars does not CSE Python UDFs)
"""

import math

import polars as pl
import pytest

import polarstation  # noqa: F401 — registers namespaces
from polarstation.frame_expr import resolve_across_columns

# ── resolve_across_columns directly ───────────────────────────────────────────


def _enum_fn(*, lf, name, col_ref, dtype):
  """Collect unique string values sorted, cast col_ref to Enum."""
  cats = lf.select(col_ref.drop_nulls().unique().sort()).collect()[name].to_list()
  return col_ref.cast(pl.Enum(cats)).alias(name)


def test_rac_real_column():
  df = pl.DataFrame({"x": ["a", "b", "b", "c"]})
  exprs = resolve_across_columns(pl.col("x"), _enum_fn)(df.lazy())
  r = df.with_columns(exprs)
  assert r["x"].dtype == pl.Enum(["a", "b", "c"])


def test_rac_multi_col():
  df = pl.DataFrame({"x": ["a", "b"], "y": ["p", "q"]})
  exprs = resolve_across_columns(pl.col(pl.String), _enum_fn)(df.lazy())
  r = df.with_columns(exprs)
  assert r["x"].dtype == pl.Enum(["a", "b"])
  assert r["y"].dtype == pl.Enum(["p", "q"])


def test_rac_transform_same_name():
  """col_ref must see the transformed (uppercased) values, not the originals."""
  df = pl.DataFrame({"x": ["a", "b", "c"]})
  expr = pl.col("x").str.to_uppercase()
  seen: list[str] = []

  def fn(*, lf, name, col_ref, dtype):
    vals = lf.select(col_ref.drop_nulls().unique().sort()).collect()[name].to_list()
    seen.extend(vals)
    return col_ref.cast(pl.Enum(vals)).alias(name)

  r = df.with_columns(resolve_across_columns(expr, fn)(df.lazy()))
  assert seen == ["A", "B", "C"]
  assert r["x"].dtype == pl.Enum(["A", "B", "C"])


def test_rac_when_then_synthetic_name():
  """when/then with pl.lit produces synthetic name 'literal'; col_ref still resolves."""
  df = pl.DataFrame({"a": ["x", "y", "z"]})
  expr = pl.when(pl.col("a") == "x").then(pl.lit("X")).otherwise(pl.lit("other"))

  schema = df.lazy().select(expr).collect_schema()
  assert list(schema.names()) == ["literal"]

  seen: list[str] = []

  def fn(*, lf, name, col_ref, dtype):
    vals = lf.select(col_ref.drop_nulls().unique().sort()).collect()[name].to_list()
    seen.extend(vals)
    return col_ref.cast(pl.Enum(vals)).alias(name)

  r = df.with_columns(resolve_across_columns(expr, fn)(df.lazy()))
  assert seen == ["X", "other"]
  assert r["literal"].dtype == pl.Enum(["X", "other"])


def test_rac_when_then_real_name_sees_transformed_values():
  """when/then with otherwise(col) produces transformed values.
  col_ref must see the post-when/then values, not the original column values.
  In current Polars the output name of this expression is 'literal'.
  """
  df = pl.DataFrame({"a": ["x", "y", "z"]})
  expr = pl.when(pl.col("a") == "x").then(pl.lit("X")).otherwise(pl.col("a"))

  schema = df.lazy().select(expr).collect_schema()
  # Polars names this "literal" (output of both branches is a literal type)
  assert len(schema.names()) == 1

  seen: list[str] = []

  def fn(*, lf, name, col_ref, dtype):
    vals = sorted(lf.select(col_ref.drop_nulls().unique()).collect()[name].to_list())
    seen.extend(vals)
    return col_ref.alias(name)

  df.with_columns(resolve_across_columns(expr, fn)(df.lazy()))
  assert "X" in seen
  assert "x" not in seen  # original value must NOT appear


def test_rac_kwargs_forwarded():
  """Extra kwargs passed to resolve_across_columns are forwarded to fn."""

  def fn(*, lf, name, col_ref, dtype, prefix):
    cats = lf.select(col_ref.drop_nulls().unique().sort()).collect()[name].to_list()
    prefixed = [prefix + c for c in cats]
    return (
      col_ref.replace_strict(cats, prefixed, return_dtype=pl.String)
      .cast(pl.Enum(prefixed))
      .alias(name)
    )

  df = pl.DataFrame({"x": ["a", "b"]})
  exprs = resolve_across_columns(pl.col("x"), fn, prefix="p_")(df.lazy())
  r = df.with_columns(exprs)
  assert r["x"].dtype.categories.to_list() == ["p_a", "p_b"]


def test_rac_dtype_passed():
  """fn receives the correct dtype for the column."""
  dtypes_seen: dict[str, pl.DataType] = {}

  def fn(*, lf, name, col_ref, dtype):
    dtypes_seen[name] = dtype
    return col_ref.alias(name)

  df = pl.DataFrame({"x": ["a"], "n": [1]})
  resolve_across_columns(pl.col("x", "n"), fn)(df.lazy())
  assert dtypes_seen["x"] == pl.String
  assert dtypes_seen["n"] == pl.Int64


# ── ps.apply — basic correctness ──────────────────────────────────────────────


def test_apply_basic_centering():
  def center(lf: pl.LazyFrame, col_ref: pl.Expr) -> pl.Expr:
    m = lf.select(col_ref.mean()).collect().item()
    return col_ref - m

  df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
  r = df.ps.with_columns(pl.col("x").ps.apply(center))
  assert r["x"].to_list() == pytest.approx([-1.0, 0.0, 1.0])
  assert r["x"].name == "x"


def test_apply_output_col_name_preserved():
  """Output column name matches the expression's output name."""

  def double(lf, col_ref):
    return col_ref * 2

  df = pl.DataFrame({"score": [1.0, 2.0]})
  r = df.ps.with_columns(pl.col("score").ps.apply(double))
  assert "score" in r.columns
  assert r["score"].to_list() == pytest.approx([2.0, 4.0])


def test_apply_preserves_nulls():
  def identity(lf, col_ref):
    return col_ref

  df = pl.DataFrame({"x": [1.0, None, 3.0]})
  r = df.ps.with_columns(pl.col("x").ps.apply(identity))
  assert r["x"].null_count() == 1
  assert r["x"][1] is None


# ── ps.apply — multi-column selector ──────────────────────────────────────────


def test_apply_multi_col_calls_fn_per_column():
  """fn is invoked once per column in the selector."""
  called: list[str] = []

  def record(lf, col_ref):
    called.append(col_ref.meta.output_name())
    return col_ref

  df = pl.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]})
  df.ps.with_columns(pl.col(pl.Float64).ps.apply(record))
  assert sorted(called) == ["x", "y", "z"]


def test_apply_multi_col_independent_aggregations():
  """Each column's fn peeks only its own values."""

  def make_enum(lf, col_ref):
    name = col_ref.meta.output_name()
    cats = lf.select(col_ref.drop_nulls().unique().sort()).collect()[name].to_list()
    return col_ref.cast(pl.Enum(cats))

  df = pl.DataFrame({"x": ["a", "b"], "y": ["p", "q", "r"][:2]})
  r = df.ps.with_columns(pl.col(pl.String).ps.apply(make_enum))
  assert r["x"].dtype == pl.Enum(["a", "b"])
  assert r["y"].dtype == pl.Enum(["p", "q"])


# ── ps.apply — transformed values ─────────────────────────────────────────────


def test_apply_transform_values_visible():
  """col_ref reflects transformed (uppercased) values, not the original column."""
  seen: list[str] = []

  def capture(lf, col_ref):
    name = col_ref.meta.output_name()
    vals = lf.select(col_ref.drop_nulls().unique().sort()).collect()[name].to_list()
    seen.extend(vals)
    return col_ref

  df = pl.DataFrame({"x": ["a", "b", "c"]})
  df.ps.with_columns(pl.col("x").str.to_uppercase().ps.apply(capture))
  assert seen == ["A", "B", "C"]


def test_apply_when_then_synthetic_name():
  """ps.apply works when expression output name is synthetic ('literal')."""

  def make_enum(lf, col_ref):
    name = col_ref.meta.output_name()
    cats = lf.select(col_ref.drop_nulls().unique().sort()).collect()[name].to_list()
    return col_ref.cast(pl.Enum(cats))

  df = pl.DataFrame({"a": ["x", "y", "z"]})
  expr = pl.when(pl.col("a") == "x").then(pl.lit("X")).otherwise(pl.lit("other"))
  r = df.ps.with_columns(expr.ps.apply(make_enum))
  assert r["literal"].dtype == pl.Enum(["X", "other"])


def test_apply_when_then_real_name_transformed():
  """col_ref sees when/then-transformed values even when output name = real column name."""
  seen: list[str] = []

  def capture(lf, col_ref):
    name = col_ref.meta.output_name()
    vals = sorted(lf.select(col_ref.drop_nulls().unique()).collect()[name].to_list())
    seen.extend(vals)
    return col_ref

  df = pl.DataFrame({"a": ["x", "y", "z"]})
  expr = pl.when(pl.col("a") == "x").then(pl.lit("X")).otherwise(pl.col("a"))
  df.ps.with_columns(expr.ps.apply(capture))
  assert "X" in seen
  assert "x" not in seen


def test_apply_multi_col_transform():
  """With pl.col(pl.String).str.to_uppercase(), each fn sees its own uppercased values."""
  seen: dict[str, list[str]] = {}

  def capture(lf, col_ref):
    name = col_ref.meta.output_name()
    vals = lf.select(col_ref.drop_nulls().unique().sort()).collect()[name].to_list()
    seen[name] = vals
    return col_ref

  df = pl.DataFrame({"x": ["a", "b"], "y": ["p", "q"]})
  df.ps.with_columns(pl.col(pl.String).str.to_uppercase().ps.apply(capture))
  assert seen["x"] == ["A", "B"]
  assert seen["y"] == ["P", "Q"]


# ── ps.apply — col_ref.meta.output_name() ─────────────────────────────────────


def test_apply_meta_output_name_real_col():
  """col_ref.meta.output_name() returns the real column name."""
  names: list[str] = []

  def record(lf, col_ref):
    names.append(col_ref.meta.output_name())
    return col_ref

  df = pl.DataFrame({"foo": [1], "bar": [2]})
  df.ps.with_columns(pl.col("foo", "bar").ps.apply(record))
  assert sorted(names) == ["bar", "foo"]


def test_apply_meta_output_name_synthetic():
  """col_ref.meta.output_name() returns the synthetic name ('literal') for when/then."""
  names: list[str] = []

  def record(lf, col_ref):
    names.append(col_ref.meta.output_name())
    return col_ref

  df = pl.DataFrame({"a": ["x"]})
  df.ps.with_columns(
    pl.when(pl.col("a") == "x").then(pl.lit("X")).otherwise(pl.lit("Y")).ps.apply(record)
  )
  assert names == ["literal"]


# ── ps.apply — lf context ─────────────────────────────────────────────────────


def test_apply_lf_has_correct_filter_context():
  """fn's lf has preceding filters applied — categories discovered from filtered data."""

  def make_enum(lf, col_ref):
    name = col_ref.meta.output_name()
    cats = lf.select(col_ref.drop_nulls().unique().sort()).collect()[name].to_list()
    return col_ref.cast(pl.Enum(cats))

  df = pl.DataFrame({"x": ["a", "b", "c", "d"]})
  r = (
    df.lazy()
    .filter(pl.col("x").is_in(["a", "b"]))
    .ps.with_columns(pl.col("x").ps.apply(make_enum))
    .collect()
  )
  assert r["x"].dtype == pl.Enum(["a", "b"])  # "c" and "d" excluded by filter


# ── ps.apply — chaining ───────────────────────────────────────────────────────


def test_apply_chained_arithmetic():
  """Two ps.apply calls compose: (x + 1) * 2."""

  def add_one(lf, col_ref):
    return col_ref + 1.0

  def double(lf, col_ref):
    return col_ref * 2.0

  df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
  r = df.ps.with_columns(pl.col("x").ps.apply(add_one).ps.apply(double))
  assert r["x"].to_list() == pytest.approx([4.0, 6.0, 8.0])


def test_apply_chained_second_sees_first_output():
  """Second fn's col_ref reflects first fn's output, not the original column."""
  seen_second: list[str] = []

  def make_upper(lf, col_ref):
    return col_ref.str.to_uppercase()

  def capture(lf, col_ref):
    name = col_ref.meta.output_name()
    vals = lf.select(col_ref.drop_nulls().unique().sort()).collect()[name].to_list()
    seen_second.extend(vals)
    return col_ref

  df = pl.DataFrame({"x": ["a", "b", "c"]})
  df.ps.with_columns(pl.col("x").ps.apply(make_upper).ps.apply(capture))
  assert seen_second == ["A", "B", "C"]


# ── ps.apply — LazyFrame / empty ──────────────────────────────────────────────


def test_apply_lazyframe_returns_lazyframe():
  def identity(lf, col_ref):
    return col_ref

  df = pl.DataFrame({"x": [1.0, 2.0]})
  result = df.lazy().ps.with_columns(pl.col("x").ps.apply(identity))
  assert isinstance(result, pl.LazyFrame)


def test_apply_lazyframe_correct_values():
  def center(lf, col_ref):
    m = lf.select(col_ref.mean()).collect().item()
    return col_ref - m

  df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
  r = df.lazy().ps.with_columns(pl.col("x").ps.apply(center)).collect()
  assert r["x"].to_list() == pytest.approx([-1.0, 0.0, 1.0])


def test_apply_empty_df():
  def identity(lf, col_ref):
    return col_ref

  df = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64)})
  r = df.ps.with_columns(pl.col("x").ps.apply(identity))
  assert len(r) == 0
  assert "x" in r.columns


# ── ps.apply — keyword alias ──────────────────────────────────────────────────


def test_apply_keyword_alias():
  def double(lf, col_ref):
    return col_ref * 2.0

  df = pl.DataFrame({"x": [1.0, 2.0]})
  r = df.ps.with_columns(doubled=pl.col("x").ps.apply(double))
  assert "doubled" in r.columns
  assert r["doubled"].to_list() == pytest.approx([2.0, 4.0])


# ── ps.apply — idf docstring example ──────────────────────────────────────────


def test_apply_idf_example():
  """Reproduces the idf docstring example end-to-end."""
  df = pl.DataFrame(
    {
      "doc_id": [1, 1, 2, 2, 2],
      "term": ["cat", "dog", "cat", "cat", "bird"],
    }
  )

  def idf(lf: pl.LazyFrame, col_ref: pl.Expr) -> pl.Expr:
    col_name = col_ref.meta.output_name()
    n = lf.select(pl.col("doc_id").n_unique()).collect().item()
    freq = lf.group_by(col_ref).agg(pl.col("doc_id").n_unique().alias("n")).collect()
    scores = {r[col_name]: math.log(n / r["n"]) for r in freq.iter_rows(named=True)}
    return col_ref.replace_strict(list(scores), list(scores.values()), return_dtype=pl.Float64)

  r = df.ps.with_columns(pl.col("term").ps.apply(idf).alias("idf"))
  # cat appears in both docs → IDF = log(2/2) = 0
  # dog appears in 1 doc  → IDF = log(2/1) = log(2)
  cat_idf = r.filter(pl.col("term") == "cat")["idf"][0]
  dog_idf = r.filter(pl.col("term") == "dog")["idf"][0]
  assert cat_idf == pytest.approx(0.0)
  assert dog_idf == pytest.approx(math.log(2))


# ── CSE / map_elements call count ─────────────────────────────────────────────


def test_apply_map_elements_not_cse():
  """Polars does NOT apply CSE to Python UDFs (map_elements).

  When col_ref = pl.struct(expr).struct.field(name) appears in both the peek
  (lf.select(col_ref.mean())) and the final expression returned by fn, the
  underlying map_elements is called more than N times for N rows.

  This is a known Polars limitation: Python UDFs inside expressions referenced
  via pl.struct(...).struct.field(...) are re-evaluated at each use site.
  """
  call_count = [0]

  def counting_transform(x):
    call_count[0] += 1
    return x + 1.0

  df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
  expr = pl.col("x").map_elements(counting_transform, return_dtype=pl.Float64)

  def center(lf, col_ref):
    # col_ref appears in the peek AND in the returned expression → double evaluation
    m = lf.select(col_ref.mean()).collect().item()
    return col_ref - m

  df.ps.with_columns(expr.ps.apply(center))

  # Peek: col_ref.mean() evaluates map_elements once per row = 3 calls
  # Final: col_ref - m evaluates map_elements once per row = 3 more calls
  # Total: > 3 (exact count depends on Polars internals, but it is not CSE'd)
  assert call_count[0] > 3, f"Expected >3 calls (no CSE for Python UDFs), got {call_count[0]}"


def test_apply_plain_col_no_extra_eval():
  """A plain pl.col (no UDF) has no evaluation overhead concern."""

  def center(lf, col_ref):
    m = lf.select(col_ref.mean()).collect().item()
    return col_ref - m

  df = pl.DataFrame({"x": [10.0, 20.0, 30.0]})
  r = df.ps.with_columns(pl.col("x").ps.apply(center))
  assert r["x"].to_list() == pytest.approx([-10.0, 0.0, 10.0])
