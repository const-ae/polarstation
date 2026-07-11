from __future__ import annotations

import operator
from collections.abc import Callable

import polars as pl


class FrameExpr:
  """An expression that requires a LazyFrame context to resolve into a list of pl.Expr.

  A plain ``pl.Expr`` is insufficient for operations like ``ps_enum.make()`` or
  ``ps_chop.chop()`` because Polars needs to know the output dtype (e.g. the exact
  ``pl.Enum([...])`` category list) at plan-construction time — before any data is
  seen.  ``FrameExpr`` defers that resolution to a two-phase execution model:

  **Phase 1 — peek**
      ``ps.with_columns`` calls ``resolve(lf)`` with the *current* ``LazyFrame``.
      The resolver runs a small aggregation (e.g. ``unique().sort()`` for category
      discovery, a handful of quantiles for binning) and collects it.  Because the
      resolver receives the full lazy plan up to that point, any preceding
      ``.filter()`` or ``.select()`` calls are already embedded and Polars'
      predicate/projection pushdown applies — only the relevant rows and columns are
      scanned.

  **Phase 2 — expression**
      The resolver uses the aggregation result to construct a concrete ``pl.Expr``
      with all dtype information baked in (e.g. ``pl.col("x").cast(pl.Enum(["a",
      "b", "c"]))``).  This expression is inserted back into the lazy plan and
      executed lazily together with all subsequent operations.

  When the peek is larger:
      Some operations — ``ps_chop.n_elements`` and ``ps_enum.reorder`` — must
      collect the full sorted column or a group aggregation to determine
      breakpoints.  These are genuinely O(N) collects, but they still only
      materialise a small result (unique values or group statistics), not the whole
      DataFrame.

  Note on parallel evaluation:
      When multiple ``FrameExpr`` columns appear in one ``ps.with_columns(...)``
      call, each resolver runs sequentially at the Python level.  This is an
      artifact of the current Python implementation; a future native Polars plugin
      could expose the resolvers to the query engine and allow parallel evaluation.
      In the meantime, prefer placing independent ``FrameExpr`` columns in the
      *same* ``ps.with_columns(...)`` call rather than chaining multiple calls, so
      each lazy plan (with its pushdown) is only materialised once.

  Performance: expressions that are expensive to evaluate
      Under the hood every resolver references the original ``expr`` via
      ``pl.struct(expr).struct.field(name)``. This means for a regular expression
      like ``expr.mean()`` this will only evaluated because of polars common subexpression
      elemination. However, for user-defined python function called via ``map_elements`` or
      ``map_batches`` polars does not apply these optimization and they are called twice.
  """

  def __init__(self, col_expr: pl.Expr, resolver: Callable[[pl.LazyFrame], list[pl.Expr]]):
    self._col_expr = col_expr
    self._resolver = resolver

  def resolve(self, lf: pl.LazyFrame) -> list[pl.Expr]:
    return self._resolver(lf)

  def over(self, *by, **kwargs) -> FrameExpr:
    """Group-aware version of this FrameExpr, analogous to pl.Expr.over().

    Partitions the frame by ``by``, reruns this (unmodified) resolver once per
    partition, and concatenates the results back together — correct for any
    ``ps_enum``/``ps_chop`` method and any composition of them (e.g.
    ``.ps_enum.infreq().ps_enum.to_level()``), since a resolver is just
    ``Callable[[LazyFrame], list[pl.Expr]]`` regardless of how deep the chain is.

    This requires an eager collect of the full frame: a following ``.filter()`` cannot
    retroactively narrow work that already happened when ``.over()`` was resolved
    (this limitation is inherent to FrameExpr's "peek eagerly" design and applies to
    the ungrouped case too — e.g. ``make()``'s categories already come from the whole
    frame regardless of a later filter). A *preceding* ``.filter()`` still helps, since
    it's embedded in the ``lf`` this resolver receives.

    An earlier version of this method dispatched to per-method "fast path"
    implementations (native ``.over()`` broadcasts, struct-keyed ``replace_strict``
    lookups) that avoided this collect. That approach was dropped: it introduced a
    second implementation of each method's semantics that could silently diverge from
    the ungrouped one (e.g. ``to_level()``'s fast path disagreed with the ungrouped
    result for an Enum column with an unobserved category), and benchmarking showed no
    reliable speed advantage over this single mechanism — some fast paths were
    measurably slower. One correct implementation beats two fast-but-risky ones.
    """
    by_exprs = _normalize_over_by(by)
    fallback = _generic_partition_resolver(self._resolver, by_exprs, **kwargs)
    return FrameExpr(self._col_expr, fallback)

  def __repr__(self) -> str:
    return f"FrameExpr({self._col_expr!r})"

  def __invert__(self) -> FrameExpr:
    old_resolver = self._resolver
    return FrameExpr(~self._col_expr, lambda lf: [~e for e in old_resolver(lf)])

  def __getattr__(self, name: str):
    if name.startswith("_"):
      raise AttributeError(name)
    attr = getattr(self._col_expr, name)
    old_resolver = self._resolver

    if callable(attr):

      def method(*args, **kwargs):
        result = attr(*args, **kwargs)
        if isinstance(result, pl.Expr):
          return FrameExpr(
            result,
            lambda lf: [getattr(e, name)(*args, **kwargs) for e in old_resolver(lf)],
          )
        return result

      return method
    else:
      return FrameNamespaceProxy(attr, name, self)


def _make_binary_dunder(op: Callable[[pl.Expr, object], pl.Expr]) -> Callable:
  def method(self: FrameExpr, other: object) -> FrameExpr:
    old_resolver = self._resolver
    return FrameExpr(op(self._col_expr, other), lambda lf: [op(e, other) for e in old_resolver(lf)])

  return method


# Comparison and arithmetic operators aren't reached by __getattr__ — Python resolves
# dunder methods on the type, bypassing instance-level attribute fallback — so FrameExpr
# needs them defined explicitly to support expressions like `frame_expr < 3`. Named-method
# equivalents (`.lt()`, `.add()`, etc.) already work today via __getattr__; this is purely
# an ergonomics addition for bare operators, especially useful with .over(), e.g.
# `(pl.col('a').ps_enum.infreq().ps_enum.to_level() < 3).over('grouping')`.
for _dunder_name, _op in {
  "__lt__": operator.lt,
  "__le__": operator.le,
  "__gt__": operator.gt,
  "__ge__": operator.ge,
  "__eq__": operator.eq,
  "__ne__": operator.ne,
  "__add__": operator.add,
  "__sub__": operator.sub,
  "__mul__": operator.mul,
  "__truediv__": operator.truediv,
  "__and__": operator.and_,
  "__or__": operator.or_,
}.items():
  setattr(FrameExpr, _dunder_name, _make_binary_dunder(_op))


def resolve_across_columns(
  expr: pl.Expr,
  fn: Callable,
  **kwargs,
) -> Callable[[pl.LazyFrame], list[pl.Expr]]:
  """Return a resolver that calls fn once per output column of expr.

  fn must accept keyword arguments: lf, name, col_ref, dtype, plus any **kwargs.

  col_ref = pl.struct(expr).struct.field(name) — evaluates the column values
  correctly for any expression shape: real column references, multi-column
  selectors, transforms, and when/then/otherwise with synthetic output names
  (e.g. 'literal'). This is the correct reference to use instead of pl.col(name),
  which breaks when name is not a column in lf or when values are transformed.
  """

  def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    col_schema = lf.select(expr).collect_schema()
    result = []
    for name in col_schema.names():
      col_ref = pl.struct(expr).struct.field(name)
      expr_result = fn(lf=lf, name=name, col_ref=col_ref, dtype=col_schema[name], **kwargs)
      result.append(expr_result.alias(name))
    return result

  return resolver


def _normalize_over_by(by: tuple) -> list[pl.Expr]:
  """Flatten and normalize .over(*by) arguments the same way pl.Expr.over() does.

  Accepts individual args, a single list/tuple of them, or a mix; strings become
  pl.col(str) so they can be used as ordinary expressions downstream.
  """
  flat: list = []
  for item in by:
    if isinstance(item, (list, tuple)):
      flat.extend(item)
    else:
      flat.append(item)
  return [pl.col(item) if isinstance(item, str) else item for item in flat]


def _unused_column_name(lf: pl.LazyFrame, base: str) -> str:
  """Return `base`, or a suffixed variant, that isn't already a column of lf."""
  existing = set(lf.collect_schema().names())
  if base not in existing:
    return base
  i = 0
  while f"{base}{i}" in existing:
    i += 1
  return f"{base}{i}"


def _union_enum_categories(dtypes: list[pl.DataType]) -> list[str]:
  """Union the categories of one or more Enum dtypes, ordered by first appearance."""
  seen: set[str] = set()
  union: list[str] = []
  for dtype in dtypes:
    if not isinstance(dtype, pl.Enum):
      continue
    for cat in dtype.categories.to_list():
      if cat not in seen:
        seen.add(cat)
        union.append(cat)
  return union


def _generic_partition_resolver(
  old_resolver: Callable[[pl.LazyFrame], list[pl.Expr]],
  by_exprs: list[pl.Expr],
  **kwargs,
) -> Callable[[pl.LazyFrame], list[pl.Expr]]:
  """Grouped resolver used by every FrameExpr.over() call: partition eagerly, rerun the
  unmodified resolver per group, concatenate back together.

  Always correct for arbitrary chains of ps_enum/ps_chop calls, since a resolver is just
  Callable[[LazyFrame], list[pl.Expr]] regardless of how deep the chain is — but requires
  collecting the whole (relevant columns of the) frame, since the per-row results of every
  group must be materialized and concatenated back together.
  """
  if kwargs:
    raise TypeError(f"over() does not accept keyword arguments for chained expressions: {kwargs}")

  def grouped_resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
    idx_name = _unused_column_name(lf, "__ps_over_idx__")
    by_names = [_unused_column_name(lf, f"__ps_over_by_{i}__") for i in range(len(by_exprs))]

    tagged = lf.with_row_index(idx_name).with_columns(
      [be.alias(n) for be, n in zip(by_exprs, by_names)]
    )
    tagged_df = tagged.collect()
    if tagged_df.height == 0:
      # No rows to group — nothing to partition. Fall back to the ordinary resolver so an
      # empty frame still gets correctly-typed (e.g. Enum([])) output, matching the
      # ungrouped behavior for empty input.
      return old_resolver(lf)
    partitions = tagged_df.partition_by(by_names, maintain_order=True)

    col_names: list[str] | None = None
    per_column_parts: dict[str, list[pl.DataFrame]] = {}
    for part_df in partitions:
      part_lf = part_df.lazy()
      part_exprs = old_resolver(part_lf)
      resolved = part_lf.select(idx_name, *part_exprs).collect()
      if col_names is None:
        col_names = [e.meta.output_name() for e in part_exprs]
      for name in col_names:
        per_column_parts.setdefault(name, []).append(resolved.select(idx_name, name))

    result = []
    for name in col_names or []:
      parts = per_column_parts[name]
      dtypes = [p[name].dtype for p in parts]
      if any(isinstance(d, pl.Enum) for d in dtypes):
        union = _union_enum_categories(dtypes)
        parts = [p.with_columns(pl.col(name).cast(pl.Enum(union))) for p in parts]
      combined = pl.concat(parts, how="vertical_relaxed").sort(idx_name)
      result.append(pl.lit(combined[name]).alias(name))
    return result

  return grouped_resolver


class FrameNamespaceProxy:
  """Wraps a Polars expression namespace so results are lifted into FrameExpr."""

  def __init__(self, namespace, ns_name: str, parent: FrameExpr):
    self._namespace = namespace
    self._ns_name = ns_name
    self._parent = parent

  def __getattr__(self, method_name: str):
    ns_attr = getattr(self._namespace, method_name)

    if not callable(ns_attr):
      return FrameNamespaceProxy(ns_attr, f"{self._ns_name}.{method_name}", self._parent)

    parent = self._parent
    ns_name = self._ns_name

    def method(*args, **kwargs):
      result = ns_attr(*args, **kwargs)
      old_resolver = parent._resolver
      col_expr = parent._col_expr

      if isinstance(result, FrameExpr):
        # Resolve parent first, then re-run the method per resolved column so that
        # multi-column selectors (e.g. pl.col(pl.String)) chain correctly.
        def chained(lf: pl.LazyFrame):
          parent_exprs = old_resolver(lf)
          temp_lf = lf.with_columns(parent_exprs)
          resolved = []
          for col in [e.meta.output_name() for e in parent_exprs]:
            col_ns = getattr(pl.col(col), ns_name)
            col_result = getattr(col_ns, method_name)(*args, **kwargs)
            if isinstance(col_result, FrameExpr):
              # Materialize on temp_lf: the resolved exprs reference pl.col(col) in
              # temp_lf (post-parent), so we cannot return them for the original lf.
              computed = temp_lf.select(col_result.resolve(temp_lf)).collect()
              for col_name in computed.columns:
                resolved.append(pl.lit(computed[col_name]).alias(col_name))
            elif isinstance(col_result, pl.Expr):
              resolved.append(col_result)
          return resolved

        return FrameExpr(col_expr, chained)

      if isinstance(result, pl.Expr):
        return FrameExpr(
          result,
          lambda lf: [
            getattr(getattr(e, ns_name), method_name)(*args, **kwargs) for e in old_resolver(lf)
          ],
        )

      return result

    return method
