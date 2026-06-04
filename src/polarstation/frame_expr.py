from __future__ import annotations

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

  def __repr__(self) -> str:
    return f"FrameExpr({self._col_expr!r})"

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
      result.append(fn(lf=lf, name=name, col_ref=col_ref, dtype=col_schema[name], **kwargs))
    return result

  return resolver


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
