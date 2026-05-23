from __future__ import annotations

from typing import Callable

import polars as pl


class FrameExpr:
  """An expression that requires a DataFrame context to resolve into a list of pl.Expr."""

  def __init__(self, col_expr: pl.Expr, resolver: Callable[[pl.DataFrame], list[pl.Expr]]):
    self._col_expr = col_expr
    self._resolver = resolver

  def resolve(self, df: pl.DataFrame) -> list[pl.Expr]:
    return self._resolver(df)

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
            lambda df: [getattr(e, name)(*args, **kwargs) for e in old_resolver(df)],
          )
        return result

      return method
    else:
      return FrameNamespaceProxy(attr, name, self)


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
        def chained(df):
          parent_exprs = old_resolver(df)
          temp_df = df.with_columns(parent_exprs)
          resolved = []
          for col in [e.meta.output_name() for e in parent_exprs]:
            col_ns = getattr(pl.col(col), ns_name)
            col_result = getattr(col_ns, method_name)(*args, **kwargs)
            if isinstance(col_result, FrameExpr):
              # Materialize on temp_df: the resolved exprs reference pl.col(col) in
              # temp_df (post-parent), so we cannot return them for the original df.
              computed = temp_df.select(col_result.resolve(temp_df))
              for col_name in computed.columns:
                resolved.append(pl.lit(computed[col_name]).alias(col_name))
            elif isinstance(col_result, pl.Expr):
              resolved.append(col_result)
          return resolved

        return FrameExpr(col_expr, chained)

      if isinstance(result, pl.Expr):
        return FrameExpr(
          result,
          lambda df: [
            getattr(getattr(e, ns_name), method_name)(*args, **kwargs)
            for e in old_resolver(df)
          ],
        )

      return result

    return method