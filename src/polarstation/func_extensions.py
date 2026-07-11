from __future__ import annotations

from collections.abc import Callable

import polars as pl

from polarstation.frame_expr import FrameExpr


def _split_expr_args(args: tuple, kwargs: dict) -> tuple[list[tuple[int | str, pl.Expr]], Callable]:
  """Pick out the pl.Expr values among args/kwargs; everything else passes through as-is.

  Only pl.Expr values need resolving against a LazyFrame — a plain string, number, or
  other Python value (e.g. `k=2` for a clustering function's cluster count) is not a
  column reference and is forwarded to fn unchanged, in its original position/keyword.

  Returns (expr_items, rebuild): expr_items is [(key, expr), ...] in call order, where
  key is an int index into args or a str keyword name. rebuild(values) reassembles the
  full (args, kwargs) pair, substituting `values` (in expr_items order) for each Expr.
  """
  expr_items = [(i, a) for i, a in enumerate(args) if isinstance(a, pl.Expr)]
  expr_items += [(k, v) for k, v in kwargs.items() if isinstance(v, pl.Expr)]

  def rebuild(values) -> tuple[list, dict]:
    resolved_args = list(args)
    resolved_kwargs = dict(kwargs)
    for (key, _), value in zip(expr_items, values, strict=True):
      if isinstance(key, int):
        resolved_args[key] = value
      else:
        resolved_kwargs[key] = value
    return resolved_args, resolved_kwargs

  return expr_items, rebuild


def F(fn: Callable) -> Callable[..., FrameExpr]:  # noqa: N802
  """Turn an arbitrary function into something callable on expressions, with real data.

  ``fn`` is called exactly once, eagerly, with the complete data (as ``pl.Series``) for
  its ``pl.Expr`` arguments — never a sample, dummy data, or a batch/slice. That means
  no ``return_dtype`` is needed: the output dtype is whatever ``fn`` actually produced.
  This is the right default for arbitrary vectorized functions (numpy, scipy, ...)
  that don't otherwise fit Polars' expression API.

  Only ``pl.Expr`` arguments (positional or keyword) are resolved against the data;
  any other argument (a plain string, number, ...) is forwarded to ``fn`` unchanged —
  useful for a function's non-column parameters, e.g. a cluster count.

  The cost is an eager ``.collect()`` of ``fn``'s ``pl.Expr`` arguments at the point
  ``ps.F(fn)(...)`` is resolved (via ``ps.with_columns``/``ps.select``), the same
  tradeoff ``ps_chop`` and ``ps_enum`` already make for operations that need to see
  real data. Preceding ``.filter()``/``.select()`` calls are still pushed down, but
  nothing after this point can narrow the collect retroactively.

  For a function that must stay fully lazy (e.g. inside a `.over()` or streaming
  pipeline) and can tolerate being called on batches/slices instead of the full
  column, use `ps.B` instead. For a plain Python function that only accepts scalars
  (not arrays), use `ps.E`.

  Examples:
    ```{python}
    import polars as pl
    import polarstation as ps
    from scipy.cluster.hierarchy import fclusterdata

    df = pl.DataFrame({"x": [0.0, 0.0, 10.0, 10.0], "y": [0.0, 0.2, 0.0, 0.2]})
    df.ps.with_columns(
        cluster=ps.F(fclusterdata)(pl.concat_arr("x", "y"), t=2, criterion="maxclust")
    )
    ```
  """

  def call(*args, **kwargs) -> FrameExpr:
    expr_items, rebuild = _split_expr_args(args, kwargs)
    col_expr = expr_items[0][1] if expr_items else pl.lit(None)

    def resolver(lf: pl.LazyFrame) -> list[pl.Expr]:
      if not expr_items:
        result = fn(*args, **kwargs)
        return [pl.lit(pl.Series(result)).alias("literal")]

      names = [f"_arg{i}" for i in range(len(expr_items))]
      data = lf.select([e.alias(n) for n, (_, e) in zip(names, expr_items, strict=True)]).collect()
      resolved_args, resolved_kwargs = rebuild([data[n] for n in names])
      result = fn(*resolved_args, **resolved_kwargs)
      return [pl.lit(pl.Series(result)).alias(col_expr.meta.output_name())]

    return FrameExpr(col_expr, resolver)

  return call


def B(  # noqa: N802
  fn: Callable,
  return_dtype: pl.DataTypeExpr | pl.DataType | None = None,
  *,
  is_elementwise: bool = False,
  **map_batches_kwargs,
) -> Callable[..., pl.Expr]:
  """Turn an arbitrary vectorized function into something callable on expressions, lazily.

  Thin wrapper around ``pl.map_batches``: ``fn``'s ``pl.Expr`` arguments (positional or
  keyword) are resolved to ``pl.Series``, but — unlike ``ps.F`` — it may be called more
  than once, and on batches/slices rather than the complete column (e.g. under
  streaming execution, or once per group inside ``.over()``/``group_by().agg()``). In
  exchange, the result stays fully lazy: no eager collect is forced at the call site.
  As with ``ps.F``, any non-``pl.Expr`` argument is forwarded to ``fn`` unchanged.

  If ``return_dtype`` is left unset, Polars infers it by calling ``fn`` once with
  synthetic dummy data — this can raise for domain-restricted functions, or infer the
  wrong dtype. Prefer ``ps.F`` unless you specifically need laziness/streaming and can
  either supply ``return_dtype`` or tolerate that inference step.

  Examples:
    ```{python}
    import numpy as np
    import polars as pl
    import polarstation as ps

    df = pl.DataFrame({"log_p": [-0.5, -3.0], "log_q": [-1.2, -0.4]})
    df.lazy().with_columns(
        combined=ps.B(np.logaddexp, return_dtype=pl.Float64)(pl.col("log_p"), pl.col("log_q"))
    ).collect()
    ```
  """

  def call(*args, **kwargs) -> pl.Expr:
    expr_items, rebuild = _split_expr_args(args, kwargs)
    exprs = [e for _, e in expr_items]

    def batched(series: list[pl.Series]):
      resolved_args, resolved_kwargs = rebuild(series)
      return fn(*resolved_args, **resolved_kwargs)

    return pl.map_batches(
      exprs,
      batched,
      return_dtype=return_dtype,
      is_elementwise=is_elementwise,
      **map_batches_kwargs,
    )

  return call


def E(  # noqa: N802
  fn: Callable,
  return_dtype: pl.DataTypeExpr | pl.DataType | None = None,
  **map_elements_kwargs,
) -> Callable[..., pl.Expr]:
  """Turn a scalar (non-vectorized) Python function into something callable on expressions.

  ``fn`` is called once per row with plain Python scalars — the multi-argument
  equivalent of ``pl.Expr.map_elements``, built on ``pl.struct(...).map_elements(...)``.
  As with ``ps.F``/``ps.B``, only ``pl.Expr`` arguments (positional or keyword) become
  per-row values; any other argument is forwarded to ``fn`` unchanged. Note that
  ``skip_nulls`` (defaults to ``True``, can be overridden via ``**map_elements_kwargs``)
  only skips a row when the *entire struct* is null, which struct values built from
  columns essentially never are — a null in a single argument still reaches ``fn`` as
  ``None``, so ``fn`` must be able to handle that itself if any argument column has
  nulls.

  Unlike ``ps.F``/``ps.B``, there is no vectorized fast path here — ``fn`` runs once per
  row. Only reach for ``ps.E`` when ``fn`` genuinely cannot operate on whole arrays at
  once (e.g. it calls into a scalar-only library). For anything that accepts numpy
  arrays or ``pl.Series`` directly, prefer ``ps.F``.

  Examples:
    ```{python}
    import polars as pl
    import polarstation as ps

    def levenshtein(a, b):
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i] + [0] * len(b)
            for j, cb in enumerate(b, 1):
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
            prev = curr
        return prev[-1]

    df = pl.DataFrame({"typed": ["aplpe", "bananna"], "correct": ["apple", "banana"]})
    df.with_columns(
        dist=ps.E(levenshtein, return_dtype=pl.Int64)(pl.col("typed"), pl.col("correct"))
    )
    ```
  """

  def call(*args, **kwargs) -> pl.Expr:
    expr_items, rebuild = _split_expr_args(args, kwargs)
    names = [f"_arg{i}" for i in range(len(expr_items))]

    def per_row(row: dict):
      resolved_args, resolved_kwargs = rebuild([row[n] for n in names])
      return fn(*resolved_args, **resolved_kwargs)

    struct_fields = {n: e for n, (_, e) in zip(names, expr_items, strict=True)}
    return pl.struct(**struct_fields).map_elements(
      per_row, return_dtype=return_dtype, **map_elements_kwargs
    )

  return call
