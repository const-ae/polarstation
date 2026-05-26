import polars as pl

from polarstation.frame_expr import FrameExpr


def _flatten(items):
  """Flatten nested iterables, but treat pl.Expr and FrameExpr as atomic."""
  for item in items:
    if isinstance(item, (pl.Expr, FrameExpr)):
      yield item
    elif hasattr(item, "__iter__") and not isinstance(item, str):
      yield from _flatten(item)
    else:
      yield item


@pl.api.register_lazyframe_namespace("ps")
class PolarstationLazyFrame:
  def __init__(self, lf: pl.LazyFrame) -> None:
    self._lf = lf

  def with_columns(self, *exprs, **named_exprs) -> pl.LazyFrame:
    """Like lf.with_columns, but also accepts FrameExpr.

    Plain pl.Expr items stay fully lazy. When a FrameExpr is encountered the
    accumulated lazy plan is collected at that point (so preceding filters and
    projections are pushed down before any data is read), resolved against the
    resulting DataFrame, then execution continues lazily.
    """
    all_items = list(_flatten(exprs))

    for key, val in named_exprs.items():
      if isinstance(val, FrameExpr):
        original = val
        all_items.append(FrameExpr(
          original._col_expr,
          lambda df, fe=original, k=key: [e.alias(k) for e in fe.resolve(df)],
        ))
      elif isinstance(val, pl.Expr):
        all_items.append(val.alias(key))
      else:
        all_items.append(pl.lit(val).alias(key))

    batch: list[pl.Expr] = []
    lf = self._lf
    for item in all_items:
      if isinstance(item, FrameExpr):
        if batch:
          lf = lf.with_columns(batch)
          batch = []
        lf = lf.with_columns(item.resolve(lf))
      else:
        batch.append(item)
    if batch:
      lf = lf.with_columns(batch)
    return lf

  def select(self, *exprs, **named_exprs) -> pl.LazyFrame:
    """Like lf.select, but also accepts FrameExpr.

    All FrameExprs peek at the lazy plan as it stands before this call, then a
    single select is issued with all resolved expressions.
    """
    all_items = list(_flatten(exprs))

    for key, val in named_exprs.items():
      if isinstance(val, FrameExpr):
        original = val
        all_items.append(FrameExpr(
          original._col_expr,
          lambda lf, fe=original, k=key: [e.alias(k) for e in fe.resolve(lf)],
        ))
      elif isinstance(val, pl.Expr):
        all_items.append(val.alias(key))
      else:
        all_items.append(pl.lit(val).alias(key))

    resolved: list[pl.Expr] = []
    for item in all_items:
      if isinstance(item, FrameExpr):
        resolved.extend(item.resolve(self._lf))
      else:
        resolved.append(item)
    return self._lf.select(resolved)


@pl.api.register_dataframe_namespace("ps")
class PolarstationDataFrame:
  def __init__(self, df: pl.DataFrame) -> None:
    self._df = df

  def with_columns(self, *exprs, **named_exprs) -> pl.DataFrame:
    """Like df.with_columns, but also accepts FrameExpr and multi-column selectors.

    Examples:
      animals = polarstation.make_example_data("animals")
      animals.ps.with_columns(
          pl.col("animal").ps_enum.make().ps_enum.reorder(by="weight")
      )
    """
    all_items = list(_flatten(exprs))

    for key, val in named_exprs.items():
      if isinstance(val, FrameExpr):
        original = val
        all_items.append(FrameExpr(
          original._col_expr,
          lambda df, fe=original, k=key: [e.alias(k) for e in fe.resolve(df)],
        ))
      elif isinstance(val, pl.Expr):
        all_items.append(val.alias(key))
      else:
        all_items.append(pl.lit(val).alias(key))

    batch: list[pl.Expr] = []
    df = self._df
    for item in all_items:
      if isinstance(item, FrameExpr):
        if batch:
          df = df.with_columns(batch)
          batch = []
        df = df.with_columns(item.resolve(df.lazy()))
      else:
        batch.append(item)
    if batch:
      df = df.with_columns(batch)
    return df

  def select(self, *exprs, **named_exprs) -> pl.DataFrame:
    """Like df.select, but also accepts FrameExpr.

    Examples:
      animals = polarstation.make_example_data("animals")
      animals.ps.select(pl.col("animal").ps_enum.make(), "weight")
    """
    all_items = list(_flatten(exprs))

    for key, val in named_exprs.items():
      if isinstance(val, FrameExpr):
        original = val
        all_items.append(FrameExpr(
          original._col_expr,
          lambda lf, fe=original, k=key: [e.alias(k) for e in fe.resolve(lf)],
        ))
      elif isinstance(val, pl.Expr):
        all_items.append(val.alias(key))
      else:
        all_items.append(pl.lit(val).alias(key))

    lf = self._df.lazy()
    resolved: list[pl.Expr] = []
    for item in all_items:
      if isinstance(item, FrameExpr):
        resolved.extend(item.resolve(lf))
      else:
        resolved.append(item)
    return self._df.select(resolved)
