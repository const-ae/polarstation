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


@pl.api.register_dataframe_namespace("ps")
class PolarstationDataFrame:
  def __init__(self, df: pl.DataFrame) -> None:
    self._df = df

  def with_columns(self, *exprs, **named_exprs) -> pl.DataFrame:
    """Like df.with_columns, but also accepts FrameExpr and multi-column selectors."""
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
        df = df.with_columns(item.resolve(df))
      else:
        batch.append(item)
    if batch:
      df = df.with_columns(batch)
    return df
