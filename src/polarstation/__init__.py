import importlib.resources

import polars as pl

from polarstation import chop_extensions as chop_extensions
from polarstation import dataframe_extensions as dataframe_extensions
from polarstation import enum_extensions as enum_extensions
from polarstation import expr_extensions as expr_extensions
from polarstation import string_extensions as string_extensions
from polarstation import typing as typing
from polarstation.func_extensions import B as B
from polarstation.func_extensions import E as E
from polarstation.func_extensions import F as F
from polarstation.typing import IntoExpr as IntoExpr


def hello() -> str:
  return "Hello from polarstation!"


def make_example_data(name: str = "animals") -> pl.DataFrame:
  """Return a small DataFrame used in the examples throughout the documentation.

  Args:
    name: Which dataset to return — ``"animals"`` (default), ``"scores"``, or
          ``"penguins"`` (344-row Palmer Penguins dataset).
  """
  if name == "animals":
    return pl.DataFrame(
      {
        "animal": ["dog", None, "bird", "cow", "bird"],
        "weight": [12.2, 7.5, 0.5, 460, None],
      }
    )
  if name == "scores":
    return pl.DataFrame({"score": [12, 45, 67, 89, 95, 23, 78]})
  if name == "penguins":
    csv = importlib.resources.files("polarstation").joinpath("data/penguins.csv")
    return pl.read_csv(csv, null_values="NA", try_parse_dates=True)
  raise ValueError(f"Unknown dataset {name!r}. Available: 'animals', 'scores', 'penguins'.")
