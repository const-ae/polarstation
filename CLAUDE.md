# polarstation

Tidy helper functions for Polars, inspired by the R tidyverse (`ps_enum`, `ps_chop`, `ps_str`,
`ps.F`/`ps.B`/`ps.E`, `ps.format`/`ps.fmt_col`, `.ps.with_columns`/`.ps.select`).

## Before cutting a release (bumping the version in `pyproject.toml`)

1. **New public API must be added to `reference_config.yml`.** The reference page
   (`reference/index.qmd`) is *not* auto-discovered from the package — `scripts/build_reference.py`
   only documents functions/methods explicitly listed in `reference_config.yml`'s `sections`. Any
   new top-level `ps.*` function or `.ps_enum`/`.ps_chop`/`.ps_str`/`.ps` method that isn't added
   there silently never appears on the docs site, with no error or warning. Cross-check the NEWS.md
   entry being written for the release against `reference_config.yml` — every newly public
   name mentioned there should have a corresponding `contents:` entry.
2. **Rebuild and spot-check the reference page**: run `python3 scripts/build_reference.py`, then
   read the regenerated `reference/index.qmd` for the new entries — confirm the description landed
   in the right place (not merged into a code fence or missing) and the example fences are balanced.
   `docstring_info()` in `build_reference.py` only takes the *first* text block of a docstring as
   the description; a docstring that puts prose at the same indentation as `Examples:` (rather than
   nested under it) creates a second top-level text section, which is rendered as a trailing note
   after the examples, not part of the description — write docstrings with that in mind, or expect
   the split.
3. Update `NEWS.md` under the current version's header (`news.qmd` just includes `NEWS.md`, no
   separate edit needed there).
