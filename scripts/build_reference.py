#!/usr/bin/env python3
"""Pre-render script: generates reference/index.qmd as a single long page."""

import re
from pathlib import Path

import griffe
import yaml

CONFIG = Path("reference_config.yml")
OUT = Path("reference/index.qmd")


def to_anchor(display_name: str) -> str:
    return display_name.lower().replace(".", "-").replace("_", "-")


def _dedent_fenced_blocks(text: str) -> str:
    """Strip the leading indentation off any ```-fenced block nested inside prose.

    Griffe preserves a docstring's original relative indentation verbatim, and this
    codebase sometimes nests a ```{python} fence one level under a prose paragraph (rather
    than at column 0). Quarto/pandoc only reliably recognizes fenced code blocks that start
    at the same indentation as their surrounding block — an indented fence here trips up its
    cell-splitting and produces stray ':::' fenced-div warnings (and a broken render).
    """
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        m = re.match(r"^(\s+)```", lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent = m.group(1)
        out.append(lines[i][len(indent) :])
        i += 1
        while i < len(lines):
            line = lines[i]
            out.append(line[len(indent) :] if line.startswith(indent) else line)
            i += 1
            if line.strip() == "```":
                break
    return "\n".join(out)


def load_obj(pkg, dotted_path: str):
    obj = pkg
    for part in dotted_path.split("."):
        obj = obj[part]
    return obj


def docstring_info(obj) -> tuple[str, dict[str, str], str, list[str]]:
    """Return (main description, {param_name: description}, examples_code, trailing_notes)."""
    main_desc, param_docs, examples_parts, trailing_notes = "", {}, [], []
    if obj.docstring is None:
        return main_desc, param_docs, "", []
    for section in obj.docstring.parsed:
        if section.kind.value == "text":
            # Google-style docstrings only nest trailing prose under "Examples:" if it's
            # indented deeper than the header; text at the same indent (a common mistake in
            # this codebase) starts a new top-level "text" section instead. Treat the first
            # such section as the description and any later ones as trailing notes, rendered
            # after the examples — never folded into examples_code, since that string gets
            # wrapped wholesale in a ```{python} fence when it isn't already fenced, and a
            # prose note caught in that fence would break execution of the rendered page.
            if not main_desc:
                main_desc = section.value.strip()
            else:
                trailing_notes.append(_dedent_fenced_blocks(section.value.strip()))
        elif section.kind.value == "parameters":
            for p in section.value:
                param_docs[p.name] = p.description.strip()
        elif section.kind.value == "examples":
            for _kind, text in section.value:
                examples_parts.append(text.strip())
    return main_desc, param_docs, "\n\n".join(examples_parts), trailing_notes


def render_signature(display_name: str, obj) -> str:
    parts = []
    for p in obj.parameters:
        if p.name == "self":
            continue
        if p.kind.name == "var_positional":
            parts.append(f"*{p.name}")
        elif p.kind.name == "var_keyword":
            parts.append(f"**{p.name}")
        else:
            parts.append(f"{p.name}={p.default}" if p.default is not None else p.name)
    if not parts:
        return f"{display_name}()"
    indent = "    "
    args = f",\n{indent}".join(parts)
    return f"{display_name}(\n{indent}{args},\n)"


def render_detail(display_name: str, obj) -> list[str]:
    main_desc, param_docs, examples_code, trailing_notes = docstring_info(obj)
    sig_params = [p for p in obj.parameters if p.name != "self"]

    lines = [f"### {display_name} {{#{to_anchor(display_name)}}}", ""]
    lines += ["```python", render_signature(display_name, obj), "```", ""]

    if main_desc:
        lines += [main_desc, ""]

    for param in sig_params:
        ann = str(param.annotation) if param.annotation is not None else ""
        default = str(param.default) if param.default is not None else ""

        if param.kind.name == "var_positional":
            label = f"**\\*{param.name}**"
        elif param.kind.name == "var_keyword":
            label = f"**\\*\\*{param.name}**"
        else:
            label = f"**{param.name}**"

        if ann and ann != "None":
            label += f" `{ann}`"
        if default and default != "None":
            label += f" = `{default}`"

        desc = param_docs.get(param.name, "")
        lines.append(label)
        if desc:
            lines.append(f":   {desc}")
        lines.append("")

    if examples_code:
        if examples_code.lstrip().startswith("```"):
            lines += ["**Examples:**", "", examples_code, ""]
        else:
            lines += ["**Examples:**", "", "```{python}", examples_code, "```", ""]

    for note in trailing_notes:
        lines += [note, ""]

    return lines


def main():
    config = yaml.safe_load(CONFIG.read_text())
    package = config["package"]
    sections = config["sections"]

    pkg = griffe.load(package, docstring_parser="google")

    # Resolve objects once
    resolved: list[tuple[str, str, object]] = []
    for section in sections:
        for item in section.get("contents", []):
            obj = load_obj(pkg, item["path"])
            resolved.append((item["display"], item["path"], obj))

    lines: list[str] = ["---", "toc: true", "toc-depth: 3", "---", ""]

    # ── Overview tables ──────────────────────────────────────────────────────
    in_overview = False
    for section in sections:
        is_overview = section.get("overview", False)
        if is_overview and not in_overview:
            lines += ["## Overview", ""]
            in_overview = True
        elif not is_overview and in_overview:
            in_overview = False

        level = "###" if is_overview else "##"
        lines += [f"{level} {section['title']}", ""]
        if desc := section.get("desc", ""):
            lines += [desc, ""]

        lines += ["| | |", "| -- | ----- |"]
        for item in section["contents"]:
            display = item["display"]
            obj = load_obj(pkg, item["path"])
            main_desc, _, _examples, _notes = docstring_info(obj)
            short_desc = main_desc.splitlines()[0] if main_desc else ""
            an = to_anchor(display)
            display_html = display.replace(".", ".<wbr>")
            link = f'<a href="#{an}"><code>{display_html}</code></a>'
            lines.append(f"| {link} | {short_desc} |")
        lines.append("")

    # ── Detailed sections ────────────────────────────────────────────────────
    setup = config.get("setup", "")
    lines += ["---", ""]
    if setup:
        lines += ["```{python}", "#| echo: false", setup.rstrip(), "```", ""]
    for section in sections:
        lines += [f"## {section['title']} {{.doc-details}}", ""]
        for item in section["contents"]:
            obj = load_obj(pkg, item["path"])
            lines += render_detail(item["display"], obj)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"Generated {OUT}")


if __name__ == "__main__":
    main()
