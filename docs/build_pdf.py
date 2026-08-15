#!/usr/bin/env python3
"""Renders the project summaries to PDF.

    python docs/build_pdf.py            # both languages
    python docs/build_pdf.py pl         # one of them

Markdown -> HTML with python-markdown, HTML -> PDF with WeasyPrint. Neither
LaTeX nor pandoc: the documents use headings, paragraphs, lists, tables and
code blocks, and nothing here needs a typesetting engine. LibreOffice was tried
first, since it was already installed, and dropped - it discards the stylesheet
on HTML import, so every table came out without rules and with its rows spread
down the page. These documents are largely tables.

Dependencies: python-markdown, python-weasyprint.
"""

import sys
from pathlib import Path

import markdown
from weasyprint import HTML

DOCS = Path(__file__).resolve().parent

# One per language, so the running footer and the PDF metadata are in the same
# language as the text.
BOOKS = {
    "pl": ("Solitito_project_summary_pl.md", "Solitito — podsumowanie projektu", "pl"),
    "en": ("Solitito_project_summary_en.md", "Solitito — Project Summary", "en"),
}

# Print styling. The screen has no say here - every length is in millimetres or
# points, and the page is A4.
CSS = """
@page {
    size: A4;
    margin: 20mm 18mm 18mm 18mm;
    @bottom-center {
        content: "__FOOTER__  ·  " counter(page) " / " counter(pages);
        font-family: "Noto Sans", "DejaVu Sans", sans-serif;
        font-size: 8pt;
        color: #777;
    }
}
/* The title page carries the title; a footer under it would be noise. */
@page :first { @bottom-center { content: ""; } }

html { font-size: 10.5pt; }
body {
    font-family: "Noto Serif", "DejaVu Serif", serif;
    line-height: 1.42;
    color: #16181c;
    hyphens: auto;
    orphans: 3;
    widows: 3;
    text-align: justify;
}

h1, h2, h3, h4 {
    font-family: "Noto Sans", "DejaVu Sans", sans-serif;
    color: #0d0f13;
    break-after: avoid;
    text-align: left;
}
h1 { font-size: 20pt; margin: 0 0 2mm 0; letter-spacing: -0.2pt; }
h2 {
    font-size: 13.5pt;
    margin: 9mm 0 2.5mm 0;
    padding-bottom: 1.2mm;
    border-bottom: 0.6pt solid #c9ced6;
    break-before: page;
}
/* The first chapter follows the title block on page one. */
h2:first-of-type { break-before: avoid; }
h3 { font-size: 11pt; margin: 6mm 0 1.5mm 0; }

p { margin: 0 0 2.6mm 0; }
ul, ol { margin: 0 0 2.6mm 0; padding-left: 5mm; }
li { margin-bottom: 1.2mm; }

strong { color: #000; }
a { color: #1a4fa0; text-decoration: none; }

code {
    font-family: "DejaVu Sans Mono", monospace;
    font-size: 0.86em;
    background: #f1f2f4;
    padding: 0.3mm 0.8mm;
    border-radius: 1pt;
}
pre {
    font-family: "DejaVu Sans Mono", monospace;
    font-size: 8.4pt;
    line-height: 1.32;
    background: #f6f7f9;
    border-left: 2pt solid #9aa3b0;
    padding: 2mm 3mm;
    margin: 0 0 3mm 0;
    white-space: pre-wrap;
    break-inside: avoid;
    text-align: left;
}
pre code { background: none; padding: 0; font-size: 1em; }

table {
    border-collapse: collapse;
    width: 100%;
    font-family: "Noto Sans", "DejaVu Sans", sans-serif;
    font-size: 8.8pt;
    margin: 0 0 4mm 0;
    text-align: left;
}
/* A table split across a page break repeats its header rather than stranding
   the reader with unlabelled columns. */
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td {
    border: 0.5pt solid #b9bfc8;
    padding: 1.3mm 1.8mm;
    vertical-align: top;
    text-align: left;
}
th { background: #eceef1; font-weight: 600; }
td code { font-size: 0.92em; }

hr { border: none; border-top: 0.5pt solid #d5d9df; margin: 6mm 0; }

/* The italic lines under the title: subtitle, version, date. */
body > p em { color: #555; }
"""


def build(lang: str) -> Path:
    name, footer, code = BOOKS[lang]
    src = DOCS / name
    text = src.read_text(encoding="utf-8")

    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html5",
    )
    html = (
        f'<!doctype html><html lang="{code}"><head><meta charset="utf-8">'
        f"<title>{footer}</title>"
        f"<style>{CSS.replace('__FOOTER__', footer)}</style></head><body>{body}</body></html>"
    )

    out = src.with_suffix(".pdf")
    HTML(string=html, base_url=str(DOCS)).write_pdf(out)
    return out


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(BOOKS)
    for lang in wanted:
        if lang not in BOOKS:
            sys.exit(f"unknown language {lang!r}; expected one of {', '.join(BOOKS)}")
        pdf = build(lang)
        print(f"{pdf.name}  {pdf.stat().st_size / 1024:.0f} kB")
