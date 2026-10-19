"""Render the Markdown reports in docs/ to styled PDFs.

The Markdown files are the source of truth (they render on GitHub); the PDFs are
generated artifacts for formal submission. Run from the project root:

    pip install -r requirements-docs.txt
    python scripts/build_docs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

REPORTS = ["AI_Opportunity_Report", "Technical_Approach_Report"]

CSS = """
@page { size: A4; margin: 2cm 2.2cm; }
body { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 20pt; color: #16233a; border-bottom: 2px solid #5B8DEF;
     padding-bottom: 6px; margin-top: 0; }
h2 { font-size: 14pt; color: #16233a; margin-top: 22px;
     border-bottom: 1px solid #d8dde6; padding-bottom: 3px; }
h3 { font-size: 11.5pt; color: #2a3850; margin-top: 16px; }
p, li { color: #24292e; }
code { background: #f2f4f8; padding: 1px 4px; border-radius: 3px;
       font-family: 'SF Mono', Menlo, monospace; font-size: 9pt; }
pre { background: #f6f8fa; padding: 10px; border-radius: 5px; overflow-x: auto;
      border: 1px solid #e1e4e8; font-size: 8.5pt; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9pt; }
th { background: #16233a; color: #fff; text-align: left; padding: 6px 9px; }
td { border: 1px solid #d8dde6; padding: 5px 9px; vertical-align: top; }
tr:nth-child(even) td { background: #f7f9fc; }
blockquote { border-left: 3px solid #E8B339; margin: 12px 0; padding: 4px 14px;
             background: #fdf9ef; color: #5c5338; }
a { color: #3a6fd0; word-break: break-word; }
hr { border: none; border-top: 1px solid #d8dde6; margin: 20px 0; }
strong { color: #16233a; }
"""

HTML_SHELL = "<html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"


def build(name: str) -> bool:
    md_path = DOCS / f"{name}.md"
    if not md_path.exists():
        print(f"  skip  {name}.md (not found)")
        return False

    html_body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )
    document = HTML_SHELL.format(css=CSS, body=html_body)
    pdf_path = DOCS / f"{name}.pdf"
    HTML(string=document, base_url=str(DOCS)).write_pdf(str(pdf_path))
    size_kb = pdf_path.stat().st_size / 1024
    print(f"  built {name}.pdf ({size_kb:.0f} KB)")
    return True


def main() -> None:
    print("Building PDF reports from Markdown...")
    built = sum(build(name) for name in REPORTS)
    print(f"Done. {built}/{len(REPORTS)} report(s) built.")
    if built == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
