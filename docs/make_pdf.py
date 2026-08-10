"""Render REPORT.md to REPORT.pdf.

    python docs/make_pdf.py

Markdown is the source of truth and the PDF is built from it, so the two cannot drift.
The route is markdown -> HTML -> headless Chrome, which is the only renderer on hand that
gets CSS tables and page-break control right; the report is mostly tables, so that
matters more than anything else about the toolchain.

Figures are inlined as data URIs rather than linked, so the intermediate HTML is
self-contained and the PDF cannot come out with missing images if it is generated from a
different working directory.

Chrome's own print header and footer carry the file:// URL, so they are switched off and
page numbers are stamped afterwards with reportlab.
"""
from __future__ import annotations

import base64
import io
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mistune
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "REPORT.md"
# Source lives beside this script; the built PDF is a deliverable, so it lands in
# reports/ with the figures rather than next to the machinery that made it.
DESTINATION = HERE.parent / "reports" / "REPORT.pdf"

CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome",
    "chromium",
    "chromium-browser",
)

# Matches the palette the figures are drawn in, so the document and its charts read as
# one artefact rather than as a report with pictures pasted into it.
STYLE = """
@page { size: A4; margin: 20mm 17mm 18mm 17mm; }

:root { --teal: #064A56; --rust: #B3261E; --slate: #607478; --rule: #D9E2E4; }

* { box-sizing: border-box; }

body {
  font-family: "Segoe UI", Inter, -apple-system, Helvetica, Arial, sans-serif;
  font-size: 9.6pt;
  line-height: 1.46;
  color: #1B2426;
  margin: 0;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}

h1 {
  font-size: 21pt;
  color: var(--teal);
  margin: 0 0 4pt;
  letter-spacing: -0.4pt;
}
h1 + p { color: var(--slate); font-size: 9pt; margin-top: 0; }

h2 {
  font-size: 12.5pt;
  color: var(--teal);
  margin: 16pt 0 6pt;
  padding-bottom: 3pt;
  border-bottom: 1.4pt solid var(--teal);
  break-after: avoid;
}

h3 { font-size: 10.5pt; margin: 13pt 0 5pt; break-after: avoid; }

p { margin: 0 0 7pt; }
strong { color: #0B1416; }

hr {
  border: none;
  border-top: 1pt solid var(--rule);
  margin: 14pt 0;
}

/* Tables flow across pages rather than being held whole. Holding them whole leaves a
   third of a page blank whenever the next one does not fit, and this report is mostly
   tables, so that happens repeatedly. The header repeats on the continuation instead,
   and no single row is ever split. */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 7pt 0 10pt;
  font-size: 8.7pt;
  font-variant-numeric: tabular-nums;
  break-inside: auto;
}
thead { display: table-header-group; }
tr { break-inside: avoid; }
thead th {
  text-align: left;
  color: var(--teal);
  font-weight: 700;
  border-bottom: 1.2pt solid var(--teal);
  padding: 4pt 6pt 4pt 0;
}
tbody td {
  border-bottom: 0.6pt solid var(--rule);
  padding: 4pt 6pt 4pt 0;
  vertical-align: top;
}
tbody tr:last-child td { border-bottom: 1pt solid var(--rule); }

code {
  font-family: "Cascadia Mono", Consolas, "SF Mono", monospace;
  font-size: 0.88em;
  background: #F1F5F6;
  /* Barely any horizontal padding: two code spans separated by a full stop otherwise
     render as "fit . predict", with the tint reading as whitespace around the period. */
  padding: 0.5pt 1pt;
  border-radius: 2pt;
}
pre {
  background: #F1F5F6;
  padding: 7pt 9pt;
  border-left: 2.5pt solid var(--teal);
  font-size: 8.4pt;
  overflow-x: auto;
  break-inside: avoid;
}
pre code { background: none; padding: 0; }

a { color: var(--teal); text-decoration: none; border-bottom: 0.5pt solid var(--rule); }

ul, ol { margin: 0 0 8pt; padding-left: 15pt; }
li { margin-bottom: 3pt; }

img {
  width: 100%;
  height: auto;
  margin: 4pt 0 3pt;
  break-inside: avoid;
}

blockquote {
  margin: 8pt 0;
  padding: 5pt 10pt;
  border-left: 2.5pt solid var(--rust);
  color: var(--slate);
}
"""

TEMPLATE = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>{title}</title><style>{style}</style></head>
<body>{body}</body>
</html>
"""


def inline_images(markdown: str, base: Path) -> str:
    """Replace ![alt](figures/x.png) with a data URI, so the HTML stands alone."""

    def replace(match: re.Match) -> str:
        source = (base / match.group(2)).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"{SOURCE.name} references a missing figure: {source}")
        encoded = base64.b64encode(source.read_bytes()).decode("ascii")
        return f"![{match.group(1)}](data:image/png;base64,{encoded})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", replace, markdown)


def stamp_page_numbers(pdf_bytes: bytes) -> bytes:
    """Overlay 'n / N' at the foot of every page."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total = len(reader.pages)
    writer = PdfWriter()

    for number, page in enumerate(reader.pages, start=1):
        overlay = io.BytesIO()
        pen = canvas.Canvas(overlay, pagesize=A4)
        pen.setFont("Helvetica", 7.5)
        pen.setFillColorRGB(0.38, 0.46, 0.47)
        pen.drawCentredString(A4[0] / 2, 26, f"{number} / {total}")
        pen.save()
        overlay.seek(0)
        page.merge_page(PdfReader(overlay).pages[0])
        writer.add_page(page)

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise FileNotFoundError(
        "no Chrome or Edge binary found. Install one, or add its path to CHROME_CANDIDATES."
    )


def main() -> int:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"{SOURCE} not found")

    render = mistune.create_markdown(plugins=["table", "strikethrough", "url"])
    title = re.sub(r"^#\s+", "", SOURCE.read_text(encoding="utf-8").splitlines()[0])
    html = TEMPLATE.format(
        title=title,
        style=STYLE,
        body=render(inline_images(SOURCE.read_text(encoding="utf-8"), HERE)),
    )

    with tempfile.TemporaryDirectory() as workspace:
        staging = Path(workspace)
        page = staging / "report.html"
        page.write_text(html, encoding="utf-8")
        raw = staging / "raw.pdf"

        subprocess.run(
            [
                find_chrome(),
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                "--virtual-time-budget=10000",
                f"--user-data-dir={staging / 'profile'}",
                f"--print-to-pdf={raw}",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
        )
        if not raw.is_file():
            raise RuntimeError("Chrome exited cleanly but wrote no PDF")
        DESTINATION.write_bytes(stamp_page_numbers(raw.read_bytes()))

    pages = len(PdfReader(str(DESTINATION)).pages)
    print(f"wrote {DESTINATION}  ({pages} pages, {DESTINATION.stat().st_size / 1024:,.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
