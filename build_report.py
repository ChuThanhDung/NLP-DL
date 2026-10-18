"""Ghép report.pdf cho một lab: README → calculations (.md + bản scan .pdf nếu có) → prediction → notebook → error_analysis → reflection.

Cách dùng:  python build_report.py lab1
Markdown và notebook được render sang HTML rồi in ra PDF bằng Microsoft Edge (headless).
Chạy lại lệnh này sau khi thêm bản scan calculations.pdf vào thư mục lab.
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown
import nbformat
import pymupdf
from nbconvert import HTMLExporter

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]
CSS = """
@page { size: A4; margin: 14mm 13mm 16mm 13mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #222; }
h1 { font-size: 17pt; border-bottom: 2px solid #345; padding-bottom: 4px; }
h2 { font-size: 13.5pt; margin-top: 18px; color: #234; }
h3 { font-size: 11.5pt; }
table { border-collapse: collapse; margin: 8px 0; font-size: 9.2pt; }
th, td { border: 1px solid #bbb; padding: 3px 6px; vertical-align: top; }
th { background: #eef2f6; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f3f3f3; padding: 0 2px; }
pre { background: #f6f6f6; padding: 6px; white-space: pre-wrap; word-break: break-word; font-size: 8.8pt; }
hr { border: 0; border-top: 1px solid #ccc; }
"""
NB_CSS = """
@page { size: A4; margin: 12mm 10mm 14mm 10mm; }
body { font-size: 9pt !important; }
.jp-Notebook, .jp-Cell, .jp-OutputArea-output { max-width: 100% !important; }
pre, .jp-RenderedText pre { white-space: pre-wrap !important; word-break: break-word !important; font-size: 7.8pt !important; }
.jp-RenderedHTMLCommon table { font-size: 7.6pt !important; }
.jp-InputPrompt, .jp-OutputPrompt { min-width: 0 !important; width: 0 !important; overflow: hidden; padding: 0 !important; }
img { max-width: 100% !important; height: auto !important; }
.jp-Cell { break-inside: auto; }
"""
SECTIONS = [  # (file, tiêu đề bookmark)
    ("README.md", "README"),
    ("calculations.md", "Calculations"),
    ("calculations*.pdf", "Calculations (bài làm viết tay)"),
    ("prediction.md", "Prediction"),
    ("*.ipynb", "Experiments"),
    ("error_analysis.md", "Error analysis"),
    ("reflection.md", "Reflection"),
]


def edge_print(html_path: Path, pdf_path: Path):
    edge = next(p for p in EDGE_CANDIDATES if os.path.exists(p))
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={profile}",
                        f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
                       check=True, capture_output=True, timeout=300)
    if not pdf_path.exists():
        raise RuntimeError(f"In PDF thất bại: {html_path}")


def md_to_html(md_path: Path) -> str:
    body = markdown.markdown(md_path.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "sane_lists"])
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"


def nb_to_html(nb_path: Path) -> str:
    nb = nbformat.read(nb_path, as_version=4)
    html, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
    return html.replace("</head>", f"<style>{NB_CSS}</style></head>", 1)


def build(lab_dir: Path):
    readme = (lab_dir / "README.md").read_text(encoding="utf-8")
    title = re.search(r"^#\s+(.+)$", readme, flags=re.M).group(1).strip()
    out = pymupdf.open()
    toc = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for pattern, label in SECTIONS:
            files = sorted(lab_dir.glob(pattern))
            if not files:
                continue
            src = files[0]
            if src.suffix == ".pdf":
                part = pymupdf.open(src)
            else:
                html = nb_to_html(src) if src.suffix == ".ipynb" else md_to_html(src)
                if src.suffix == ".ipynb":
                    label = f"{label} ({src.name})"
                p = tmp / f"{src.stem}.pdf"
                # ghi html cạnh file nguồn để ảnh tham chiếu tương đối vẫn hiển thị
                h_local = lab_dir / f"__tmp_{src.stem}.html"
                h_local.write_text(html, encoding="utf-8")
                try:
                    edge_print(h_local, p)
                finally:
                    h_local.unlink(missing_ok=True)
                part = pymupdf.open("pdf", p.read_bytes())
            toc.append([1, label, out.page_count + 1])
            out.insert_pdf(part)
            print(f"  + {src.name:28s} {part.page_count:3d} trang")
    n = out.page_count
    for i, page in enumerate(out):
        r = page.rect
        page.insert_htmlbox(pymupdf.Rect(30, r.height - 26, r.width - 30, r.height - 8),
                            f"<div style='font-family:sans-serif;font-size:7.5pt;color:#777;text-align:right'>{title} — Trang {i + 1}/{n}</div>")
    out.set_toc(toc)
    out.set_metadata({"title": f"{title} report"})
    dst = lab_dir / "report.pdf"
    out.save(dst, garbage=3, deflate=True)
    print(f"→ {dst} ({n} trang)")


if __name__ == "__main__":
    build(Path(sys.argv[1]))
