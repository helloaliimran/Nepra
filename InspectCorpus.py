"""
inspect_corpus.py

Step 1 of the NEPRA retrieval project: find out what we are actually dealing with
before writing a single line of RAG code.

Usage:
    pip install pypdf
    python inspect_corpus.py "D:\\path\\to\\your\\nepra\\folder"

What it tells you:
    - how many pages each PDF has
    - how many characters of text can be extracted per page
    - which pages are probably scanned images (almost no extractable text)
    - a verdict per file: TEXT / SCANNED / MIXED

Why it matters:
    If the PDFs are scanned images, no chunking strategy in the world helps you.
    You need OCR first, and OCR quality becomes the ceiling on everything else.
"""

import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    sys.exit("Run:  pip install pypdf")

# A page with fewer extractable characters than this is almost certainly an image.
SCANNED_PAGE_THRESHOLD = 100


def inspect_pdf(path: Path) -> dict:
    reader = PdfReader(str(path))
    per_page_chars = []

    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        per_page_chars.append(len(text.strip()))

    total_pages = len(per_page_chars)
    scanned_pages = sum(1 for c in per_page_chars if c <
                        SCANNED_PAGE_THRESHOLD)

    if total_pages == 0:
        verdict = "EMPTY"
    elif scanned_pages == total_pages:
        verdict = "SCANNED"
    elif scanned_pages == 0:
        verdict = "TEXT"
    else:
        verdict = "MIXED"

    return {
        "name": path.name,
        "pages": total_pages,
        "scanned_pages": scanned_pages,
        "avg_chars": round(sum(per_page_chars) / total_pages) if total_pages else 0,
        "verdict": verdict,
        "per_page_chars": per_page_chars,
    }


def first_sample(path: Path, chars: int = 400) -> str:
    """Grab the first readable text we can find, so you can eyeball OCR quality."""
    reader = PdfReader(str(path))
    for page in reader.pages:
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            continue
        if len(text) > SCANNED_PAGE_THRESHOLD:
            return text[:chars]
    return "(no extractable text found)"


def main():
    if len(sys.argv) < 2:
        sys.exit(
            'Usage: python inspect_corpus.py "D:\\Learning code examples\\RAG Learning Project\\Nepra\\knowledge_Base"')

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        sys.exit(f"Not a folder: {folder}")

    pdfs = sorted(folder.glob("*.pdf")) + sorted(folder.glob("*.PDF"))
    if not pdfs:
        sys.exit(f"No PDFs found in {folder}")

    print(f"\nInspecting {len(pdfs)} PDFs in {folder}\n")
    print(f"{'FILE':<55} {'PAGES':>6} {'SCANNED':>8} {'AVG CHARS':>10}  VERDICT")
    print("-" * 95)

    results = []
    for pdf in pdfs:
        try:
            r = inspect_pdf(pdf)
        except Exception as e:
            print(f"{pdf.name[:54]:<55} {'ERROR':>6}  {e}")
            continue
        results.append(r)
        print(
            f"{r['name'][:54]:<55} {r['pages']:>6} {r['scanned_pages']:>8} "
            f"{r['avg_chars']:>10}  {r['verdict']}"
        )

    # Summary
    print("\n" + "=" * 95)
    total_pages = sum(r["pages"] for r in results)
    total_scanned = sum(r["scanned_pages"] for r in results)
    print(f"Total pages: {total_pages}")
    if total_pages:
        pct = round(100 * total_scanned / total_pages)
        print(f"Pages with no usable text: {total_scanned} ({pct}%)")
        if pct > 50:
            print(
                "\n>>> Your corpus is mostly scanned. OCR is your first problem, not retrieval.")
        elif pct > 10:
            print("\n>>> Mixed corpus. You will need OCR as a fallback for some pages.")
        else:
            print("\n>>> Text layer is present. You can extract directly. Good news.")

    # Show a sample so you can judge OCR noise with your own eyes
    if results:
        sample_file = folder / results[0]["name"]
        print("\n" + "=" * 95)
        print(f"SAMPLE TEXT from: {sample_file.name}\n")
        print(first_sample(sample_file))
        print("\n(Read this closely. Garbled words here become garbled retrieval later.)")


if __name__ == "__main__":
    main()
