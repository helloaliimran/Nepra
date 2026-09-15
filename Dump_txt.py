"""
dump_text.py

Step 2: see what extraction actually does to your documents, especially tables.

Usage:
    python dump_text.py "D:\\path\\to\\SRO 1468 XWDISCOs 4th QTR FY 2024-25 Dated 07-08-2025.pdf"

Writes a .txt file next to the PDF with clear page markers, then prints a
short report on which pages look like they contain tables.

What to look for when you read the output:
    Find the page with the DISCO table (FESCO, GEPCO, HESCO, LESCO, MEPCO...).
    Ask yourself: if you were handed ONLY this text, could you say with
    certainty what MEPCO's amount was? Or have the columns collapsed into
    an unreadable row of numbers?

    That question is the whole project.
"""

import re
import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    sys.exit("Run:  pip install pypdf")

DISCOS = ["FESCO", "GEPCO", "HESCO", "IESCO", "LESCO",
          "MEPCO", "PESCO", "QESCO", "SEPCO", "TESCO"]


def looks_like_table(text: str) -> bool:
    """Rough heuristic: lots of numbers, or several DISCO names on one page."""
    disco_hits = sum(1 for d in DISCOS if d in text.upper())
    if disco_hits >= 3:
        return True
    # Lines that are mostly numbers and separators
    numeric_lines = 0
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) < 5:
            continue
        digits = sum(c.isdigit() for c in stripped)
        if digits / len(stripped) > 0.4:
            numeric_lines += 1
    return numeric_lines >= 4


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python dump_text.py "D:\\Learning code examples\\RAG Learning Project\\Nepra\\knowledge_Base\\SRO 1468 XWDISCOs 4th QTR FY 2024-25 Dated 07-08-2025.pdf"')

    pdf_path = Path(sys.argv[1])
    if not pdf_path.is_file():
        sys.exit(f"Not a file: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    out_path = pdf_path.with_suffix(".extracted.txt")

    table_pages = []
    chunks = []

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            text = f"(extraction failed: {e})"

        chunks.append(f"\n{'=' * 80}\nPAGE {i}\n{'=' * 80}\n{text}")

        if looks_like_table(text):
            table_pages.append(i)

    out_path.write_text("".join(chunks), encoding="utf-8")

    print(f"\nWrote: {out_path}")
    print(f"Pages: {len(reader.pages)}")

    if table_pages:
        print(f"Pages that probably contain tables: {table_pages}")
        print("\n>>> Open the .txt file and go to those pages FIRST.")
        print(">>> That is where your retrieval will fail.")
    else:
        print("No obvious tables detected. Read the whole file anyway.")

    # Show the first suspected table page inline so you see it immediately
    if table_pages:
        first = table_pages[0]
        page_text = reader.pages[first - 1].extract_text() or ""
        print(f"\n{'=' * 80}")
        print(f"PREVIEW — PAGE {first}")
        print("=" * 80)
        print(page_text[:2000])


if __name__ == "__main__":
    main()
