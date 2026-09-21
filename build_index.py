"""
build_index.py

Baseline retrieval index for the NEPRA corpus.

    python build_index.py "D:\\...\\knowledge_Base"

EVERY CHOICE HERE IS DELIBERATELY THE DUMBEST OPTION AVAILABLE.

    Fixed-size character chunks. No sentence awareness. No table handling.
    No header propagation. No overlap tuning. No reranking.

That is the point. This is the baseline. If you start clever, you will
never know which of your clever ideas actually mattered. Every improvement
from here gets measured against this number.

The one thing this does carefully is PROVENANCE: every chunk remembers
which document and which page it came from. Without that you cannot score
retrieval separately from generation, and that separation is the whole
project.

Outputs (written next to your PDFs, in an  index/  folder):
    chunks.json   — the text chunks plus their source doc and page
    vectors.npy   — the embedding matrix, one row per chunk
    index_meta.json — the config used, so this run is reproducible
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import pdfplumber
except ImportError:
    sys.exit("Run:  pip install pdfplumber")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Run:  pip install sentence-transformers")


# ---------------------------------------------------------------- config
# These are the knobs you will turn later. Record them, change ONE at a
# time, and measure. Changing three at once tells you nothing.
CONFIG = {
    "chunk_size": 800,          # characters, not tokens — crude on purpose
    "chunk_overlap": 150,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "extractor": "pdfplumber",
    "strategy": "fixed_char_per_page_v1",
}


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...]. Page numbers are 1-based."""
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                print(f"    ! page {i} extraction failed: {e}")
                text = ""
            pages.append((i, text))
    return pages


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """
    Naive fixed-width chunking with overlap.

    This will cut tables in half, orphan column headers from their values,
    and split row labels across chunks. Those are exactly the failure modes
    in your eval set. Do not fix it yet — measure it first.
    """
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start:start + size])
        start += step
    return chunks


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python build_index.py "D:\\path\\to\\knowledge_Base"')

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        sys.exit(f"Not a folder: {folder}")

    # Windows filesystems are case-insensitive, so globbing both *.pdf and
    # *.PDF returns every file twice. Deduplicate by resolved path.
    pdfs = sorted({p.resolve() for p in folder.glob("*.pdf")} |
                  {p.resolve() for p in folder.glob("*.PDF")})
    if not pdfs:
        sys.exit(f"No PDFs found in {folder}")

    print(f"\nBuilding index from {len(pdfs)} documents\n")

    records = []
    for pdf in pdfs:
        print(f"  {pdf.name}")
        pages = extract_pages(pdf)
        doc_chunks = 0
        for page_no, page_text in pages:
            for piece in chunk_text(page_text,
                                    CONFIG["chunk_size"],
                                    CONFIG["chunk_overlap"]):
                records.append({
                    "chunk_id": len(records),
                    "source_doc": pdf.stem,
                    "source_page": page_no,
                    "text": piece,
                })
                doc_chunks += 1
        print(f"      {len(pages)} pages -> {doc_chunks} chunks")

    if not records:
        sys.exit("No text extracted. Check your PDFs.")

    print(f"\nTotal chunks: {len(records)}")
    print(f"Loading embedding model: {CONFIG['embedding_model']}")
    print("(first run downloads ~90MB, then works offline)\n")

    model = SentenceTransformer(CONFIG["embedding_model"])
    vectors = model.encode(
        [r["text"] for r in records],
        show_progress_bar=True,
        normalize_embeddings=True,   # lets cosine similarity be a dot product
        batch_size=32,
    ).astype("float32")

    out_dir = folder.parent / "index"
    out_dir.mkdir(exist_ok=True)

    np.save(out_dir / "vectors.npy", vectors)
    (out_dir / "chunks.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "index_meta.json").write_text(
        json.dumps({
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "config": CONFIG,
            "documents": [p.stem for p in pdfs],
            "chunk_count": len(records),
            "vector_dim": int(vectors.shape[1]),
        }, indent=2), encoding="utf-8")

    print(f"\nWrote index to: {out_dir}")
    print(f"  {len(records)} chunks, {vectors.shape[1]}-dim vectors")
    print("\nThat is your vector store. A matrix of numbers and a list of")
    print("text. Chroma, Pinecone and the rest add persistence, filtering")
    print("and scale on top of exactly this. At 5 documents you do not need")
    print("them, and building it yourself means you know what they do.")
    print("\nNext:  python run_eval.py <eval_set_v1.json> <index folder>")


if __name__ == "__main__":
    main()
