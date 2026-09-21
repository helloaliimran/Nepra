"""
run_eval.py

Scores RETRIEVAL ONLY against your frozen eval set. No LLM, no API key,
no cost.

    python run_eval.py "D:\\...\\eval_set_v1.json" "D:\\...\\index"

WHY RETRIEVAL FIRST, WITH NO GENERATION

    If you measure the whole pipeline at once and get 40%, you learn
    nothing about where the 60% went. Retrieval failure and reasoning
    failure need completely different fixes.

    So: measure retrieval alone first. It sets a hard ceiling. If the
    correct page is never retrieved, no model on earth can answer
    correctly from it.

WHAT IS BEING SCORED

    retrieval_hit @ k — did any of the top-k chunks come from the
    document and page recorded as ground truth?

    Refusal questions are excluded from this score. They have no correct
    page by definition, and they only become meaningful once generation
    is added.

Output:
    results_<timestamp>.json  next to your eval set
"""

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("Run:  pip install numpy")


try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Run:  pip install sentence-transformers")


TOP_K_VALUES = [1, 3, 5, 10]


def normalize(s: str) -> list[str]:
    """Lowercase alphanumeric tokens, so 'CPPA-G' and 'CPPA G' match."""
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def docs_match(eval_doc: str, indexed_doc: str) -> bool:
    """
    Eval set names documents loosely ('SRO 1468 XWDISCOs 4th QTR FY 2024-25')
    while filenames carry extra suffixes ('... Dated 07-08-2025 12380-83').
    Match on token overlap rather than exact string equality.
    """
    a, b = set(normalize(eval_doc)), set(normalize(indexed_doc))
    if not a:
        return False
    return len(a & b) / len(a) >= 0.6


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def main():
    if len(sys.argv) < 3:
        sys.exit('Usage: python run_eval.py "<eval_set.json>" "<index folder>"')

    eval_path = Path(sys.argv[1])
    index_dir = Path(sys.argv[2])

    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    chunks = json.loads(
        (index_dir / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(index_dir / "vectors.npy")
    index_meta = json.loads(
        (index_dir / "index_meta.json").read_text(encoding="utf-8"))

    questions = eval_data["questions"]
    eval_hash = file_hash(eval_path)

    print(f"\nEval set : {eval_path.name}  (hash {eval_hash})")
    print(f"Questions: {len(questions)}")
    print(
        f"Index    : {len(chunks)} chunks, {index_meta['config']['strategy']}\n")

    model = SentenceTransformer(index_meta["config"]["embedding_model"])

    max_k = max(TOP_K_VALUES)
    per_question = []

    for q in questions:
        is_refusal = q.get("failure_mode") == "refusal" or q.get(
            "source_page") is None

        q_vec = model.encode(
            [q["question"]], normalize_embeddings=True).astype("float32")
        scores = (vectors @ q_vec.T).ravel()
        top_idx = np.argsort(-scores)[:max_k]

        retrieved = [{
            "rank": rank,
            "chunk_id": int(chunks[i]["chunk_id"]),
            "source_doc": chunks[i]["source_doc"],
            "source_page": chunks[i]["source_page"],
            "score": round(float(scores[i]), 4),
            "preview": chunks[i]["text"][:160].replace("\n", " "),
        } for rank, i in enumerate(top_idx, start=1)]

        hit_at = {}
        if not is_refusal:
            for k in TOP_K_VALUES:
                hit_at[f"hit@{k}"] = any(
                    r["source_page"] == q["source_page"]
                    and docs_match(q["source_doc"], r["source_doc"])
                    for r in retrieved[:k]
                )

        per_question.append({
            "id": q["id"],
            "question": q["question"],
            "failure_mode": q.get("failure_mode"),
            "expected_doc": q.get("source_doc"),
            "expected_page": q.get("source_page"),
            "scored": not is_refusal,
            **hit_at,
            "retrieved": retrieved,
        })

    scored = [r for r in per_question if r["scored"]]

    # ------------------------------------------------------------ report
    print("=" * 78)
    print("RETRIEVAL ACCURACY")
    print("=" * 78)
    summary = {}
    for k in TOP_K_VALUES:
        hits = sum(1 for r in scored if r[f"hit@{k}"])
        pct = hits / len(scored) if scored else 0
        summary[f"hit@{k}"] = round(pct, 3)
        print(f"  hit@{k:<3} {hits}/{len(scored)}   {pct:.0%}")

    print(f"\n  ({len(per_question) - len(scored)} refusal questions excluded — "
          f"they need generation to score)")

    print("\n" + "=" * 78)
    print("BY FAILURE MODE  (hit@5)")
    print("=" * 78)
    modes = {}
    for r in scored:
        modes.setdefault(r["failure_mode"] or "unlabelled",
                         []).append(r["hit@5"])
    for mode, hits in sorted(modes.items()):
        print(f"  {mode:<28} {sum(hits)}/{len(hits)}")

    print("\n" + "=" * 78)
    print("PER QUESTION  (hit@5)")
    print("=" * 78)
    for r in per_question:
        if not r["scored"]:
            print(f"  {r['id']}  SKIP  (refusal)   {r['question'][:46]}")
            continue
        mark = "PASS" if r["hit@5"] else "FAIL"
        print(f"  {r['id']}  {mark}            {r['question'][:46]}")
        if not r["hit@5"]:
            print(
                f"        wanted: {r['expected_doc'][:40]} p.{r['expected_page']}")
            got = r["retrieved"][0]
            print(f"        top-1 : {got['source_doc'][:40]} p.{got['source_page']} "
                  f"(score {got['score']})")

    # ------------------------------------------------------------ persist
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = eval_path.parent / f"results_{stamp}.json"
    out.write_text(json.dumps({
        "run_id": f"baseline-{stamp}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "eval_set": eval_path.name,
        "eval_hash": eval_hash,
        "index_config": index_meta["config"],
        "chunk_count": len(chunks),
        "retrieval_summary": summary,
        "per_question": per_question,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nWrote: {out.name}")
    print("\nThis file records the eval hash AND the index config. A number")
    print("without the settings that produced it is not a result, it is a")
    print("rumour. Keep every run.")


if __name__ == "__main__":
    main()
