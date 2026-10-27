import argparse
import dataclasses
import json
import sys
import unicodedata
from datetime import date
from pathlib import Path

from healthrag.config import K, get_settings
from healthrag.conversation import (
    Turn,
    answer_stream,
    check_citations,
    finish_turn,
    log_turn,
    prepare_turn,
)
from healthrag.llm import LLMError
from healthrag.retrieval import Chunk, Retriever, open_retriever

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS = EVAL_DIR / "questions.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
RETRIEVAL_STRATEGIES = ("raw", "joined", "gold")
MIN_OVERALL = 0.8
MIN_FOLLOWUP_HITS = 4


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    for dash in ("–", "—", "‐", "‑"):
        text = text.replace(dash, "-")
    return " ".join(text.split())


def is_hit(chunks: list[Chunk], expect: list[dict]) -> bool:
    return any(
        chunk.source == item["source"] and normalize(item["phrase"]) in normalize(chunk.text)
        for chunk in chunks
        for item in expect
    )


def load_cases(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def strategy_query(case: dict, strategy: str) -> str:
    if case["type"] != "followup" or strategy == "raw":
        return case["question"]
    if strategy == "joined":
        return f"{case['history'][-1]['question']} {case['question']}"
    if strategy == "gold":
        return case["gold_query"]
    raise ValueError(strategy)


def row(case: dict, strategy: str, query: str, chunks: list[Chunk], **extra) -> dict:
    return {
        "case_id": case["id"],
        "type": case["type"],
        "strategy": strategy,
        "query": query,
        "hit": None if case["type"] == "offtopic" else is_hit(chunks, case["expect"]),
        "chunk_ids": [chunk.id for chunk in chunks],
        "distances": [round(chunk.distance, 4) for chunk in chunks],
        **extra,
    }


def run_retrieval(cases: list[dict], retriever: Retriever) -> list[dict]:
    rows = []
    for case in cases:
        for strategy in RETRIEVAL_STRATEGIES:
            query = strategy_query(case, strategy)
            chunks = retriever.search(query)
            rows.append(
                row(
                    case,
                    strategy,
                    query,
                    chunks,
                    relevant_hit=None
                    if case["type"] == "offtopic"
                    else is_hit(retriever.relevant(chunks), case["expect"]),
                    covered=bool(retriever.relevant(chunks)),
                )
            )
    return rows


def history_turns(case: dict) -> list[Turn]:
    return [
        Turn(question=item["question"], source_filter=None, answer=item["answer"])
        for item in case["history"]
    ]


def run_e2e(cases: list[dict], retriever: Retriever, label: str) -> list[dict]:
    rows = []
    for case in cases:
        history = history_turns(case)
        prepared = prepare_turn(
            case["question"],
            history,
            None,
            retriever,
            session=f"eval-{label}",
            turn_no=len(history) + 1,
        )
        result = None
        error = None
        if prepared.covered:
            result = answer_stream(prepared, history)
            try:
                for _ in result:
                    pass
            except LLMError as e:
                error = e.kind
            finally:
                result.close()
        outcome = "not_covered" if result is None else "answered"
        log_turn(prepared, outcome if error is None else f"error_{error}")
        turn = finish_turn(prepared, result)
        citations_ok, citations = check_citations(turn.answer, turn.chunks)
        records = list(prepared.research.records)
        if result is not None:
            records.append(result.record)
        rows.append(
            row(
                case,
                "agent",
                " | ".join(prepared.research.queries),
                prepared.chunks,
                searches=len(prepared.research.queries),
                covered=prepared.covered,
                fallback=prepared.research.fallback,
                input_tokens=sum(r.input_tokens or 0 for r in records),
                output_tokens=sum(r.output_tokens or 0 for r in records),
                cost_usd=round(turn.cost_usd, 6),
                ttft_ms=result.ttft_ms if result is not None else None,
                citations_ok=citations_ok and error is None,
                citations=citations,
                error=error,
                answer=turn.answer,
            )
        )
    return rows


def hit_rate(rows: list[dict], types: set[str], key: str = "hit") -> tuple[int, int]:
    scored = [r for r in rows if r["type"] in types and r.get(key) is not None]
    return sum(1 for r in scored if r[key]), len(scored)


def summarize(rows: list[dict], mode: str) -> dict:
    summary: dict = {}
    strategies = RETRIEVAL_STRATEGIES if mode == "retrieval" else ("agent",)
    for strategy in strategies:
        subset = [r for r in rows if r["strategy"] == strategy]
        first_hits, first_total = hit_rate(subset, {"first"})
        follow_hits, follow_total = hit_rate(subset, {"followup"})
        all_hits, all_total = hit_rate(subset, {"first", "followup"})
        summary[strategy] = {
            "first": f"{first_hits}/{first_total}",
            "followup": f"{follow_hits}/{follow_total}",
            "overall": round(all_hits / all_total, 3) if all_total else None,
        }
        if mode == "retrieval":
            kept, total = hit_rate(subset, {"first", "followup"}, key="relevant_hit")
            summary[strategy]["overall_after_cutoff"] = round(kept / total, 3) if total else None
    offtopic_rows = [r for r in rows if r["type"] == "offtopic"]
    refused = sum(1 for r in offtopic_rows if not r["covered"])
    summary["offtopic_not_covered"] = f"{refused}/{len(offtopic_rows)}"
    ontopic = [r["distances"][0] for r in rows if r["type"] != "offtopic" and r["distances"]]
    if ontopic:
        summary["ontopic_top1_distance_max"] = max(ontopic)
    if mode == "e2e":
        answered = [r for r in rows if r["type"] != "offtopic"]
        summary["citations_ok"] = f"{sum(1 for r in answered if r['citations_ok'])}/{len(answered)}"
        summary["mean_searches"] = round(sum(r["searches"] for r in answered) / len(answered), 2)
        summary["mean_cost_usd"] = round(sum(r["cost_usd"] for r in answered) / len(answered), 6)
        summary["max_cost_usd"] = max(r["cost_usd"] for r in answered)
        ttfts = [r["ttft_ms"] for r in answered if r["ttft_ms"] is not None]
        summary["mean_ttft_ms"] = round(sum(ttfts) / len(ttfts)) if ttfts else None
    return summary


def gate_failures(rows: list[dict], mode: str) -> list[str]:
    failures = []
    offtopic_covered = [r["case_id"] for r in rows if r["type"] == "offtopic" and r["covered"]]
    if offtopic_covered:
        failures.append(f"off-topic questions got passages: {sorted(set(offtopic_covered))}")
    if mode == "retrieval":
        first_misses = [
            r["case_id"] for r in rows if r["type"] == "first" and not r["relevant_hit"]
        ]
        if first_misses:
            failures.append(f"first-turn misses after cutoff: {sorted(set(first_misses))}")
        gold = [r for r in rows if r["strategy"] == "gold"]
        hits, total = hit_rate(gold, {"first", "followup"}, key="relevant_hit")
        if total and hits / total < MIN_OVERALL:
            failures.append(f"gold strategy overall {hits}/{total} is below {MIN_OVERALL}")
    else:
        hits, _ = hit_rate(rows, {"followup"})
        if hits < MIN_FOLLOWUP_HITS:
            failures.append(f"agent strategy hit {hits} follow-ups; {MIN_FOLLOWUP_HITS} needed")
        bad = [r["case_id"] for r in rows if r["type"] != "offtopic" and not r["citations_ok"]]
        if bad:
            failures.append(f"citation check failed: {bad}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score retrieval (and answers) on questions.jsonl."
    )
    parser.add_argument("--mode", choices=("retrieval", "e2e"), required=True)
    parser.add_argument("--label", required=True, help="Name for the results file, e.g. tuned.")
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--no-gate", action="store_true", help="Report only; never exit 1.")
    parser.add_argument("--no-save", action="store_true", help="Don't write a results file.")
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = get_settings()
    if args.index_dir:
        settings = dataclasses.replace(settings, index_dir=args.index_dir)
    retriever = open_retriever(settings)
    cases = load_cases()
    rows = (
        run_retrieval(cases, retriever)
        if args.mode == "retrieval"
        else run_e2e(cases, retriever, args.label)
    )
    summary = summarize(rows, args.mode)
    failures = [] if args.no_gate else gate_failures(rows, args.mode)

    print(f"\n{args.mode} eval · {args.label} · {retriever.manifest['embedding_model']} · k={K}")
    for r in rows:
        if r["hit"] is None:
            mark = "OUT " if not r["covered"] else "IN? "
        else:
            mark = "HIT " if r["hit"] else "MISS"
        print(f"  {mark:<4} {r['strategy']:<7} {r['case_id']:<28} {r['query'][:70]}")
    print(json.dumps(summary, indent=2))

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / f"{date.today().isoformat()}-{args.mode}-{args.label}.json"
        if out.exists():
            print(f"error: {out} already exists; choose another --label", file=sys.stderr)
            return 1
        payload = {
            "mode": args.mode,
            "label": args.label,
            "manifest": retriever.manifest,
            "summary": summary,
            "rows": rows,
        }
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"wrote {out}")

    for failure in failures:
        print(f"GATE FAILED: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
