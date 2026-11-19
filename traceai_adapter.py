#!/usr/bin/env python3
"""Minimal TraceAI -> U/D/A/H adapter (Python, standard library only).

Purpose
-------
Read a trace exported by current TraceAI (OTLP/GenAI JSON or JSONL), reconstruct
per-session conversation turns, and score **each turn exactly once**.

Deduplication rule
------------------
TraceAI exports the full prompt of every request, including the history that was
already scored in earlier turns.  For each session we keep the messages already
seen, compute the longest common prefix with the current ``gen_ai.input.messages``
and score only the residual (new) user messages plus the current assistant output
from ``gen_ai.output.messages``.  History is never scored twice.

Input fields
------------
- session.id                       (span or resource attribute; fallback
                                    gen_ai.conversation.id / traceId)
- gen_ai.input.messages            (JSON string or array; supports role+content
                                    and role+parts[{type:text, content}])
- gen_ai.output.messages           (same shape)
- gen_ai.usage.input_tokens        (fallback gen_ai.usage.prompt_tokens)
- gen_ai.usage.output_tokens       (fallback gen_ai.usage.completion_tokens)

Scores (same definitions as thinkcheck_harmony/core.py)
-------------------------------------------------------
U = mean term consistency: for every key term, mean pairwise cosine similarity
    of the sentences that contain it (char 3-gram vectors here, so no model
    download is required).  Terms appearing once score 1.0.
D = distribution of key terms: std(first-occurrence position / #sentences) * 2.5,
    capped at 1.0; 0.0 when fewer than two terms are present.
A = adversarial pressure: 0.6 * sentence marker density
    + 0.4 * min(1, contradiction_rule_hits * 0.3).
H = 0.4*U + 0.4*D - 0.2*A  (lambdas configurable).

Usage
-----
    python traceai_adapter.py examples/traceai_sample_trace.json \
        --out traceai_turn_scores.csv --show-terms
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_LAMBDA_U = 0.4
DEFAULT_LAMBDA_D = 0.4
DEFAULT_LAMBDA_A = 0.2

# Same Chinese markers as thinkcheck_harmony.config plus English equivalents so
# that synthetic English traces are scorable.
ADVERSARIAL_MARKERS = [
    "但是", "然而", "不过", "尽管", "需要注意的是", "必须指出",
    "与此相反", "另一方面", "存在风险", "有待商榷",
    "but", "however", "although", "though", "on the other hand",
    "that said", "may not", "might not", "cannot be guaranteed", "it depends",
]

# Phrase pairs that cannot both be true in the same turn.  Extend or disable
# with --no-rules.  This mirrors ContradictionDetector(contradiction_rules=...).
DEFAULT_CONTRADICTION_RULES = [
    ("original card", "store credit"),
    ("payment card", "store credit"),
    ("原路退回", "账户余额"),
    ("全额", "手续费"),
    ("free of charge", "service fee"),
]

SENTENCE_SPLIT_RE = re.compile(r"[。！？；!?;\n]+")
CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")

CHINESE_STOP = {
    "我们", "你们", "他们", "这个", "那个", "这些", "那些", "可以", "如果",
    "因为", "所以", "但是", "还是", "就是", "已经", "现在", "应该", "需要",
    "什么", "没有", "自己", "一个", "不是", "不能", "不会",
}
ENGLISH_STOP = {
    "the", "and", "for", "with", "that", "this", "you", "your", "will", "can",
    "have", "has", "are", "was", "were", "from", "not", "but", "all", "any",
    "our", "was", "it", "is", "to", "of", "in", "on", "at", "be", "as", "or",
    "please", "would", "could", "should", "there", "their", "them", "they",
    "been", "being", "into", "than", "then", "when", "what", "which", "who",
}


# ---------------------------------------------------------------------------
# OTLP parsing
# ---------------------------------------------------------------------------

def _attr_value(value: Any) -> Any:
    """Unwrap an OTLP AnyValue ({"stringValue": ...}) or a plain value."""
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        try:
            return int(value["intValue"])
        except (TypeError, ValueError):
            return value["intValue"]
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [_attr_value(v) for v in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {
            kv["key"]: _attr_value(kv.get("value"))
            for kv in value["kvlistValue"].get("values", [])
        }
    return value


def attrs_to_dict(attrs: Any) -> Dict[str, Any]:
    if isinstance(attrs, dict):
        return {k: _attr_value(v) for k, v in attrs.items()}
    result: Dict[str, Any] = {}
    for item in attrs or []:
        if isinstance(item, dict) and "key" in item:
            result[item["key"]] = _attr_value(item.get("value"))
    return result


def _resource_spans(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        # A bare list of spans.
        yield {"resource": {"attributes": []},
               "scopeSpans": [{"spans": payload}]}
        return
    if not isinstance(payload, dict):
        return
    if "resourceSpans" in payload:
        for rs in payload["resourceSpans"] or []:
            yield rs
    elif "scopeSpans" in payload:
        yield payload
    elif "spans" in payload:
        yield {"resource": {"attributes": []},
               "scopeSpans": [{"spans": payload["spans"]}]}
    elif "data" in payload and isinstance(payload["data"], dict):
        yield from _resource_spans(payload["data"])


def load_spans(path: str) -> List[Dict[str, Any]]:
    """Load spans from an OTLP JSON export or a JSONL export."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        payloads = [json.loads(text)]
    except json.JSONDecodeError:
        payloads = [json.loads(line) for line in text.splitlines() if line.strip()]

    spans: List[Dict[str, Any]] = []
    for payload in payloads:
        for resource_span in _resource_spans(payload):
            resource_attrs = attrs_to_dict(
                (resource_span.get("resource") or {}).get("attributes")
            )
            scopes = (
                resource_span.get("scopeSpans")
                or resource_span.get("instrumentationLibrarySpans")
                or []
            )
            for scope in scopes:
                for span in scope.get("spans", []) or []:
                    spans.append({
                        "span": span,
                        "resource_attrs": resource_attrs,
                        "order": len(spans),
                    })
    return spans


# ---------------------------------------------------------------------------
# GenAI message parsing
# ---------------------------------------------------------------------------

def _message_text(message: Dict[str, Any]) -> str:
    parts = message.get("parts")
    if isinstance(parts, list):
        chunks = []
        for part in parts:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                value = part.get("content")
                if value is None:
                    value = part.get("text")
                if isinstance(value, str):
                    chunks.append(value)
        if chunks:
            return "\n".join(chunks)
    for key in ("content", "text", "body"):
        value = message.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            chunks = []
            for item in value:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            if chunks:
                return "\n".join(chunks)
    return ""


def _normalize_message(message: Any, fallback_role: str) -> Dict[str, str]:
    if isinstance(message, str):
        return {"role": fallback_role, "content": message}
    if not isinstance(message, dict):
        return {"role": fallback_role, "content": str(message)}
    role = message.get("role") or message.get("author") or fallback_role
    return {"role": str(role).lower(), "content": _message_text(message)}


def parse_messages(value: Any, fallback_role: str = "user") -> List[Dict[str, str]]:
    """Parse gen_ai.*.messages: JSON string, list, or single message object."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in "[{":
            try:
                return parse_messages(json.loads(text), fallback_role)
            except json.JSONDecodeError:
                pass
        return [{"role": fallback_role, "content": text}]
    if isinstance(value, dict):
        for key in ("messages", "items", "value"):
            if key in value:
                return parse_messages(value[key], fallback_role)
        if "content" in value or "parts" in value or "text" in value:
            return [_normalize_message(value, fallback_role)]
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, dict):
                result.append(_normalize_message(item, fallback_role))
            elif isinstance(item, str):
                result.append({"role": fallback_role, "content": item})
        return [m for m in result if m["content"].strip()]
    return []


def _first_attr(attrs: Dict[str, Any], names: List[str]) -> Any:
    for name in names:
        if name in attrs and attrs[name] not in (None, ""):
            return attrs[name]
    return None


def collect_turns(spans: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group scoreable chat spans by session.id."""
    sessions: Dict[str, List[Dict[str, Any]]] = {}
    for index, entry in enumerate(spans):
        span = entry["span"]
        attrs = attrs_to_dict(span.get("attributes"))
        resource_attrs = entry["resource_attrs"]

        input_raw = _first_attr(attrs, ["gen_ai.input.messages", "gen_ai.prompt"])
        output_raw = _first_attr(attrs, ["gen_ai.output.messages", "gen_ai.completion"])
        if input_raw is None and output_raw is None:
            continue

        session_id = _first_attr(attrs, ["session.id", "gen_ai.conversation.id"])
        if session_id is None:
            session_id = _first_attr(
                resource_attrs, ["session.id", "gen_ai.conversation.id"]
            )
        if session_id is None:
            session_id = span.get("traceId") or f"session-{index}"

        start = span.get("startTimeUnixNano") or span.get("timeUnixNano") or 0
        try:
            start_ns = int(start)
        except (TypeError, ValueError):
            start_ns = 0

        input_messages = parse_messages(input_raw, "user")
        output_messages = parse_messages(output_raw, "assistant")

        sessions.setdefault(str(session_id), []).append({
            "order": entry["order"],
            "start_ns": start_ns,
            "span_id": span.get("spanId", ""),
            "input_messages": input_messages,
            "output_messages": output_messages,
            "input_tokens": _first_attr(
                attrs, ["gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens"]
            ),
            "output_tokens": _first_attr(
                attrs, ["gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens"]
            ),
        })

    for turns in sessions.values():
        turns.sort(key=lambda t: (t["start_ns"], t["order"]))
    return sessions


# ---------------------------------------------------------------------------
# Deduplication: score each turn once
# ---------------------------------------------------------------------------

def _message_key(message: Dict[str, str]) -> Tuple[str, str]:
    content = re.sub(r"\s+", " ", message.get("content", "")).strip()
    return message.get("role", ""), content


def strip_scored_history(
    messages: List[Dict[str, str]],
    history: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], int]:
    """Return (new messages, number of history messages dropped).

    Fast path: longest common prefix (cumulative traces).  Fallback path: drop
    any residual message that already appeared verbatim in history (reordered
    or re-serialized traces).
    """
    prefix = 0
    while (
        prefix < len(messages)
        and prefix < len(history)
        and _message_key(messages[prefix]) == _message_key(history[prefix])
    ):
        prefix += 1
    residual = messages[prefix:]
    dropped = prefix

    if residual and prefix == 0:
        seen = {_message_key(m) for m in history}
        kept = []
        for message in residual:
            if _message_key(message) in seen:
                dropped += 1
            else:
                kept.append(message)
        residual = kept
    return residual, dropped


# ---------------------------------------------------------------------------
# U / D / A / H
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def extract_terms(text: str, top_n: int = 10) -> List[str]:
    """Key terms = English words + repeated Chinese n-grams (2..4)."""
    candidates: Counter = Counter()

    for word in WORD_RE.findall(text):
        lowered = word.lower()
        if lowered not in ENGLISH_STOP:
            candidates[lowered] += 1

    for run in CJK_RUN_RE.findall(text):
        for size in (2, 3, 4):
            for i in range(len(run) - size + 1):
                gram = run[i:i + size]
                if gram not in CHINESE_STOP:
                    candidates[gram] += 1

    ranked = sorted(
        candidates.items(),
        key=lambda kv: (kv[1] * max(1, len(kv[0]) - 1), len(kv[0])),
        reverse=True,
    )

    selected: List[str] = []
    for term, count in ranked:
        if len(selected) >= top_n:
            break
        if any(term in longer and candidates[longer] == count for longer in selected):
            continue
        selected.append(term)
    return selected


def _char_vector(text: str) -> Counter:
    compact = re.sub(r"\s+", "", text.lower())
    if len(compact) < 3:
        return Counter(compact)
    return Counter(compact[i:i + 3] for i in range(len(compact) - 2))


def _cosine(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 1.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    norm_left = math.sqrt(sum(value * value for value in left.values()))
    norm_right = math.sqrt(sum(value * value for value in right.values()))
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


def compute_U(sentences: List[str], terms: List[str]) -> float:
    scores = []
    for term in terms:
        indices = [i for i, sentence in enumerate(sentences) if term in sentence]
        if len(indices) < 2:
            scores.append(1.0)
            continue
        vectors = [_char_vector(sentences[i]) for i in indices]
        similarities = [
            _cosine(vectors[a], vectors[b])
            for a in range(len(vectors))
            for b in range(a + 1, len(vectors))
        ]
        scores.append(sum(similarities) / len(similarities))
    return sum(scores) / len(scores) if scores else 1.0


def compute_D(sentences: List[str], terms: List[str]) -> float:
    if not sentences:
        return 0.0
    positions = []
    for term in terms:
        for i, sentence in enumerate(sentences):
            if term in sentence:
                positions.append(i)
                break
    if len(positions) <= 1:
        return 0.0
    normalized = [position / len(sentences) for position in positions]
    mean = sum(normalized) / len(normalized)
    std = math.sqrt(
        sum((value - mean) ** 2 for value in normalized) / len(normalized)
    )
    return min(1.0, std * 2.5)


def _marker_in(sentence: str, marker: str) -> bool:
    if marker.isascii():
        return re.search(r"(?<![A-Za-z])" + re.escape(marker) + r"(?![A-Za-z])",
                         sentence.lower()) is not None
    return marker in sentence


def compute_A(
    sentences: List[str],
    text: str,
    rules: List[Tuple[str, str]],
) -> float:
    if not sentences:
        return 0.0
    marker_hits = sum(
        1 for sentence in sentences
        if any(_marker_in(sentence, marker) for marker in ADVERSARIAL_MARKERS)
    )
    marker_density = marker_hits / len(sentences)
    rule_hits = sum(1 for first, second in rules if first in text and second in text)
    rule_score = min(1.0, rule_hits * 0.3)
    return min(1.0, 0.6 * marker_density + 0.4 * rule_score)


def score_text(
    text: str,
    top_terms: int,
    lambda_u: float,
    lambda_d: float,
    lambda_a: float,
    rules: List[Tuple[str, str]],
) -> Dict[str, Any]:
    sentences = split_sentences(text)
    terms = extract_terms(text, top_terms)
    unity = compute_U(sentences, terms)
    development = compute_D(sentences, terms)
    adversarial = compute_A(sentences, text, rules)
    harmony = lambda_u * unity + lambda_d * development - lambda_a * adversarial
    return {
        "U": round(unity, 4),
        "D": round(development, 4),
        "A": round(adversarial, 4),
        "H": round(harmony, 4),
        "terms": terms,
        "sentences": len(sentences),
    }


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------

def score_trace(
    sessions: Dict[str, List[Dict[str, Any]]],
    top_terms: int = 10,
    lambdas: Tuple[float, float, float] = (
        DEFAULT_LAMBDA_U, DEFAULT_LAMBDA_D, DEFAULT_LAMBDA_A,
    ),
    rules: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Score each session turn exactly once.  Returns (rows, summary)."""
    rules = DEFAULT_CONTRADICTION_RULES if rules is None else rules
    rows: List[Dict[str, Any]] = []
    skipped_duplicates = 0
    skipped_empty = 0

    for session_id in sessions:
        history: List[Dict[str, str]] = []
        turn_index = 0
        for turn in sessions[session_id]:
            residual, dropped = strip_scored_history(turn["input_messages"], history)
            user_messages = [m for m in residual if m["role"] != "assistant"]
            if not user_messages:
                user_messages = residual
            output_messages = [
                m for m in turn["output_messages"] if m["content"].strip()
            ]
            assistant_text = "\n".join(m["content"] for m in output_messages)
            turn_text = "\n".join(
                [m["content"] for m in user_messages] + [assistant_text]
            ).strip()

            # No new user message means this span only replays scored history
            # (retry / duplicated export): count it zero times.
            if not residual:
                skipped_duplicates += 1
                continue
            if not turn_text:
                skipped_empty += 1
                continue

            # Advance the scored history only for turns that were counted.
            history = turn["input_messages"] + output_messages

            turn_index += 1
            scores = score_text(
                turn_text, top_terms, lambdas[0], lambdas[1], lambdas[2], rules
            )
            rows.append({
                "session.id": session_id,
                "turn_index": turn_index,
                "span_id": turn["span_id"],
                "gen_ai.input.messages": json.dumps(
                    user_messages, ensure_ascii=False
                ),
                "gen_ai.output.messages": json.dumps(
                    output_messages, ensure_ascii=False
                ),
                "gen_ai.usage.input_tokens": turn["input_tokens"] or "",
                "gen_ai.usage.output_tokens": turn["output_tokens"] or "",
                "history_messages_dropped": dropped,
                "input_messages_total": len(turn["input_messages"]),
                "scored_chars": len(turn_text),
                "U": scores["U"],
                "D": scores["D"],
                "A": scores["A"],
                "H": scores["H"],
                "_terms": scores["terms"],
                "_text": turn_text,
            })

    summary = {
        "sessions": len(sessions),
        "turns_scored": len(rows),
        "skipped_duplicate_spans": skipped_duplicates,
        "skipped_empty_spans": skipped_empty,
        "history_messages_dropped": sum(
            row["history_messages_dropped"] for row in rows
        ),
    }
    return rows, summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "session.id",
    "turn_index",
    "span_id",
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "history_messages_dropped",
    "input_messages_total",
    "scored_chars",
    "U",
    "D",
    "A",
    "H",
]


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: List[Dict[str, Any]]) -> None:
    header = f"{'session.id':<18}{'turn':>4}{'dropped':>8}{'chars':>7}" \
             f"{'U':>8}{'D':>8}{'A':>8}{'H':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['session.id']:<18}{row['turn_index']:>4}"
            f"{row['history_messages_dropped']:>8}{row['scored_chars']:>7}"
            f"{row['U']:>8.4f}{row['D']:>8.4f}{row['A']:>8.4f}{row['H']:>8.4f}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Minimal TraceAI (OTLP/GenAI) -> U/D/A/H adapter."
    )
    parser.add_argument("trace", help="OTLP JSON / JSONL trace export")
    parser.add_argument("--out", default="traceai_turn_scores.csv",
                        help="per-turn score CSV")
    parser.add_argument("--top-terms", type=int, default=10,
                        help="key terms per turn (default 10)")
    parser.add_argument("--lambda-u", type=float, default=DEFAULT_LAMBDA_U)
    parser.add_argument("--lambda-d", type=float, default=DEFAULT_LAMBDA_D)
    parser.add_argument("--lambda-a", type=float, default=DEFAULT_LAMBDA_A)
    parser.add_argument("--no-rules", action="store_true",
                        help="disable built-in contradiction phrase pairs")
    parser.add_argument("--show-terms", action="store_true",
                        help="print the extracted key terms per turn")
    args = parser.parse_args(argv)

    spans = load_spans(args.trace)
    if not spans:
        print(f"No spans found in {args.trace}", file=sys.stderr)
        return 1

    sessions = collect_turns(spans)
    if not sessions:
        print("No spans with gen_ai.input.messages / gen_ai.output.messages found.",
              file=sys.stderr)
        return 1

    rules = [] if args.no_rules else DEFAULT_CONTRADICTION_RULES
    rows, summary = score_trace(
        sessions,
        top_terms=args.top_terms,
        lambdas=(args.lambda_u, args.lambda_d, args.lambda_a),
        rules=rules,
    )

    write_csv(args.out, rows)
    print_table(rows)
    print()
    print(
        f"sessions={summary['sessions']} turns_scored={summary['turns_scored']} "
        f"duplicate_spans_skipped={summary['skipped_duplicate_spans']} "
        f"empty_spans_skipped={summary['skipped_empty_spans']} "
        f"history_messages_dropped={summary['history_messages_dropped']}"
    )
    print(f"wrote {args.out}")

    if args.show_terms:
        print()
        for row in rows:
            print(f"{row['session.id']} T{row['turn_index']}: {row['_terms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
