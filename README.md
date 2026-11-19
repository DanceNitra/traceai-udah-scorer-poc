# traceAI U/D/A/H Scorer ¡ª Scoped POC

Minimal Python adapter: reads traceAI-exported OTLP/GenAI traces, scores each turn with U/D/A/H.

## Usage
    python traceai_adapter.py examples/traceai_sample_trace.json --out traceai_turn_scores.csv

## Layout
- traceai_adapter.py        core adapter (stdlib only)
- trajectory_v3_ultra.py    analysis helper
- examples/                 synthetic trace + sample scores
- gold_standard/            47-case benchmark + diagnostics
- CAVEATS.md                known limits
