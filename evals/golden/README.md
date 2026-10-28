# Golden eval set

`golden.jsonl`, one object per line:

```json
{"qid": "q001", "question": "...", "reference_answer": "...", "gold_chunk_ids": ["doc3_p2_c1"], "answerable": true}
```

Process: LLM-generate candidates per chunk → hand-verify → dedupe → keep ~10–15% unanswerable.
Target size: 50–200.
