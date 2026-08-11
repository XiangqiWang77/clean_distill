# Qwen3-8B DeepMath-10 surrogate study

## Paper claim

TRSD's advantage comes from projecting an unstable privileged direction into a student-local neighborhood, yielding a less wrapper-sensitive and more reliable surrogate update. Raw privileged supervision is not assumed to be inherently helpful.

## Protocol

Exact frozen queries: 3,116. Reference tokens: 797,696. Query-bootstrap resamples: 10,000. Tokens are not treated as independent replicates.

## Headline evidence

- Prompt variance retained: **20.7%** (a 79.3% reduction).
- Style-token movement retained: **44.3%** (a 55.7% reduction).
- Mean gold-token gain: **-0.0391 → -0.0126** nats/token; it improves but remains negative.
- Queries with higher gold gain under TRSD: **96.8%**.
- Queries with higher worst-wrapper gain under TRSD: **98.9%**.

## Preregistered decisions

- Figure-1 signal-preservation rule: **False**. The positive-mean condition failed; the paper figure therefore does not claim that absolute signal quality is positive.
- Figure-2 surrogate-reliability rule: **True**.
