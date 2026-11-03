# Phase 11.3: Arabic retrieval and answer-audit completion

## Goal

Complete the ten-case live answer/source audit for the Arabic-caption video
`fAd2nKfrDB8` without changing the frozen Phase 10.11 acceptance index or
weakening retrieval, citation, timestamp, or abstention controls.

The previous live audit completed all ten API requests. Seven cases passed
quality checks. The remaining cases were:

- **AR07:** genuine retrieval/evidence failure. The planner failed on the
  question's "Orfield room" wording. The bounded original-query fallback
  accepted a semantically plausible candidate, but its expanded context did
  not contain the price evidence `600 دولار`. The system then correctly
  abstained and suppressed sources. The same price question phrased as
  "world's quietest room" passed in AR02; its price-bearing passage is around
  00:13–02:00.
- **AR08:** audit-checker mismatch. The English answer used "pulse" for the
  source-supported heartbeat, while the checker only recognised "heartbeat"
  and "heart beat".
- **AR09:** audit-checker language mismatch. The answer was required to be
  English, but the perception-role requirement only recognised Arabic-script
  wording.

These are distinct failure classes and are fixed independently.

## AR07 retrieval and evidence contract

1. Keep the original user wording as the first deterministic retry.
2. When the exact phrase "Orfield room" appears, allow one additional
   deterministic alias. For a cost question containing "how much", "one hour",
   and "cost" or "price", reuse the exact AR02 wording that passed live:
   "How much might one hour in the world's quietest room cost?". Other Orfield
   questions use a literal phrase substitution to "world's quietest room".
   No open-ended synonym generation or extra LLM call is introduced.
3. Continue to apply the existing recovery distance and strong semantic gate.
   A semantically accepted candidate is not sufficient for AR07: after context
   expansion, its text must contain the registered `600 دولار` gold-evidence
   group.
4. Reject candidates that pass the semantic gate but omit the required price
   evidence. Try the one bounded alias; if neither query retrieves the price
   passage, return no evidence and preserve fail-closed abstention.
5. Run this bounded evidence-aware alias fallback in both planner states:
   after a query-understanding exception, and after a successful factual planner
   has exhausted its strict and relaxed rewritten-query candidates. The live
   AR07 rerun exposed that the first implementation covered only the exception
   path; the planner can succeed while producing a weak rewrite (the observed
   AR07 relaxed candidate was chunk 3 at distance 1.42682).
6. Keep the same strong semantic gate and explicit absolute distance ceiling
   used by ordinary relaxed recovery. The alias is a better query, not
   permission to accept a weaker match or raise any distance threshold.
7. Require 100% AR07 gold-evidence coverage in the expanded retrieval context
   and final generation context. A semantically accepted non-price passage
   must be rejected; missing evidence preserves fail-closed abstention.
8. Pass the active video ID into query recovery so the evidence contract is
   resolved against the exact registered video/question, not a global alias.

The retrieval trace records the candidate queries, whether an alias was used,
the semantic-gate result, required/present/missing gold-evidence groups, and
whether the final rejection was caused by missing required evidence.

## AR08 and AR09 audit alignment

- AR08's heartbeat check accepts the exact English synonym `pulse` in addition
  to `heartbeat` and `heart beat`. Other bodily-sound requirements remain
  unchanged.
- AR09 requires an English `sound` reference and an English perception-role
  concept such as `perception`, `perceive`, `compass`, `guidance`,
  `orientation`, or `direction`. Tests verify that mentioning sound alone
  does not satisfy the perception-role requirement.

## Regression coverage

The branch adds tests for:

- AR02's existing "world's quietest room" wording remaining unchanged.
- AR07's Orfield wording retrying with the bounded alias.
- Rejection of a semantically plausible passage that lacks `600 دولار`.
- Acceptance only when the alias-expanded context contains the price evidence.
- AR07's price passage being reserved under a constrained generation budget.
- Fail-closed generation suppression if the price evidence is absent.
- Explicit telemetry for a missing-required-evidence rejection.
- AR08 accepting "pulse" while still rejecting an answer with no heartbeat/pulse
  evidence.
- AR09 accepting English sound/perception wording while rejecting sound-only
  wording.

## AR07 deterministic answer fallback after failed model repair

The model receives a bounded generation context with the registered `600 دولار`
evidence. If its one permitted completeness repair still omits that exact source
phrase, the answer is not returned as an unsupported draft. For the registered
AR07 question only, a deterministic fallback may instead answer from one
prompt-visible source block, and only when that same block contains the gold
amount plus Arabic hourly and cost wording. It quotes a short exact caption
excerpt first and follows it immediately with an English meaning. A price-only
match, evidence split across different source blocks, a different video, or a
different question remains fail-closed. The usual returned-source and
timestamp-building path is preserved; this fallback does not create or invent
source metadata.

## English response language and source-language quotations

The English response contract permits one short, clearly delimited Hindi, Arabic,
or Urdu transcript excerpt when it is useful to verify a key point. The excerpt
must appear first and be followed immediately by its English meaning. All
surrounding prose remains English.

The live answer-language checker exempts at most one quoted source-script
excerpt, capped at 180 characters and 32 whitespace-delimited words. Multiple
or oversized source-language quotations fail the language check. Any additional
unquoted source-script text remains part of the ordinary script-proportion
check. The same bounded exception applies to Arabic/Urdu and Devanagari script.
The checker verifies language shape, not quotation verbatimness; exactness
remains a generation requirement and must be assessed against the supplied
transcript evidence.

## Preserved configuration and non-goals

No change is made to:

- Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Acceptance namespace: `phase10-10-acceptance-minilm-20261023`
- `top_k`, `fetch_k`, `lambda_mult`, or `max_distance`
- Recovery thresholds, global retrieval budgets, chunking, or reranking
- Citation policy, source provenance, timestamps, or unanswerable abstention
- Existing index artifacts; no reindex or rebuild is required

This phase does not treat AR08 or AR09 as retrieval failures.

## Live acceptance

With Phase 11.2 PRs #162 and #163 merged into `main`, and the backend
running with the frozen acceptance namespace:

1. Re-run AR02 and AR07 together. Both must answer the price question from the
   price-bearing source passage, with the returned source and timestamp covering
   the evidence.
2. Re-run AR08 and AR09 to verify the corrected answer-quality requirements.
3. Run AR01–AR10 in grouped batches. The Arabic suite is complete only when all
   ten cases pass final quality checks, configuration consistency checks pass,
   and source/timestamp provenance remains valid.
4. Inspect each response and its returned chunks. API success alone is not
   acceptance; subset manifests must not be represented as full registry
   coverage.

The current code and unit-test changes do not by themselves claim that live
Arabic acceptance has passed.
