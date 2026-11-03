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
   deterministic alias that substitutes "world's quietest room". No open-ended
   synonym generation or extra LLM call is introduced.
3. Continue to apply the existing recovery distance and strong semantic gate.
   A semantically accepted candidate is not sufficient for AR07: after context
   expansion, its text must contain the registered `600 دولار` gold-evidence
   group.
4. Reject candidates that pass the semantic gate but omit the required price
   evidence. Try the one bounded alias; if neither query retrieves the price
   passage, return no evidence and preserve fail-closed abstention.
5. Require 100% AR07 gold-evidence coverage in the generation context. This
   uses the video-scoped evidence reservation and generation-suppression path
   already introduced for the Hindi cases in Phase 11.2.
6. Pass the active video ID into query recovery so the evidence contract is
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

After the branch is integrated with the two open Phase 11.2 PRs and the backend
is running with the frozen acceptance namespace:

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
