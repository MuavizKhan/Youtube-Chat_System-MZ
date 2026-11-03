# Phase 11.2: Hindi named-movement answer evidence

## Scope

This change corrects the live answer-quality contract for the Hindi Rajasthan peasant-movements video (`INpnuL7qnzY`) and the two English questions over its Hindi captions:

- `HI01`: Which peasant movement is named as the first major movement?
- `HI06`: Which farmers' movement is described as the first major movement?

## Source basis

The video's published, automatically generated Hindi caption text in the opening 30 seconds reads in part:

`पहला बड़ा किसान आंदोलन कौन सा हुआ था अपने बिजोलिया`

The speaker names Bijolia. The evidence source is the video's caption transcript:
https://prepublish.ai/youtube-transcript/INpnuL7qnzY

The answer contract therefore requires the named movement (`Bijolia` / `बिजोलिया`), not merely a restatement such as "the first major farmers' movement".

## Failure mechanism

The previous gold-evidence annotation protected only the generic phrase `पहला बड़ा किसान आंदोलन`. It did not mark the answer-bearing name as required generation evidence. The generation-context packer already supports reserving registered gold evidence and suppressing generation when required evidence is absent; the annotation was not activating that mechanism for these two cases.

HI06 also had a permissive audit alternative that accepted the generic phrase without requiring the movement's name.

## Changes

- Register `बिजोलिया` / `Bijolia` as the gold answer evidence for HI01 and HI06.
- Set generation-context gold-evidence coverage to 100% for both questions. The existing packer must retain the named movement before generation; if it is absent from the available/prompted context, the existing fail-closed path abstains.
- Require the name Bijolia in both live audit answers.
- Normalize Unicode smart apostrophes in the audit's answer-requirement text so punctuation does not create accidental matcher differences.
- Add unit tests for budgeted caption retention, generation repair, fail-closed behavior, audit acceptance/rejection, and apostrophe normalization.

## Configuration guard

No embedding model, index namespace, retrieval threshold, MMR parameter, reranking option, chunking setting, or index artifact is changed.

The live recheck must continue to use the frozen Phase 10.11 acceptance configuration:

- Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Namespace: `phase10-10-acceptance-minilm-20261023`
- `top_k=4`, `fetch_k=10`, `lambda_mult=0.7`, `max_distance=1.3`

## Acceptance

1. Unit tests prove that the Hindi caption containing Bijolia is retained under context-budget truncation for HI01 and HI06.
2. Unit tests prove generation is suppressed if the named movement is absent from the final generation context.
3. The audit passes only when the answer names Bijolia in English or Hindi.
4. The two live cases return HTTP success, report the expected frozen retrieval configuration, include source/timestamp provenance, and pass answer quality.
5. Full-registry coverage remains a separate requirement; a two-case diagnostic is not claimed to validate all 43 registered cases.


## Follow-up integration correction

CI on the first PR revision exposed that `chain.py` still looked up generation-time gold evidence through the legacy global `gold_evidence.py` registry. That registry contains the original Kingfisher benchmark only. Later per-video annotations (including the Hindi pilot) live on `VideoBenchmark.gold_evidence_by_question`.

The chain now obtains gold evidence from the matched video's own benchmark. This is required for the Hindi context reservation, repair, and fail-closed behavior to activate. Regression tests verify that HI01 and HI06 resolve to their video-scoped evidence records. Audit tests also select HI cases from the Phase 11.0 registry rather than the older general audit subset.
