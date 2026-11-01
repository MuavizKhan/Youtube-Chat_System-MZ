# Phase 11.1 — English response contract and robust overview routing

## Why this change exists

The live-video review exposed a product-contract defect, not merely a property of the Hindi or Arabic captions.

The production answer prompt explicitly instructed the model to answer in the user's question language. A second instruction in the human prompt repeated that rule. The response validator also inferred the expected output language from the question, and the provider-failure fallback could return Hindi or Arabic. Consequently, the system was designed to change answer language when the question changed language.

That behaviour conflicts with the product requirement: the main answer is always English, independent of the question, transcript, captions, or conversation history. Original-language evidence is supplementary only.

The review also found that natural short overview requests such as "What is in this video?" and "What is he talking about?" were not all covered by the deterministic overview router. Those phrasings can be routed into ordinary similarity retrieval even though they are requests for representative video coverage rather than a specific fact.

## Changes

### 1. Fixed English response contract

- System and human generation instructions now require English for every main answer.
- Transcript language and question language are explicitly not response-language selectors.
- Hindi, Arabic, and Urdu excerpts are allowed only as short exact quotations copied from supplied context, followed immediately by an English meaning.
- The answer-language validator now checks for English as the fixed required language, while allowing short source-script quotations inside an otherwise English answer.
- The bounded language-repair instruction now rewrites non-English drafts into English without changing the evidence; if the provider ignores that repair, the application fails closed to the English abstention instead of returning a non-English main answer.
- When a model abstains despite non-empty retrieved context, the chain performs one bounded evidence recheck. It can recover supported partial answers, but must retain the exact English abstention when the supplied passages do not answer the question. It does not run a second repair loop or add new retrieval calls.
- Provider-quota and no-evidence fallbacks are English regardless of question language.
- The source-extractive Hindi tax-name fallback now explains the finding in English while preserving the exact Hindi term.
- Enumeration guidance now requires distinct proposals/teams to remain separate, forbids padding a requested count with invented items, and requires the model to disclose when the transcript supports fewer or an uncertain number.

Legacy observability keys named question_language_repair_* are retained for compatibility; their behaviour now enforces the fixed English response contract rather than matching the question language.

### 2. Natural overview routing

The deterministic overview detector now recognises additional common requests, including:

- "What is in this video?"
- "What's inside this video?"
- "What is he talking about?"
- "What is the speaker discussing?"
- "What does this video cover?"
- "Tell me about this video."

These questions route to the existing representative transcript overview mechanism before ordinary similarity retrieval. This does not weaken relevance thresholds or change retrieval ranking for specific factual questions.

### 3. Preserved safeguards

This phase does not change:

- Embedding model, vector indexes, index namespace, or chunking.
- Dense/lexical retrieval ranking, MMR, reranking, distance thresholds, or recovery budgets.
- Prompt-visible citation provenance or timestamp selection from source metadata.
- The Phase 11 semantic assessor's provider contract or human-annotation requirements.

No reindexing is required.

## Regression coverage

Tests cover:

- English-only output instructions when both the question and transcript are Hindi.
- Arabic/Hindi answer drafts being detected as non-English even when the question is also non-English.
- English answers containing short Arabic/Hindi terms being accepted.
- English-only abstention fallback for English, Hindi, and Arabic questions.
- English repair instructions for non-English answer drafts, with fail-closed behaviour if the provider ignores the repair.
- One bounded evidence recheck for model abstentions, with a retained abstention when the supplied passages remain insufficient.
- Natural short overview phrasings routing to overview retrieval.
- Specific factual questions remaining outside overview routing.

## Validation and release boundary

This PR must pass the repository's GitHub Actions checks and full automated test suite before merge. A passing unit suite proves the deterministic contract and routing code, not the configured provider's semantic accuracy.

After merge, run the live answer/source audit with English questions against English, Hindi, and Arabic transcript videos. Confirm:

1. The main answer is English in every case.
2. Any original-language excerpt is present verbatim in a prompt-visible transcript passage and is immediately translated.
3. Source links and timestamp intervals remain exact and prompt-visible.
4. Unanswerable questions still abstain in English.
5. Short overview prompts return a useful overview, while specific unrelated questions continue to abstain.

The live validation must include at least one unseen video and paraphrased questions. These safeguards reduce known classes of failure; no prompt or software change can guarantee that a generative model will never make a semantic error on every future YouTube video. The release criterion is measured generalisation with retained evidence provenance, not an absolute claim of perfection.
