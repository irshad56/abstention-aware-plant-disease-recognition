# ANTIGRAVITY IMPLEMENTATION TASK
## Integrate an abstention-aware scientific literature evidence retrieval module

You are modifying an existing deployed Streamlit plant-disease application.

The existing application already has:
- A trained multi-model plant-disease inference system.
- Image quality assessment.
- Leaf verification.
- Supported taxonomy/OOD decision logic.
- Disease prediction confidence and abstention.
- A Groq API integration using model openai/gpt-oss-120b.
- An existing user interface and prediction workflow.

Your task is to integrate a reliable, disease-specific scientific literature
retrieval and evidence synthesis module WITHOUT rebuilding the application.

IMPORTANT: Inspect the existing codebase first. Do not assume that the
Kaggle research prototype is the same as the existing Antigravity implementation.
Preserve working functionality.

## 1. NON-NEGOTIABLE PRESERVATION RULES

Do not:
- Redesign or remove the existing Streamlit UI.
- Retrain, replace, or modify the five vision models.
- Change model preprocessing or prediction classes.
- Change IQA, leaf verification, taxonomy thresholds, or disease thresholds.
- Modify the existing abstention policy.
- Send disease-specific requests for abstained predictions.
- expose API keys or credentials.
- Invent papers, DOI values, journal rankings, or scientific findings.
- Describe a paper as Q1 unless independently verified.
- Present abstract-only evidence as full-text verified evidence.

Keep existing working Groq functionality. Reuse the current secret-management
approach and avoid hardcoding credentials.

## 2. REQUIRED INFERENCE-TO-RETRIEVAL SAFETY CONTRACT

The literature retrieval component must receive the structured inference result,
not an arbitrary disease name supplied by an untrusted client.

Disease-specific retrieval is allowed only if all conditions are satisfied:

- status == "ACCEPTED"
- evidence_eligible is exactly True
- disease prediction is non-empty
- disease confidence is finite and within [0, 1]
- leaf probability is finite and within [0, 1]
- supported taxonomy probability is finite and within [0, 1]

If any condition fails:
- Do not execute disease-specific literature search.
- Do not send the predicted disease name as a disease-specific evidence request.
- Return only generic safe guidance.
- Preserve the existing abstention explanation.

Use the hardened safety bridge from the research release as a reference,
but inspect and adapt it to the application's actual inference-result schema.

## 3. LITERATURE RETRIEVAL PIPELINE

Implement a modular pipeline:

A. Disease query normalization
- Use the accepted disease label.
- Resolve the likely pathogen using a controlled taxonomy/alias mapping.
- Keep the original predicted label visible.
- Do not silently substitute a different disease.

B. Scientific discovery
- Use OpenAlex as a discovery source.
- Use Crossref to verify DOI and bibliographic metadata.
- Search disease name, pathogen synonyms, and scientifically relevant aliases.
- Deduplicate by DOI and normalized title.
- Preserve source URL, DOI, title, journal, publication year and abstract.
- Record the query and retrieval timestamp.

C. Relevance ranking
- Score direct disease relevance above general crop context.
- Distinguish direct pathogen/disease studies from related diseases.
- Consider study type, publication year, abstract availability and citation
  metadata without treating citation count as proof of quality.
- Do not use a simple citation-count sort as the primary relevance rank.
- Exclude unrelated diseases from disease-specific evidence synthesis.

D. Metadata and journal quality
- Verify DOI metadata against Crossref.
- Do not claim Q1/Q2 status from journal name or citation count.
- If quartile data is unavailable, label it "NOT VERIFIED".
- Record publication type: review, field study, laboratory study, genomic
  study, book chapter, preprint or unclear.
- Keep preprints and book chapters distinguishable from peer-reviewed
  journal articles.

E. Evidence extraction
- Send selected abstracts and metadata to Groq.
- Require source IDs and exact DOI references for claims.
- Require short verbatim quotations for directly supported claims.
- Separate:
  1. Disease symptoms
  2. Epidemiology and infection
  3. Environmental conditions
  4. Pathogen biology
  5. Host resistance
  6. Management and control
- Distinguish direct support, partial support and unsupported information.
- Never fill gaps with model-generated general knowledge.
- Clearly label abstract-only evidence.
- Do not convert recommendations in a review into proven field efficacy.

F. Citation integrity
- Every displayed paper must link to its DOI or verified publisher page.
- Every evidence claim must map to one or more retrieved source IDs.
- Reject unknown source IDs and DOI mismatches.
- Store the original title, journal, year, DOI and source URL.
- Never invent a source to complete a report.

## 4. OUTPUT FORMAT

For an accepted disease prediction, present:

1. Predicted disease and pathogen (if supported by verified metadata).
2. Short evidence overview.
3. Three to five relevant papers when sufficient quality sources exist.
4. Each paper's title, journal, year, DOI link and publication type.
5. Evidence grouped by the six topics above.
6. Exact quotations for directly supported claims.
7. Partial evidence and evidence gaps.
8. Limitations and a clear abstract-only/full-text-reviewed label.
9. A clear statement when journal quartiles are not verified.

Do not force five papers if fewer suitable papers exist.

## 5. CACHING, PERFORMANCE AND ERROR HANDLING

- Cache search and metadata results with a configurable expiry.
- Avoid repeating the same API queries unnecessarily.
- Add request timeouts, retries and rate-limit handling.
- Handle empty search results and unavailable abstracts gracefully.
- If a source cannot be verified, do not cite it as verified.
- Keep retrieval failures separate from model inference failures.
- Do not block the user's basic disease prediction if literature services fail.

## 6. TESTING REQUIREMENTS

Create automated tests for:

- Accepted inference -> disease-specific retrieval allowed.
- Abstained inference -> disease-specific retrieval blocked.
- Missing confidence -> blocked.
- Invalid or non-finite probability -> blocked.
- Unsupported taxonomy -> blocked.
- DOI mismatch -> source rejected.
- Unknown citation source ID -> source rejected.
- Missing abstract -> evidence marked insufficient.
- Irrelevant paper -> excluded from disease-specific synthesis.
- Empty retrieval response -> graceful result.
- Groq failure -> prediction interface remains usable.
- API key never appears in logs or UI.

Use mocks for API tests. Do not incur unnecessary live API calls.

## 7. RESEARCH PROTOTYPE LIMITATIONS

The supplied research package concerns Grape Black Rot.

Its current evidence review:
- Contains nine selected journal papers.
- Has exact-quote validation and a separate relevance review.
- Identifies eight direct, three partial and one irrelevant quotation assessments.
- Does not establish environmental-condition claims from the supplied abstracts.
- Has not undergone full-text review.
- Has not verified journal quartiles.
- Is not an exhaustive systematic review.

These findings are a disease-specific prototype, not a universal evidence
quality guarantee for every disease.

Use the supplied reports as test fixtures and examples, not as hardcoded
production evidence for all diseases.

## 8. DELIVERABLES

Before editing:
- Inspect the current project structure.
- Identify the actual inference entry point, safety bridge, Groq client,
  Streamlit UI and configuration files.
- Report the integration plan and files to be changed.

Then implement:
- A separate literature retrieval module.
- A separate evidence validation module.
- A clear interface from accepted inference results.
- Automated tests.
- A short README describing configuration and limitations.

Keep changes minimal and modular.

After implementation:
- Run tests.
- Report exact files changed.
- Report tests passed and failed.
- Confirm that vision checkpoints, thresholds, abstention logic and UI
  remain unchanged.
- Demonstrate one accepted and one abstained inference route.

Do not claim successful deployment until the application has actually
been run and verified.
