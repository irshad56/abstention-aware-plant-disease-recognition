# Final Plant Disease Literature Evidence Release

## Purpose
A research evidence package for integrating scientific literature retrieval
with an existing abstention-aware plant disease application.

## Contents
- evidence_reports/: retrieved papers and Blocks 7–9 evidence audits
- ANTIGRAVITY_INTEGRATION_PROMPT.md: implementation specification

## Research scope
Current worked example: Grape Black Rot.

## Evidence limitations
- Abstract-level review only.
- Exact quotation matching does not establish scientific validity.
- No full-text review has been completed.
- Journal quartiles have not been verified.
- Search was not an exhaustive systematic review.
- Environmental-condition claims were not established by the supplied abstracts.

## Integration safety
Disease-specific evidence retrieval must be blocked unless the inference
result is ACCEPTED and explicitly eligible for evidence retrieval.

## Production status
This package is a research handoff. It does not modify the deployed
application, model checkpoints, thresholds or inference engine.
