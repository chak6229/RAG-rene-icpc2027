# Review Cases and Disagreement Analysis

This discussion analyzes the original human-review returns and the completed supplementary code-relevance assessments. It preserves both judgments and interprets their relationship to the cached retrieval outcomes. It does not create new reviewer labels or a consensus adjudication. Original agreement and weighted rates are reported in mechanism_review_return_results.md, with source labels in results/rene_review_returns_20261020/. Completed supplementary labels and paired relevance diagnostics are in results/rene_review_completion_20261021/ and relevance_sensitivity_results.md.

## Where the judgments differ

There are 90 semantic-class disagreements among 383 rewrites. Sixteen additional rewrites have matching semantic classes but different issue types, yielding 106 rewrite disagreements. Among 327 code panels, 151 relevance labels differ; 35 of the 128 query-clarity labels differ. At least one difference occurs in 122 sampled queries. These counts refer to displayed items and are not frame-weighted prevalence estimates.

|A / B semantic class|Count|Interpretation for discussion|
|---|---:|---|
|drift / preserved|47|Whether additional wording constrains the task or expresses assumed context|
|preserved / drift|7|The reverse boundary; semantic disagreement is not one-directional|
|drift / uncertain|14|A definite alternative reading versus insufficiently determined meaning|
|malformed / uncertain|16|Incoherent text versus an unresolved intended request|
|malformed / preserved|2|Whether a readable original task can still be recovered|
|drift / malformed|2|Changed task versus inability to state a coherent task|
|preserved / uncertain|2|A recoverable original task versus residual ambiguity|

These are complete class-pair counts. The interpretations name discussion issues, not a new exhaustive thematic coding of the 90 texts. For example, adding `http` to `set python request headers` is drift for A because it restricts the protocol, but preserved for B because it expresses the usual request-header context. Adding `value` raises a similar boundary. The factual question is whether the added wording changes acceptable answers under the original request, not simply whether a new token appears. By contrast, both reviewers call `python get linux from clipboard` drift, but A marks an unsupported constraint and B an operation change. This illustrates type disagreement within an agreed semantic problem; both original type labels are retained.

Code disagreement has two distinct numerical components. Sixty-five panels have a numeric label from A and unknown from B (29 at A=2, 32 at A=1 and four at A=0). The remaining 86 differences are between numeric grades: B is lower on 82 and higher on four. B uses unknown on 67 panels in total, A on two. Missing helper implementations, surrounding classes and query context recur in their written reasons; full versus partial functionality is also a substantive issue. The jointly judged subset therefore cannot replace the full-panel result: its linear-weighted kappa 0.569 is reported with 260/327 coverage, alongside full-panel nominal kappa 0.371.

Query clarity differs in 28 A-clear/B-ambiguous cases, four A-clear/B-insufficient cases, two A-ambiguous/B-clear cases and one A-ambiguous/B-insufficient case. A labels 124/128 queries clear, whereas B labels 94 clear. The resulting 72.7% exact agreement and kappa 0.029 reflect strongly imbalanced marginal labels; neither alone certifies query clarity. These judgments help explain why a short query can admit more than one relevance interpretation.

## Example selection and provenance

We use the lexicographically first query ID with each of three already computed query flags: any definite semantic issue, meaning-preserved verified-stage loss, and displayed non-target relevance. This is a transparent post-hoc illustrative rule, not a representative or independently sampled case series. Rates and uncertainty come from all 128 queries under the original eight-stratum design, not from these examples. The query IDs retain upstream `dev`/`train` names; study membership follows the frozen H manifest rather than these strings.

Full original notes, code panels, candidate fractions, sample weights and decisions are exported in results/rene_manuscript_integration_20261021/REVIEW_CASES.json and REVIEW_CASES.md, with input hashes and label-file line references. The three cases below concern the CodeBERT H frame.

## Case 1: an unsupported constraint need not change target RR

**cosqa-dev-400.** Original: `python output colors from a string`. Candidate position 5 is `python output colors from a json string`. Both reviewers label drift/unsupported constraint: the original did not require a JSON input. Its target RR stays at 1/5, so its delta is exactly zero. The best candidate in the same pool, `python output the colors from a string`, is marked preserved by both and improves target RR from 1/5 to 1/3. The other preserved candidate reaches rank four.

The gain, verified and risk stages retain the original query in this case; no cross-encoder work is allocated. Thus the existence of a drifted candidate is not proof that it caused the policy's loss, and semantic drift can be invisible to this single-target RR outcome. The case also contains a preserved positive opportunity that the policy does not adopt. Its separate indicators may overlap at the query level.

## Case 2: a preserved request can lose labeled-target rank

**cosqa-dev-106.** Original: `get index of an item in list python`. The adopted position-6 rewrite is `get index of an item in the list python`. Both reviewers mark preserved/none, describing `the` as an article that leaves the index-finding request intact. Target RR changes from 1/16 to 1/18, delta -1/144. Another both-preserved candidate, `get index value of an item in list python`, reaches rank 14, delta +1/112. The policy's cached gain score orders the adopted action above this positive alternative, providing a concrete observed prediction/action mismatch.

The RIDGE_verified stage makes the same action and allocates zero cross-encoder pairs here. The stage name must not be interpreted as evidence that the cross-encoder inspected this query and made an erroneous judgment. The risk policy retains the action. This case supports the claim that the observed RR loss is not automatically a semantic-drift label; it does not identify which retriever, label-coverage or learning intervention would remove the loss.

The displayed code also contains a substantive relevance disagreement. Document 456 returns `index_or_name` unchanged if it is an integer, and otherwise calls `lst.index(index_or_name)`. A grades it fully relevant (2); B grades it partially relevant (1), noting that an integer value is treated as an index rather than searched as an element. This distinction follows an actual code branch. For document 6214, which searches through `self.iter()`, A assigns 2 while B uses unknown because the iteration behavior is not shown. The known target uses bisection; both grade it partial because the original request does not ensure sorted input. These three different judgments should not be collapsed into a single rater-style explanation.

## Case 3: relevant displayed alternatives outside the known target

**cosqa-dev-198.** Original: `set python request headers`. The selected rewrite, `set the python request headers`, is marked preserved by both. The known target moves from rank one to rank two, delta -1/2. The top displayed alternative, document 3330, constructs a `requests.Session` and calls `self.req_session.headers.update(self.headers)`. Both reviewers grade it fully relevant (2), although it is outside the known-target labels. This is direct displayed-panel evidence that the target-only RR loss can coexist with a result assessed as useful for the request.

The gain and verified stages adopt the position-1 rewrite without cross-encoder allocation; the risk stage restores the original. The known-target panel calls an unseen `self.with_header` helper: A grades it 2, B uses unknown. The same query's `http` rewrite supplies the added-context semantic disagreement discussed above. These observations flag relevance coverage and context boundaries; they do not estimate full-corpus missing-label prevalence or prove that relabeling would eliminate the aggregate policy gap.

## Supplementary code-relevance disagreements

The same two reviewers independently assess 344 additional panels across 126 of the original queries. These are new code judgments, not additional queries, rewrite judgments or reviewers. The original 106 rewrite, 151 code and 35 clarity disagreements remain unchanged.

|Supplementary relevance grade|A|B|
|---|---:|---:|
|2: fully relevant|21|58|
|1: partially relevant|103|146|
|0: not relevant|172|140|
|u: unknown/context insufficient|48|0|

Supplementary exact agreement is 55.23%, with nominal kappa 0.321 including unknown. The jointly judged subset covers 296/344 panels (86.05%); its linear weighted kappa is 0.484. Of 154 disagreements, 48 involve A's unknown and B's numeric judgment. The remaining 106 differ between numeric grades: B is higher on 83 and lower on 23. This direction differs from the original panels' numeric-grade pattern and does not establish that either reviewer is invariably stricter. The panels also differ between batches, so their kappas cannot measure a change in assessment quality.

In the merged 610-panel Top-3 pool, 266 panels inherit original labels and 344 receive supplementary labels. Each reviewer retains 49 unknown judgments. Jointly judged coverage is 513/610 (84.10%), with linear weighted kappa 0.527; nominal kappa including unknown is 0.340. These coverage figures describe the merged pool, not all 327 original code panels plus the supplement.

### Additional illustrative cases

For each of the 0/2, u/2 and 1/2 disagreement categories, we take the lexicographically first query_id/panel_id pair after receiving the labels. This transparent post-hoc choice illustrates functional boundaries; it is not a prespecified representative sample or an exhaustive thematic coding of the 154 disagreements. Full code and both original notes are in results/rene_review_completion_20261021/REPRESENTATIVE_DISAGREEMENTS.json.

**Request versus response: cosqa-train-13132, `python get post content type`.** The `json_or_text(response)` implementation reads the response's Content-Type and decodes its body. A assigns 0 because it does not read the POST request's content type; B assigns 2 because Content-Type processing and parsing are usable. The disagreement concerns which functional object must satisfy the request.

**Missing helper versus visible function: cosqa-dev-169, `python if type array`.** The `is_integer_array` function calls an unseen `is_np_array` helper before testing an integer dtype. A uses unknown because the array-type check cannot be confirmed from the displayed context; B assigns 2 based on the visible NumPy integer-array check. The different labels expose how missing context enters relevance assessment.

**Transformation versus completed output: cosqa-dev-400, `python output colors from a string`.** The `highlight` function returns a string wrapped in ANSI control codes. A assigns 1 because it supplies a display-ready transformation without an output step or parsing colors from the string; B assigns 2 because the ANSI-wrapped string can display color. The disagreement concerns what completion of the short request requires.

These accounts preserve the observed 0/2, u/2 and 1/2 pairs without choosing a winning interpretation. They supplement the three original semantic/ranking cases. Their relationship to the aggregate utility reading is tested through separate reviewer-specific relevance sets and exact unknown-relevance bounds, not by merging labels or assigning causal percentages.

## What is resolved by this discussion

The final-label agreement, complete class-pair counts, weighted issue/event rates, completed supplementary coverage and traceable examples now form a coherent diagnostic account. It distinguishes semantic changes, meaning-preserved ranking effects, displayed alternative relevance, functional boundaries and missing context. Both original ratings and all unresolved judgments remain available. No semantic class is reassigned and no new consensus rate is substituted for the reported agreement.

The remaining scientific boundary is causal attribution: generation, retrieval, relevance coverage and decision learning have not been intervened on independently. The paper uses the present evidence to explain what observed outcomes can mean, rather than assigning exclusive shares of the unrealized gain. Any later reviewer consensus should be recorded as an additional version, with pre-consensus labels retained.
