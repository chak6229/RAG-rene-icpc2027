# Double review returns: consistency, weighting and disagreement

This report analyses the final returns submitted by both reviewers. User confirmed completion of manual review. Current consistency corresponds to the final return label; The original independent label prior to the review has not been reconciled and, therefore, the consistency of the original independent judgement is not estimated in this report.

This time keep the original directory, original zip and item-by-item labels, only return results, without implementing attachments HTML or pending code.

## Integrity and the role of evidence

Both returns overwhelm the same scheduled 128 query/ 120 group, with different changes to each 383 article and 327 code panel. packet, A/B Identity, collection of entries, label combinations, justification length and sequence approval; The semantic records embedded in each final document are consistent with the first phase of independence JSON. B With two pages HTML identical to the original package.

These labels supplement the type of error and mechanism interpretation; Statistics cannot separate semantics, relevance and profit-learning factors as a source of cross-resortment causes and consequences of income gaps.

## Consistency and judgement distribution

| Judgement. | Number of items/overrided | Accuracy Consistency Rate | Cohen κ |
|---|---:|---:|---:|
| Rewrite meaning, 4 Class | 383 | 76.5% | 0.525 |
| The code is relevant and incomprehensible. | 327 | 53.8% | 0.371 |
| Codes that can be determined by both parties, 0 / 1 / 2 | 260/327（79.5%） | 66.9% | 0.569 (linear weight) |
| Level of clarity of original query | 128 | 72.7% | 0.029 |

Consistency is calculated on the basis of presentation entries and multiple locations of the same text are not counted; It does not represent 8530 overall weighted consistency. The reservation is entirely unjudgeable and does not replace the whole result with a hundred that can be judged higher.

| Rewrite Tab | A | B |
|---|---:|---:|
| meaning preserved | 235 | 275 |
| meaning drift | 129 | 73 |
| Unreadable | 19 | 3 |
| cannot determine | 0 | 32 |

| Code Label | A | B |
|---|---:|---:|
| Satisfied | 96 | 34 |
| Partially satisfied | 151 | 102 |
| Unsatisfactory | 78 | 124 |
| cannot determine | 2 | 67 |

Semantic labels 90 are different; Article 106 is rewritten to be discussed, with different types of errors, where 16 is considered by both parties as being biased but not the same type. The code 151 is different, and subparagraph 65 relates to a party that cannot be judged by the other party. B of the code available to both parties has a lower rating than A 82, higher 4. There was also a disagreement on the clarity of the original article 35; Article 124 was considered clear by A and article 94 only by B. There is at least one difference between the total 122/128 query. There was no automatic award or deletion of the disagreement.

## Summary by weight of original sample

The following denominator is a fixed CodeBERT H general 8530 query, with the indicator " Query contains an evaluation mark/event " and not a candidate error rate. Every one of them. 16 Article, with N_h/n_h weights; Can't treat the nudity ratio of 128 as an overall ratio. (a) Inter-situational over-geometric inverse weighting of eight layers within each indicator according to the original scheme Bonferroni is a conservatively limited total area 95%; Not all indicators are claimed to be covered at the same time 95%. The interval reflects only sample error, excluding label error judgement or return version difference.

| Query level indicator (allowing overlap) | Weighted estimate | Conservative 95% |
|---|---:|---|
| At least one rewrite has been marked as bias/not readable by both sides | 35.0% | [9.9%, 69.6%] |
| At least one rewrite that either side does not maintain | 75.5% | [39.6%, 95.4%] |
| At least one rewrite is marked as unjudgeable. | 26.9% | [5.6%, 62.6%] |
| The modification actually applied after verification was marked as a semantic issue by both parties | 4.2% | [0.1%, 36.6%] |
| Upon verification, RR losses were sustained by rewriting the meaning of both parties. | 12.9% | [7.6%, 40.3%] |
| The existence of the best positive opportunity for both parties not fully realized and maintained | 14.3% | [4.3%, 45.9%] |
| At least one non-existent target display code was found to be met by both parties. | 12.0% | [1.3%, 46.2%] |
| At least one of the existing target display codes was partially/unsatisfactory on both sides | 45.1% | [15.7%, 77.6%] |
| A Mark at least one offset/not readable | 75.4% | [39.6%, 95.4%] |
| B Mark at least one offset/not readable | 37.6% | [10.9%, 72.0%] |

35.0% of the “equal issue” and 75.5% of the “any unreserved” use different definitions, which contain incalculable and divergent statistical boundaries that are not the same true error rate. It also shows that the test of “restrictive change” will materially affect the outcome.

| The same type of error was marked by both parties | Weighted Query Ratio | Conservative 95% |
|---|---:|---|
| Add a new limit without basis | 20.8% | [4.6%, 54.9%] |
| Loss of original limit | 0.0% | [0.0%, 30.0%] |
| Change Operation | 4.6% | [0.2%, 36.8%] |
| Change object/ API | 1.5% | [0.0%, 32.3%] |
| Change the negative polarity | 0.0% | [0.0%, 30.0%] |
| Unreadable/incoherent | 0.7% | [0.0%, 31.1%] |

The type is counted in at least one article of the query, which overlaps and cannot be added to 100%. Zero observations cannot be inferred from the overall absence of this category; For example, the conservative upper bounds of two zero-point estimates remain 30.0%.

(a) For the purpose of determining the denominator of events, a separate three-conditional ratio; The zone is divided by unconditional molecular conservatism by the total number of known incidents, so that it can be more wide and cut to [0, 1 ]:

| Event Total | Total known | Mark Event Ratio | Conservative 95% |
|---|---:|---:|---|
| In a rewritten query once verified, the issue of spelling is shared by both parties | 5890 | 6.0% | [0.1%, 53.0%] |
| RR Upon verification, both sides used rewriting to maintain | 1280 | 86.0% | [50.8%, 100.0%] |
| In inquiries where there is a positive candidate, the best opportunity exists for both parties that have not been fully realized | 3092 | 39.4% | [11.8%, 100.0%] |

Validation of the “both sides maintain” point estimate in the query 86.0% (Conservative compartments) 50.8% – 100% It suggests that all RR losses cannot be directly equated with semantic drift or recognize the respective contribution of search, qrels or learning errors. The non-target-related threads in the display code can also only point to a reference coverage for review and cannot introduce an error rate for the entire text.

## Why are the differences happening?

1. **Default context and new limit.** `set python request headers `→`set python http request headers `: A Considers the addition of HTTP restrictions, B considers it only to clarify the common context in which requests are made. Article 47 has been reworded to A offset/B maintain, with only article 7 in reverse. Manual review is to be judged by “actions, objects, whether or not the restrictions result in a change of answer required”, and cannot be determined by new terms alone.
2. **Unreadable borders with incalculable boundaries.** 16 Article A Non-readable/B cannot be judged to involve unknown or destructive penetration (including UNK). Neither is usually confirmed, but is divided into different categories and affects the proportion of types. Two categories and originals should be retained, clearly identifying the boundaries of “unable to create demand” and “the existence of multiple legitimate needs”.
3. **Function Snippet and context are missing.** Callback `self.with_header `When the code is not displayed, A is satisfied according to the intent of the interface, and B is more often left undecided. 65 Code differences are legible/unable to judge differences; It cannot be considered an error in relevance by a majority vote alone.
4. **The satisfaction of needs is partly related.** `get index of an item in list `a function returns the parameter directly with an integer before the rest is called `lst.index `. A Judgement satisfaction, B indicates that integer elements cannot be generally checked; the return code does indeed have that branch.`repeat sequence `The fragments clearly repeat each element (0, 0, 1, 1, as distinct from the whole sequence (0, 1, 0, 1). These differences, on which the substantive function is based, should not be treated solely as differences in scoring style.
5. **Type names are not mutually exclusive.** `parse mysql get list...`Can be classified by A as an additional limit, and B as a change operation. Two multiple labels are allowed, but A all items are only single master tags. The main/multi-type rule should be discussed after the original judgement had been retained, and not all types of difference should be interpreted directly as differences on whether or not there was a problem.

The researcher identified these topics by checking original wording and static code. They do not constitute a third independent set of judgments. The original rationales and unresolved items are recorded in `item_labels.jsonl` and `DISAGREEMENTS.jsonl`.

## Role of the current paper main line and next steps

These returns complement the exploratory thread of “not all candidate opportunities turned into automatic gains”: there are semantic changes on both sides, as well as RR losses sustained by both parties and positive opportunities for incomplete realization. They do not separate semantic drift, relevance coverage and benefit learning error into cross-repeated mechanisms or explain how much loss of earnings is explained for one reason.

Two final returns were completed. If there is consistency in the independent evaluation prior to review of the body report, recalculations will be required for the original version; For the time being, it should be indicated that there is a “conformity of final returns”. The discussion then proceeded to discuss the differences in clarity between 106 rewriting, 151 items and 35, retaining the original decision and saving the results of the consultations without covering the current statistics without a consistent rate of consultation.

The paper could use the current results as a diagnostic for the designation of a limited overall ex post evaluation exercise to support the discussion of fault lines and mechanisms; Do not write for a unique mechanism to identify or new independent confirmation of effects.

## File & Verify

- `RESULTS.json `: Full source, distribution, confusion table, consistency, weighted query/type ratio and denominator of conditions.
- `DISAGREEMENTS.jsonl `: 106 Rewriting, 151 code, 35 clarity difference, all undecided;`item_labels.jsonl `Saves all 710 double-review entries.
- `weighted_query_rates.csv `: ten weighted search marks;`query_flags.jsonl `Keeps the 128 query event map.
- `received/`and `RECEIPT_MANIFEST.json `: original byte copy and source;`INPUT_VALIDATION.json `The record format validates the statistical version of the final draft.
- `VALIDATION.json `309 Independent hundred/weight/supergeometric cross-checks are passed and the old formal pointer/confirmation is maintained with the previous round of delivery.
