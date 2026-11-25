from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.rene_mechanism_review_20261020.common import ROOT, now, put, read, rows, sha
from .analyze import OUT


LABELS = {
    'any_definite_semantic_issue': 'At least one rewrite has been marked as bias/not readable by both sides',
    'any_possible_semantic_issue': 'At least one rewrite that either side does not maintain',
    'any_uncertain_semantics': 'At least one rewrite is marked as unjudgeable.',
    'verified_action_definite_drift': 'The modification actually applied after verification was marked as a semantic issue by both parties',
    'verified_harm_with_preserved_meaning': 'Upon verification, RR losses were sustained by rewriting the meaning of both parties.',
    'missed_preserved_positive_opportunity': 'The existence of the best positive opportunity for both parties not fully realized and maintained',
    'displayed_alternative_relevant': 'At least one non-existent target display code was found to be met by both parties.',
    'known_target_both_incomplete': 'At least one of the existing target display codes was partially/unsatisfactory on both sides',
    'rater_A_any_semantic_issue': 'A Mark at least one offset/not readable',
    'rater_B_any_semantic_issue': 'B Mark at least one offset/not readable',
}
TYPE_LABELS = {'unsupported_constraint': 'Add a new limit without basis', 'lost_constraint': 'Loss of original limit',
               'operation_change': 'Change Operation', 'entity_api_change': 'Change object/ API',
               'polarity_change': 'Change the negative polarity', 'incoherent': 'Unreadable/incoherent'}


def percent(v):
    return f'{v:.1%}'


def ci(r):
    return '[' + ', '.join(percent(v) for v in r['ci95_conservative_finite_frame']) + ']'


def make_report():
    r = read(OUT / 'RESULTS.json')
    agreement = r['agreement']
    rates = r['weighted_query_rates']
    lines = ['# Double review returns: consistency, weighting and disagreement', '',
        'This report analyses the final returns submitted by both reviewers. User confirmed completion of manual review. Current consistency corresponds to the final return label; The original independent label prior to the review has not been reconciled and, therefore, the consistency of the original independent judgement is not estimated in this report.', '',
        'This time keep the original directory, original zip and item-by-item labels, only return results, without implementing attachments HTML or pending code.', '',
        '## Integrity and the role of evidence', '',
        'Both returns overwhelm the same scheduled 128 query/ 120 group, with different changes to each 383 article and 327 code panel. packet, A/B Identity, collection of entries, label combinations, justification length and sequence approval; The semantic records embedded in each final document are consistent with the first phase of independence JSON. B With two pages HTML identical to the original package.', '',
        'These labels supplement the type of error and mechanism interpretation; Statistics cannot separate semantics, relevance and profit-learning factors as a source of cross-resortment causes and consequences of income gaps.', '',
        '## Consistency and judgement distribution', '',
        'Z Number of items/overlays | Accuracy Convergence Rate Cohen', '|---|---:|---:|---:|',
        f'| Rewrite meaning, 4 Class | 383 | {percent(agreement["semantics"]["agreement"])} | {agreement["semantics"]["kappa"]:.3f} |',
        f'| code relevance, including incalculable | 327 {percent(agreement["relevance_including_unknown"]["agreement"])} | {agreement["relevance_including_unknown"]["kappa"]:.3f} |',
        f'| Codes available to both parties, 0 / 1/ 2 | 260 / 327 (79.5%) | {percent(agreement["relevance_ordinal_judged_subset"]["agreement"])} | {agreement["relevance_ordinal_judged_subset"]["kappa"]:.3f}(linear weights)',
        f'| Level of clarity in original query 128 {percent(agreement["clarity"]["agreement"])} | {agreement["clarity"]["kappa"]:.3f} |', '',
        'Consistency is calculated on the basis of presentation entries and multiple locations of the same text are not counted; It does not represent 8530 overall weighted consistency. The reservation is entirely unjudgeable and does not replace the whole result with a hundred that can be judged higher.', '',
        'Z Rewrite Label A| B|', '|---|---:|---:|',
        '♪ Keep meaning 235 ♪ 275 ♪', '| Meaning deviation | 129 | 73 |', 'Z Unreadable 19 | 3 |', "It's impossible to judge. It's 0, 32...", '',
        '| Code Label | A | B |', '|---|---:|---:|',
        'Z Meet 96 | 34 |', '| Partially meet 151 | 102 |', 'Z 78 | 124 |', "♪ Can't judge ♪ 2  |  67  |", '',
        'Semantic labels 90 are different; Article 106 is rewritten to be discussed, with different types of errors, where 16 is considered by both parties as being biased but not the same type. The code 151 is different, and subparagraph 65 relates to a party that cannot be judged by the other party. B of the code available to both parties has a lower rating than A 82, higher 4. There was also a disagreement on the clarity of the original article 35; Article 124 was considered clear by A and article 94 only by B. There is at least one difference between the total 122/128 query. There was no automatic award or deletion of the disagreement.', '',
        '## Summary by weight of original sample', '',
        'The following denominator is a fixed CodeBERT H general 8530 query, with the indicator " Query contains an evaluation mark/event " and not a candidate error rate. Every one of them. 16 Article, with N_h/n_h weights; Can\'t treat the nudity ratio of 128 as an overall ratio. (a) Inter-situational over-geometric inverse weighting of eight layers within each indicator according to the original scheme Bonferroni is a conservatively limited total area 95%; Not all indicators are claimed to be covered at the same time 95%. The interval reflects only sample error, excluding label error judgement or return version difference.', '',
        '| Query level indicator (permissible for overlap) | weighted estimate | Conservative 95% interval |', '|---|---:|---|']
    for name, value in rates.items():
        lines.append(f'| {LABELS[name]} | {percent(value["estimate"])} | {ci(value)} |')
    lines += ['',
        '35.0% of the “equal issue” and 75.5% of the “any unreserved” use different definitions, which contain incalculable and divergent statistical boundaries that are not the same true error rate. It also shows that the test of “restrictive change” will materially affect the outcome.', '',
        '| The same type of error rewrited is indicated by both parties as the ratio of weighted queries | conservative 95% area |', '|---|---:|---|']
    for tag, label in TYPE_LABELS.items():
        value = r['weighted_query_issue_type_rates']['both_same_rewrite_' + tag]
        lines.append(f'| {label} | {percent(value["estimate"])} | {ci(value)} |')
    lines += ['',
        'The type is counted in at least one article of the query, which overlaps and cannot be added to 100%. Zero observations cannot be inferred from the overall absence of this category; For example, the conservative upper bounds of two zero-point estimates remain 30.0%.', '',
        '(a) For the purpose of determining the denominator of events, a separate three-conditional ratio; The zone is divided by unconditional molecular conservatism by the total number of known incidents, so that it can be more wide and cut to [0, 1 ]:', '',
        '| Total event | Total known | Marked incident ratio | Conservative 95% Area |', '|---|---:|---:|---|']
    for name, label in [('verified_action_definite_drift', 'In a rewritten query once verified, the issue of spelling is shared by both parties'),
                         ('verified_harm_with_preserved_meaning', 'RR Upon verification, both sides used rewriting to maintain'),
                         ('missed_preserved_positive_opportunity', 'In inquiries where there is a positive candidate, the best opportunity exists for both parties that have not been fully realized')]:
        value = r['conditional_query_rates'][name]
        lines.append(f'| {label} | {value["event_frame_n"]} | {percent(value["estimate"])} | {ci(value)} |')
    lines += ['',
        'Validation of the “both sides maintain” point estimate in the query 86.0% (Conservative compartments) 50.8% – 100% It suggests that all RR losses cannot be directly equated with semantic drift or recognize the respective contribution of search, qrels or learning errors. The non-target-related threads in the display code can also only point to a reference coverage for review and cannot introduce an error rate for the entire text.', '',
        '## Why are the differences happening?', '',
        '1. ** Default context and new limit. ** `set python request headers`→ `set python http request headers`: A considers the addition of a new HTTP limit, B considers it only to clarify the common context of the request. Article 47 has been reworded to A offset/B maintain, with only article 7 in reverse. Manual review is to be judged by “actions, objects, whether or not the restrictions result in a change of answer required”, and cannot be determined by new terms alone.',
        '2. ** Unreadable with undecided boundaries. **  16 Article A Non-readable/B cannot be judged to involve unknown or destructive penetration (including UNK). Neither is usually confirmed, but is divided into different categories and affects the proportion of types. Two categories and originals should be retained, clearly identifying the boundaries of “unable to create demand” and “the existence of multiple legitimate needs”.',
        '3. ** The functional segment and context are missing. ** For unshowed codes such as call `self.with_header`, A is intended to be satisfied by the interface, and B is more often left unjudgeable. 65 Code differences are legible/unable to judge differences; It cannot be considered an error in relevance by a majority vote alone.',
        '4. ** The satisfaction of needs is partly related. ** A function `get index of an item in list` encounters an integer to return the parameter directly and the rest calls `lst.index`. A Judgement satisfaction, B indicates that integer elements cannot be generally checked; Return code does have the branch. The other example `repeat sequence` explicitly repeats each element (0, 0, 1, 1, as distinct from the whole sequence (0, 1, 0, 1. These differences, on which the substantive function is based, should not be treated solely as differences in scoring style.',
        '5. ** The name of the type is not mutually exclusive. ** `parse mysql get list... `may be classified by A as an additional limitation, and B as a change operation. Two multiple labels are allowed, but A all items are only single master tags. The main/multi-type rule should be discussed after the original judgement had been retained, and not all types of difference should be interpreted directly as differences on whether or not there was a problem.', '',
        'The above-mentioned topic of discussion for researchers after checking the original paragraph with static code is not a third independent expert label. Original item-by-item and outstanding items are stored in `item_labels.jsonl` and `DISAGREEMENTS.jsonl`.', '',
        '## Role of the current paper main line and next steps', '',
        'These returns complement the exploratory thread of “not all candidate opportunities turned into automatic gains”: there are semantic changes on both sides, as well as RR losses sustained by both parties and positive opportunities for incomplete realization. They do not separate semantic drift, relevance coverage and benefit learning error into cross-repeated mechanisms or explain how much loss of earnings is explained for one reason.', '',
        'Two final returns were completed. If there is consistency in the independent evaluation prior to review of the body report, recalculations will be required for the original version; For the time being, it should be indicated that there is a “conformity of final returns”. The discussion then proceeded to discuss the differences in clarity between 106 rewriting, 151 items and 35, retaining the original decision and saving the results of the consultations without covering the current statistics without a consistent rate of consultation.', '',
        'The paper could use the current results as a diagnostic for the designation of a limited overall ex post evaluation exercise to support the discussion of fault lines and mechanisms; Do not write for a unique mechanism to identify or new independent confirmation of effects.', '',
        '## File & Verify', '',
        '- `RESULTS.json`: Full source, distribution, confusion table, consistency, weighted query/type ratio and denominator of conditions.',
        '- `DISAGREEMENTS.jsonl`: 106 Rewrite, 151 Code, 35 Clarity difference, all undecided; `item_labels.jsonl` Saves all 710 double rating entries.',
        '- `weighted_query_rates.csv`: ten weighted search marks; `query_flags.jsonl` Retains 128 query event mapping.',
        '- `received/` and `RECEIPT_MANIFEST.json`: original byte copy and source; `INPUT_VALIDATION.json` Recording the validation and final return of the statistical version of the draft.',
        '- `VALIDATION.json`: 309 Independent hundred/weighted/supergeometric cross-checks are passed and the old formal pointer/confirmation is maintained with the previous round of delivery.', '',
    ]
    with (OUT / 'FINDINGS.md').open('w', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    summary = '\n'.join([
        '# Agreement and weighted diagnostics of the final review returns', '',
        'The researcher reports that two reviewers completed the human assessment. We analyze their submitted final returns, covering 128 sampled queries (120 groups), 383 unique rewrites and 327 code panels per reviewer. Agreement estimates describe these final returned labels; original independent labels before subsequent review have not been reconstructed, so the estimates should not be presented as original independent-assessment agreement.', '',
        f'The semantic exact agreement was 76.5% (nominal Cohen κ={agreement["semantics"]["kappa"]:.3f}). Code relevance agreement including unknown was 53.8% (κ={agreement["relevance_including_unknown"]["kappa"]:.3f}); the jointly judged 0/1/2 subset covered 260/327 panels (79.5%) with linear-weighted κ={agreement["relevance_ordinal_judged_subset"]["kappa"]:.3f}. Unknown code labels were 2 for A and 67 for B. Semantic classes differed for 90 rewrites; including issue types, 106 rewrites differed. Code labels differed for 151 panels. All submitted judgments and disagreements were retained without automated adjudication.', '',
        'Using the original eight-stratum query weights for the fixed H8530 frame, the query proportion with at least one rewrite marked drift/malformed by both raters was 35.0% (conservative finite-frame 95% CI 9.9%–69.6%). The proportion with any rewrite not marked preserved by at least one rater was 75.5% (39.6%–95.4%), which includes unknown and disagreements. The weighted proportion with RR loss after verification while both raters marked the adopted rewrite preserved was 12.9% (7.6%–40.3%); displayed non-target code marked fully relevant by both occurred in 12.0% (1.3%–46.2%). These overlapping indicators describe marked query events, not a causal decomposition of retrieval loss. Intervals reflect finite-frame sampling alone and omit label error and review-version uncertainty.', '',
        'The returns expose differences in how added wording, query ambiguity, missing code context and partial functionality are treated. They support a post-hoc diagnostic discussion without identifying unique causal contributions. Raw materials, per-item disagreements, weights and numerical validation are recorded in results/rene_review_returns_20261020/.', '',
    ])
    paper = ROOT / 'paper/rene2027/mechanism_review_return_results.md'
    with paper.open('w', encoding='utf-8') as stream:
        stream.write(summary)
    # Read-back verifies the public CSV against the fixed result.
    with (OUT / 'weighted_query_rates.csv').open(encoding='utf-8-sig', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    for row in csv_rows:
        v = rates[row['indicator']]
        if float(row['estimate']) != v['estimate'] or [float(row['ci95_low']), float(row['ci95_high'])] != v['ci95_conservative_finite_frame']:
            raise AssertionError('Export differs')
    validation = read(OUT / 'VALIDATION.json')
    for item in validation['protected_files']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise AssertionError('Prior protected file changed')
    if sha(OUT / 'RESULTS.json') != validation['results_sha256']:
        raise AssertionError('Scientific result changed')
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in ('DELIVERY_MANIFEST.json', 'FINAL_ACCEPTANCE.json')) + list(Path(__file__).parent.glob('*.py')) + [paper]
    manifest = {'created_utc': now(), 'files': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in files]}
    with (OUT / 'DELIVERY_MANIFEST.json').open('w', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    for item in manifest['files']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise AssertionError('Delivery file changed')
    acceptance = {'created_utc': now(), 'passed': True,
        'role': r['role'], 'human_review_reported_by_user': True,
        'review_return_analysis_complete': True, 'agreement_version': 'final_submitted_returns',
        'original_independent_agreement_established': False,
        'reviews_analyzed': 2, 'complete_queries': 128, 'rewrite_items': 383, 'code_panels': 327,
        'unresolved_item_disagreements': 257, 'unresolved_clarity_disagreements': 35,
        'independent_numeric_checks': validation['independent_kappa_estimate_and_interval_checks'],
        'originals_and_protected_states_unchanged': True, 'csv_checked': len(csv_rows),
        'manifest_files': len(files), 'manifest_sha256': sha(OUT / 'DELIVERY_MANIFEST.json'),
        'results_sha256': sha(OUT / 'RESULTS.json'), 'no_causal_attribution': True}
    with (OUT / 'FINAL_ACCEPTANCE.json').open('w', encoding='utf-8') as stream:
        json.dump(acceptance, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print({'report': str(OUT / 'FINDINGS.md'), 'manifest_files': len(files), 'agreement_version': 'final_submitted_returns'})


if __name__ == '__main__':
    make_report()
