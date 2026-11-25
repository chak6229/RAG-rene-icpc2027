# CoSQA Data and SSQR Recurring review

Date: 2026 - 10 - 19. Task: Check the source of the query, associated labels, permits, data overlaps, author code and checkpoint and determine whether it supports RENE main line.

**Conclusion: CoSQA can serve as an external replica of SSQR data on new query sources. Adopts the author ' s 6,267 definition of document retrieval language and single target relevance; The migration which advocates limiting the search to real Web cannot be written as a completely uncoded authentication. The current local process requires the creation of an independent SSQR alignment branch, which cannot run directly as the original author.**

This review of data and sources, full-scale overload scanning, actual tokenizer inspections, registration of weighting identification and training steps, achievement of discrepancy tables and consistent acceptances has been completed. No generation, retrieval rankings, model training or methodological impact assessment on run CoSQA; The relevant labels have been used for the source review and cannot be described as never visited tags. Official CURRENT/STABLE 5 protection files remain the same as Hashi.

## 1. Data version and real source

| Source | Fixed Version / Evidence |
|---|---|
| CoSQA / CoCLR Author repository | `Jun-jie-Huang/CoCLR@14ebcacf9e9bc3e7109102632bc63047876f27d2` |
| Microsoft Distribution | `microsoft/CodeXGLUE@ac74a62802a0dd159b3258c78a2df8ad36cdf2b9` |
| CodeSearchNet Source | `github/CodeSearchNet@106e827405c968597da938f6b373d30183918869` |
| SSQR Author repository | `RedSmallPanda/SSQR@7fe5d780740eb4348500658ce6decf188a542d2b` |
| C-UDA Official text | `microsoft/Computational-Use-of-Data-Agreement@a28ca06f6f8ecac0b5856ca6179ac49e55f00104` |

The three local CoSQA QA documents are identical to the Git blob version of the CoCLR author and the CodeXGLUE distribution version. Co-check 15 local CodeXGLUE documents, 21 selected author source/data files and 3 official C-UDA texts; Original files, Git blob SHA-1 and SHA-256 have been retained. CoCLR dev500 Initial network reading failed, followed by original blob interface successfully obtained and validated, with final status `VERIFICATION.json `;the original failure record is kept.

CoSQA Original paper § 3 – 4, § 6.1

1. Query from Microsoft Bing search log, keep a query containing 'python' and filter it by code.
2. The code is derived from the Python function of CodeSearchNet, visible functions and their documents when indicated.
3. Candidate collects CodeBERT, which is fine-tuned above 148K StaQC and selects Top-1 code with at least 0.5 similarity for queries; The frequency of occurrence of the same code is limited.
4. For each pair of at least 3 notationists, there were more than 100 participants; Removes low-consistency and non-coded search intended samples, with the final average Krippendorff alpha being 0.63.

That's why it answered:**The real Web query that is filtered through a code search for intent and CodeBERT candidate collection**Problem. It is not a random sample of any developer or click on the successful log. Cross-researcher results need to be interpreted in conjunction with this collection.

## 2. Relevance and retrieval architecture

### 2.1 The two tasks must be separated.

| File / Cut | Query - Code Correct | - Yes. | Negative | Different Codes |
|---|---:|---:|---:|---:|
| QA all | 20,604 | 10,020 | 10,584 | 6,267 |
| QA train | 20,000 | 9,707 | 10,293 | 6,184 |
| QA dev | 604 | 313 | 291 | 552 |
| Official retrieval dev | 500 | 500 | 0 | 478 Targets |
| Official retrieval test | 500 | 500 | 0 | 472 Targets |

20,604 The original query is the only one under the current normative rule, without duplicate pairing, conflict labels or multiple formal inquiries. QA all Exactly equals QA train + QA dev.**QA 20,000 / 604 cut cannot be used as a development/test split for retrieval.**

Label 0 indicates only that the pairing code does not answer the query; It does not mean that the query has no answer in the entire text. No formal 10,584 query can be directly used to calculate a meaningful target RR. Other unlabelled queries - code combinations are uncalculated items that cannot be referred to as artificially confirmed unrelated documents.

The author also provided access to dev500/test500 and `code_idx_map.txt `. The following article by article has been verified:

- The text(s) correspond to 6,267 different functions, ID is 0 – 6266 consecutively, identical to the local CoSQA function collection.
- Two search slices of all label = 1; each line `retrieval_idx` Language equal to its target function ID.
- Original field by item equals QA all corresponding records; dev/test Query intersection to 0.
- The different target functions for the official test are actually: **472**; the 473 in the source note is not used as a statistical basis.

Retrieval of 19,604 rows of train by all minus the official 500 rows, including 9,020 positive examples and 10,584.**No bytes downloaded and verified for original search train JSON**, therefore this item only states that the line identity and labels are consistent.

It is recommended that the official 6,267 document be retained as a common medium. CoSQA The original assessment was the full ranking target RR; SSQR The public result is exported to a maximum of 100 answers. The new experiment should be fixed cutoff at the same time: to generate clear MRR@100 and full rankings MRR from a complete ranking, the main indicator is written into the protocol before the effects are calculated and cannot be mixed. best-of-3 of old SSQR Table 4 does not contain original queries back; Automatically adopt, integrate and include oracle breakdowns.

### 2.2 Key fields in data access

| Fields | The meaning of this verification | Access requirements |
|---|---|---|
| `doc` | Bing Web Question | T5 and original query used by the searcher |
| `code` | Full function, usually containing document | Keep author text ID; do not default on pure code indexing text |
| `code_tokens` | Supplyed code token string | Use this string directly when the code for alignment SSQR indicates |
| `docstring_tokens` | Document matching function token string | Not Web Query, not entered as a query |
| `label` / `retrieval_idx` | Match Relevance / Retrieval Targets ID | Only enter labels/assessment channels |

CoCLR `code_search/run_siamese_test.py:68–88` Use same `js['doc']` As a query. 20,604 None of the rows `doc` and `docstring_tokens` Same after a blank; the latter 20,487 row is not empty. `generate_codexglue_rewrites.py` Priority `docstring `Again. `docstring_tokens` When the list is folded,**Direct application will spell the target function document into a character-by-character, not for CoSQA**.

Same function `code_tokens` There is complete consistency between the records, and 6,267 index strings are also unique. It is recommended that the code token index be used primarily for comparison purposes; If you run another CoCLR original index containing the document, you need to be shown as a data expression condition and not mixed with the SSQR code index. The associated label still corresponds to the same function ID.

### 2.3 Actual tokenizer Review

Inspections have been conducted using local actual T5TokenizerFast and author source categories T5Tokenizer; The latter was reconstructed using a local t5-base SetencePiece vocabulary and did not pretend to restore the author tokenizer. The current directory best contains only fast tokenizer files and directly loads slow tokenizer failed; Reasons have been documented and no environmental or model catalogue has been changed.

- All 20,604 entries, original number of words 4 – 21; T5 Pure search maximum 35 tokens.
- Thesis n+1 **156,603** Sentry input; open JSON source code 1... n **135,999** A few.
- The maximum input for both tokenizer entries is **36 tokens**; under the length set by 128 or 512, no sentry was missing or entered to cut off.
- CodeBERT Maximum query 23 tokens. Code token Index at 256 limit **27/6,267** sector is interrupted; if the direct index contains raw code of a document, then **287/6,267**. The cut-off rule must be fixed with the index configuration.

These are input and terminological checks that have not yet been tested for output quality.

## 3. Permission check and artifact processing

CoCLR Authors README state unequivocally:**Code MIT, Data Computational Use of Data Agreement (CUDA)**CodeXGLUE Root directory MIT cannot remark CoSQA data as MIT.

The text of v1.0 has been retrieved and verified from Microsoft's official C-UDA repository. Its core provisions are:

- § 1.1, § 2.1: Allows the use, modification and distribution of data according to terms limited to computational use.
- § 3.1: Redistributes data while retaining the signature/attribution information received and binds downstream receivers to C-UDA.
- § 2.2, § 5.5: The analysis that does not contain more than trace data, the defined model, etc. belongs to Results and is not subject to any additional distribution restrictions.

CoCLR Write only CUDA names, with no clear version number or full Hashi; This time, the official v1.0 text corresponding to that name is saved without stating that the author has attached the version verbatim. Current scientific calculations are compatible with its declarations.

CodeSearchNet README also indicates that the code is the original licence for data in each language `_licenses.pkl` Medium; Local import directory not found corresponding sidecar. SSQR The repository does not have a unified root LICENSE, partially inheriting the file belt MIT head. This information is retained separately at the time of publication.

The specific treatment of this item artifact is:**Publication of our appropriate/assessment codes, data acquisition instructions, source links, division ID and Hash; Third-party data and original author ' s source code are obtained by source from the acquisition script.** If the original data is subsequently packaged, the data terms and attribution must be kept with the package and cannot be posted uniformly MIT. There is no need to suspend the current read-only review or scientific calculations.

## 4. Overlap with training and consumption data

Scan has been completed and contains original code, retention of case-based blanks into one code, existing code tokens three layers; Query using NFKC, case and blank combinations and adding Unicode word fingerprints. The following "overlapping" is a combination of these fingerprints.**There is no claim that semantic cloning or all basic model pre-training exposures have been excluded**.

| Source of Control | Number of rows | Hit CoSQA Different code | Query with these codes as positive targets | Normative/word query repetition |
|---|---:|---:|---:|---:|
| Local CodeXGLUE Retrieval calibration train | 251,820 | **4,011** | **6,585** | 0 / 0 |
| CodeSearchNet Python Original train | 412,178 | 5,404 | 8,743 | 0 / 0 |
| CodeSearchNet Python Original valid | 23,107 | 335 | 532 | 0 / 0 |
| CodeSearchNet Python Original test | 22,176 | 305 | 493 | 0 / 0 |
| Local AdvTest valid | 9,604 | 0 | 0 | 0 / 0 |
| AdvTest test | 19,210 | 0 | 0 | 0 / 0 |
| Original AdvTest test_code | 45,244 | 0 | 0 | — |
| Local T5 CODEnn train | 1,000,000 | not applicable | not applicable | 0 / 0 |
| Local T5 CODEnn valid | 10,000 | not applicable | not applicable | 0 / 0 |

AdvTest Contains anonymization of the identifier, and a fingerprint 0 cannot be pushed further into semantics. The number of code hits for each CSN cut cannot be simply added to the total of different functions.

Historical anonymization language has been returned to the original function by source index, URL or repository+query:

| Consumed against | Hit code. | Query on hit target | Repeat queries |
|---|---:|---:|---:|
| History action 4096 | 66 | 95 | No checkup of the pool |
| History semantic 4096 | 59 | 90 | No checkup of the pool |
| Strict confirmation 4096 | **88** | **128** | Not applicable: document language |
| Strictly confirm 1536 Query Target | 28 | 34 | 0 |
| 1024 Queryed Target | 25 | 37 | 0 |

This supports the external recovery design of "requiring sources have changed", while requiring that code exposure layers be retained. Article 10,020 has a known regular query to fine-tune the exposure by local searcher as follows:

| Layer | Number of active queries |
|---|---:|
| Target code matches local fine-tuning track | 6,585 |
| Target code does not match, but source repository is fine-tuned train | 2,087 |
| Neither the target nor the recovered repository showed up locally fine-tuned track | **1,096** |
| No matched target and repository not recovered | 252 |

These are...**Local fine-tuned exposure layer**, the base model cannot be changed is completely unobserved. CoSQA In official test500, the target code for 317 query matches local fine-tuning train.

## 5. Grouping and research adaptation

Of 6,267 code snippets, **6,044** were mapped to source repositories, covering 2,933 repositories. The remaining **223** were unresolved. Connected components based on code identity and recovered repositories yielded 3,156 groups across all queries and 2,475 groups containing a positive query. The largest positive-query group contains 137 queries. Unmapped repositories are grouped by code identity alone, leaving repository independence unestablished for those items.

| Official access cut-off. | Cross-Query | Cross-Code | Intersection of connected groups restored |
|---|---:|---:|---:|
| dev500 / test500 | 0 | **70** | **108** |
| train Wizard Identity / dev500 | 0 | 408 | 390 |
| train Wizard Identity / test500 | 0 | 396 | 371 |

The official cut is therefore appropriate for the “new queries” assessment and cannot be considered as a direct isolation of the new repository. In the case of a fixed official test500, it involves 396 active query connectivity groups; After excluding these groups, there is still one. **6,490 Article Query / 2,079 Group** Available for development design, where 6,282 /1,949 groups have restored repositories. Compares to excluding only the test 500 rows, group isolation removes another 3,030 query.

This audit assessed sample availability before assigning D/H roles, choosing thresholds or establishing power. The 500 confirmation rows do not by themselves ensure power to detect small increments, test equivalence or certify rare harms.

| Research issues | How this data is supported | It has to be fixed. |
|---|---|---|
| RQ1: SSQR ENTR Relative RAND/PROB Can you move? | The same new query, the same 6,267 document, the same position candidate cache | Tests for ranking returns of individual stated targets; original best-of-3 separated from automatic selection |
| RQ2: How much proceeds can be achieved through automatic adoption? | Adopt a line-by-line calculation target for original sentence, candidate, validation and risk | Damage is defined as a decline in the ranking of the stated target, which does not extend to the harm caused by artificial confirmation; downstream alignment and recognition role grouping are isolated |
| RQ3: How should prior UQ evidence be interpreted? | The new distribution can test the same prespecified signal. Previous positive or conditionally equivalent findings retain their original scope. | The data provide no ground truth for epistemic or aleatoric uncertainty. An association with entropy does not validate disentanglement. |
| RQ4: Reliance on search mechanisms? | CodeBERT, BM25 to share query, function ID, code index text and qrels | The interpretation of the result must contain CodeBERT candidate collection preferences; Change checkpoint does not amount to a second search mechanism |

**Data acceptance decision: Use CoSQA as the preferred source for new queries; main questions are limited to search migration and single target retrieval proceeds.** The code does not need to be discarded because it comes from the same upstream, but is designed to retain exposure layers and clusters.

## 6. SSQR Source code, paper and local differences

Author checkpoint is not simply “not found in this round”. [author at]issue#1](https://github.com/RedSmallPanda/SSQR/issues/1#issuecomment-2007139087)(2024 - 03 - 19) Make it clear that checkpoint cannot be provided after graduation and can be retrained on the basis of data. The current complete repository tree does not have weight, and the list release is empty. README The link to Google Drive is pre-trained data, not weight; The local visit was time-consuming and did not render it permanently invalid.

| Item | Thesis / Author Open Source | Current local realization | Formal experimental processing |
|---|---|---|---|
| Weighted identity | The original weight is no longer available to the author. | Local independent training T5/ CodeBERT | Marked as an independent recurrence, recording the actual weight of Hashi |
| T5 Training data | Dissertation CODEnn ex 1M; source reading `train_context.csv` and clean up/sub-paragraphs | Local HDF5 Decoding 1 M, followed by 10 K for fixed authentication | Retain source, sequence and treatment of differences; no author CSV byte, not claim to be identical |
| CQC Cover | Series of papers 15%; source code `ceil(.15*n/3)` 3 length span, much sentinel | Single Continuous `round(.15*n)` span | Locally closer to paper description; alignment contracts need to indicate the use of the paper or source code and cannot be consistent |
| T5 Length | dissertation called default length; public README/source input output average 128 | 512 | Registration of training differences;this CoSQA query input is not affected by this cut |
| T5 Optimizer | Open Source Adafactor | AdamW | The two are different; the current weight can only be referred to as local training |
| T5 Learning rate | Source linear, mandatory `t_total=100000 `, warmup=0 | constant, 93,750 steps | Write source differences, not only initial lr |
| T5 Random Feeds | Thesis 101; source code available `seed=101` Configure, end of actual call `set_seed(42)` | 101 | Training RNG Disaggregated from reasoning RNG; source conflict visible registration, not only configuration fields |
| T5 checkpoint Select | Original weight/real choice not recoverable | Training completed 3 epoch; Use best step92, 000, final step93, 750 Save | Distinguishing best and final; local early stop rules are not triggered and cannot be written as early cessation of training |
| Insert Location | Thesis n+1 (0 n); Open JSON Path 1... n | Local 0... n | Formal master settings and source sensitivity are fixed before effects can be seen and co-located caches can be used |
| Decoding & Range m | Thesis describes sampling, m = 10; source code only `model.generate(input_ids)` | Local sampling, top-k50 maximum 12 New tokens Equivalent Configuration | Author trained config Non-recoverable; recording of the selected independence cannot be assumed to call author equals local sampling |
| ENTR Distribution | (a) The distribution of the full vocabulary of the papers; Mr. Source, again teacher forcing, full vocabulary softmax counts entropy | Local history generator `generation.scores` Post-processing distribution | **Formal negative entropy must resume full wordsheet teacher-forcing rating**;warper Subsequent entropy is named separately |
| ENTR Average position | Source Repackage `<extra_id_0> span <extra_id_1>`, with special tab position | Local exclusion segment special token, different process | Freezing Average mask; Sphere span Distinguished from the range containing sentinel/EOS |
| Entropy direction | Thesis and main JSON path to the smallest entropy; another unused text helper to the maximum entropy | Locally by negative entropy | Based on main path; not miscalculating uncalled helper as the author's master algorithm counter |
| PROB / RAND | Table 4 Description of existence, public generator does not include corresponding realization | History process does not complete the exact author's version | Independence to achieve both and freeze PROB aggregation definitions, RAND RNG; ambiguity predefined sensitivity not selected by result |
| Query Input | SSQR Original data `docstring_tokens` is the list of queries token | History priority raw docstring; CoSQA The same name segment is the target document string | New CoSQA adapter Visible extraction `doc `Disable old field retreat |
| CodeBERT Vector / Loss | Public `model.py` with pooler output, unintegrated point accumulation, batch CE | Local training and reasoning integration CLS / point accumulation | **Only inference exchange pooler does not constitute a faithful recurrence; the main baseline requires matching training targets** |
| CodeBERT Training completion | Articles 2 epoch; README batch32、block256、seed123456 | Current direction step3, 000; other model completed 15,740 steps | Two checkpoint have been registered; The complete training model remains CLS target and cannot be automatically upgraded to an author. |
| CodeBERT Learning rate | Articles constant 5e-5; Open code linear warmup/decay (10% warmup) | Local condition 5e-5 | Thesis does not match the source code itself; the main set is on protocol selection and the other is sensitive as necessary |
| Ranking, Cutting and tie | Author code first sorted and then press `topK` interval export, evaluation of given answer sequence; original production tie not recoverable | Historic modules have CLS/pooler, MRR@100 different paths to ranking | New uniform evaluation function ID, stable tie and cutoff, save full ranking/target rank, cannot assume the same indicator after copying out details |
| Operation Dependence and Repair | Old transformers/Lightning; missing unused module import;`args.ouput` Spelling error | Modern transformers/PyTorch | Only necessary compatibility repairs in the new branch to preserve original files and patches; no recovery of post-source code impersonating original release documents |

Another open training realization problem is that short text span samples may not have a viable location; The author training script was not implemented this time, and the follow-up should be preceded by a feasibility border verification. The author is also in issue # 5](https://github.com/RedSmallPanda/SSQR/issues/5#issuecomment-2488160495) Confirms that the missing unused import can be deleted.

This table defines the conditions for realization of an external recurrence without denying the relative effects of historical comparisons with processes; Historical results continue to retain their original data and scope of processing.

## 7. Local checkpoint Actual identity

| Model | Actual Status | Weights SHA-256 |
|---|---|---|
| T5 best | step92, 000; training operation finalised 93,750 steps / 3 epochs | `7acd4cb8379a1f5297bb978a748bc002b708ef5fd60a8b4737a6f8c7d2ebcaa5` |
| T5 final | checkpoint-93750 | `b543ed2878dfebe8c36acb87564d73ea50f8e05b98d6aa77d3b68785364c3335` |
| Current CodeBERT | checkpoint-3000; planned 15,740 steps, actual only 3,000 | `c3bb2df27ddc777d5f0d6fb22525ca74cfc1c19b459630f60df2fddc97e42bd1` |
| Full Local CodeBERT | checkpoint-15740; two rounds of training completed; still targeted for integration CLS | `802943bd77ea247fbe92e1ae2d26f520ef2a08ecdd79b76e1292f8de1a6da903` |

CodeBERT Full running of a separate best_step8500 so that the model root directory, best and final cannot be swapped by only a directory name. Base CodeBERT The actual file is `pytorch_model.bin `His Hash has been replaced. `INPUT_CONTRACT.json `;does not value the non-existent safetensors as a base model lost.

## 8. Acceptance and next steps

**The review of the source was completed; the data route was adopted. The formal release is not yet in place, as alignment and new statistical agreements have not been implemented.**

`VERIFICATION.json` The 91 conformity check was approved, covering: author Git blob, local copy, C-UDA text, sticker maintenance, official target ID, article-by-article fingerprinting, code/storage group consistency, exposure layer, recorded tokenizer input contract and 5 formal compass. Big data entry has been recorded at the time of the initial full-scale scan, and this acceptance does not duplicate complete scanning or calculate any method gain.

The next steps are set out in the following order:

1. CoSQA adapter and SSQR repeat entrance: query =`doc `, syntax ID fixed, ENTR full vocabulary teacher forcing; Cachebreeding at the same location ENTR, PROB, RAND. Completing contractual checks to achieve differences before entering a small number of development queries.
2. Prior to new data results, fixed paper or source-led reconstruction options were available. Clearer with papers n+1, m10, k3; Discrepancies between open source domains, covers, optimizers and seeds are registered separately. CodeBERT Master baseline shall complete pooler+ matching training for non-integrated point accumulation; Existing CLS models are first used for migration/compatibility diagnostics and cannot be directly labelled as author duplicates. The training budget is based on realistic estimates before implementation.
3. Draft group roles and efficacy without method results, then run CodeBERT/BM25 on small batch queries permanently assigned to D. Determination of the size of confirmation based on D deviations, stand-alone arrays and number of events; No official test500 manoeuvre will be allowed to claim a one-time confirmation.

strong84, gain-priority, threeway, semantic-pairwise UQ or old models;`risk_certified=false` Maintained with NoRewrite deployment status. The adoption of the review does not represent a return on positive gains, nor does it mean that new risk strategies have been adopted.

## 9. Evidence portal

- `DATASET_STATISTICS.json `: Full pairing, relevance, repetition and QA split statistics.
- `RETRIEVAL_CONTRACT.json `: Author retrieves text, qrels, cutting overlap and local fine-tuning of the exposure layer.
- `INPUT_CONTRACT.json `: field synonyms, actual tokenizer, sentry, code cut and tokenizer Hash.
- `OVERLAP_AUDIT.json` / `OVERLAP_SUMMARY.json `: Complete input list, layered overlap and summary.
- `QUERY_GROUP_INVENTORY.jsonl` / `MATCHED_SOURCE_RECORDS.jsonl `: Query groups and hit-source rows.
- `ROLE_FEASIBILITY.json `: Usage after excluding official test associated groups is not frozen D/H.
- `LOCAL_CHECKPOINTS.json `Weights, training completion status, metadata and official pointer snapshots.
- `CUDA_LICENSE_SOURCES.json `Official provisions and fixed versions; network failure and subsequent success respectively.
- `VERIFICATION.json` / `AUDIT_DECISION.json` / `DATA_ACCESS_LEDGER.json` / `AUDIT_DELIVERY.json `(b) Receiving and inspection, conclusions, scope of visits and delivery to Hashi.
- `selected_source/`: Git blob Validation of author files;`sources/`: Open API with dissertation snapshot.

Source: CoSQA Papers](https://aclanthology.org/2021.acl-long.442/)、[CoCLR](https://github.com/Jun-jie-Huang/CoCLR/tree/14ebcacf9e9bc3e7109102632bc63047876f27d2)SSQR Papers](https://arxiv.org/html/2307.00267)SSQR Source code](https://github.com/RedSmallPanda/SSQR/tree/7fe5d780740eb4348500658ce6decf188a542d2b)C-UDA v1.0 Text](https://github.com/microsoft/Computational-Use-of-Data-Agreement/blob/a28ca06f6f8ecac0b5856ca6179ac49e55f00104/C-UDA-1.0.md)CodeSearchNet Licence description](https://github.com/github/CodeSearchNet/tree/106e827405c968597da938f6b373d30183918869#licenses).
