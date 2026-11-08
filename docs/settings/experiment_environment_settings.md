# Experimental environment and settings

Tables S16–S25. The PDF and Word versions are in supplementary/. Exact original settings and source identities remain in the adjacent JSON file.

## Table S16. Software and hardware environment

| Item | Configuration | Source |
| --- | --- | --- |
| Operating system | Windows; runtime identifier Windows-10-10.0.26200-SP0 (build 26200) | R01, R02 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU; 8,151 MiB memory recorded at runtime | R01, R02 |
| GPU driver | 577.05 | R01 |
| CPU | AMD Ryzen 9 8945HX with Radeon Graphics; 16 cores and 32 logical processors (hardware recheck) | R34 |
| Memory | 32 GiB (hardware recheck) | R34 |
| Python | 3.10.11, 64-bit | R01, R02 |
| PyTorch / CUDA | 2.7.0+cu128 / 12.8 (PyTorch build and runtime) | R01, R02 |
| Transformers / Tokenizers | 4.57.6 / 0.22.2 | R01 |
| SentencePiece / Safetensors | 0.2.2 / 0.8.0 | R01 |
| NumPy / SciPy | 2.2.6 / 1.15.3 | R01, R02 |
| Scikit-learn / Joblib | 1.7.2 / 1.6.0 | R07, R09 |
| Threadpoolctl / Matplotlib | 3.6.0 / 3.10.9 | R01 |
| Training precision | Main T5, CodeBERT and limited-training controls use BF16; limited-control model weights are FP32 | R03, R05, R15 |
| Generation and retrieval precision | Main T5 generation/scoring and neural retrieval use FP32 with TF32 disabled; cross-encoder verification uses BF16 autocast and converts outputs to FP32 | R02, R06, R20 |
| Inference state | Models use eval; dropout is disabled for generation, scoring and retrieval; training retains each model's training configuration | R17, R20, R31 |

Software, GPU and driver values come from historical run records. CPU and memory were checked when the original settings supplement was compiled. CUDA 12.8 denotes the runtime used by PyTorch.

## Table S17. Training settings for the rewriter, neural retriever and verifier

| Parameter | Local CQC T5 | CodeBERT retriever | Cross-encoder verifier |
| --- | --- | --- | --- |
| Base model | T5-base | CodeBERT-base | CodeBERT-base |
| Training data | First 1,000,000 comments in the local CODEnn source | 251,820 CodeXGLUE query-code pairs | Historical large_dense_4096: 4,096 queries, 16,384 pairs |
| Objective | Recover one contiguous masked span of length max(1, round(0.15n)) | Full query-to-code in-batch cross-entropy | Binary relevance BCEWithLogits |
| Epochs / updates | 3 / 93,750 | 2 / 15,740 | 1 / 512 |
| Microbatch / accumulation | 8 / 4 | 32 / 1 | 8 / 4 |
| Effective batch | 32 | 32 | 32 |
| Optimizer | AdamW | AdamW | AdamW |
| Learning rate | 0.001 | 0.00005 | 0.00002 |
| Schedule | Constant | 10% warmup (1,574 updates), then linear decay | Constant |
| Weight decay | 0.01 (execution-code default) | 0 | 0.01 |
| Gradient clipping | L2 norm capped at 1 | L2 norm capped at 1 | L2 norm capped at 1 |
| Input length | 512 | 256 each for query and code | 64 query / 160 code, plus special tokens |
| Seed | 101 | 123456 | 141414 |
| Weight selection | Best source-validation loss, update 92,000 | Prespecified final weights after two epochs | Historical completed-training weights, frozen before H inference |
| Training precision | BF16 | BF16 | BF16 autocast |

Sources: T5, R03/R04/R29; CodeBERT, R05; verifier, R27/R28. The verifier retains historical training weights. Its data and negative construction remain in the source code; it was not retrained using H labels.
Main T5 training completed 93,750 updates. Generation uses the source-validation-best weights at update 92,000. This training is separate from the eight 1,024-update sensitivity configurations.

## Table S18. Generation, scoring, selection and retrieval settings

| Stage | Setting | Source |
| --- | --- | --- |
| Main insertion positions | Split into n whitespace-delimited words; insert after each word from 1 through n; generate 1 candidate per position | R06, R17 |
| Main generation | greedy; do_sample=false; num_beams=1; num_return_sequences=1; max_new_tokens=19 | R06, R14 |
| Length convention | Decoder length at most 20 including the start token; inputs longer than 512 raise an error rather than being silently truncated | R17 |
| Batch sizes and seed | Early/CoSQA/limited-training generation: 8; StaQC and new paired-weight generation: batch 32 and query block 32; seed 101 throughout | R06, R07, R11, R14, R16 |
| T5 tokenization | T5Tokenizer legacy=True; local T5-base SentencePiece; saved tokenizer vocabulary checked | R17 |
| ENTR / PROB | Same generated label sequence; ENTR averages full-vocabulary predictive entropy over non-padding positions (lower first); PROB averages token log probabilities (higher first) | R17 |
| Scoring labels | <extra_id_0> + span + <extra_id_1>, including EOS; break score ties by original insertion position | R17 |
| Selection and aggregation | k=1,2,3, capped at pool size; mean / label-assisted best-k; no additional original-query option | R26, R14 |
| Duplicates and empty spans | Retain original positions and their selection probability; do not pad short pools with the original query | R11, R14 |
| Exact random / RAND20 | Exact random is the uniform expectation over positions sampled without replacement; RAND20 seeds 2026101900–2026101919 | R06, R07, R26 |
| CodeBERT representation | pooler_output; no normalization; FP32 raw dot product; query and code lengths capped at 256 | R05, R11 |
| CodeBERT batch sizes | CoSQA H encoding: 16; new paired-weight StaQC encoding: 32; ranking score block: 128 | R20, R14 |
| BM25 | k1=1.2, b=0.75; IDF=ln(1+(N−df+0.5)/(df+0.5)); local lexical implementation | R07, R18 |
| BM25 tokenization and scoring | Split lower-case/digit-to-upper-case camel boundaries, then extract lower-case [a-z][a-z0-9]*; unique query terms; no document truncation; FP64 | R18 |
| UniXcoder | <encoder-only>; query 128, code 256; mean final-layer non-padding token representations including special tokens; L2 normalization and FP32 dot product; batch 16 | R12, R31 |
| Ranking | Top100; descending score, then ascending original document ID; all within-condition controls fix corpus and labels | R06, R11, R12, R14 |

temperature=1, top_k=0 and top_p=1 apply to sampling sensitivity configurations; the main greedy setting does not use these sampling cutoffs. Selection k counts candidates retained from a pool; generation still enumerates all specified positions.

## Table S19. Study weight identities and checksums

| Model | Identity and use | Weight-file SHA-256 |
| --- | --- | --- |
| W1 Local CQC T5 | Independent CQC training from T5-base; source-validation-best weights at update 92,000 | 7acd4cb8379a1f5297bb978a748bc002b708ef5fd60a8b4737a6f8c7d2ebcaa5 |
| W2 Public T5-base | Public base weights; paired-weight comparison and shared initialization for limited training | a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4 |
| W3 CodeBERT retriever | SSQR pooler/dot-product objective; final weights after two epochs | 0d533c04602916c22d1f13d282056f081b6bd04f9866ccc1ded46a0516e13e3d |
| W4 UniXcoder-base | Public zero-shot weights; revision 5604afdc964f6c53782a6813140ade5216b99006 | 6732b60037b086557e4c731c3748c99b4fec735d9ffe3c4c70061fd142e6b9d9 |
| W5 CodeBERT cross-encoder | Historical large_dense_4096 cross-encoder; frozen before full H | 3a0f26dcae197b2e7cb97b62349ccd743e4211a835a8e2ab2842376a7c4c2f39 |

Sealed identities: W1–W3, R14; W4, R12; W5, R07. The original settings audit calculated each file hash and matched the sealed record. File locations are listed in the source index.
Paired-weight analyses use each weight's own complete candidate pool. Queries, corpus, labels and retriever are matched; candidate pools are not assumed identical across weights.

## Table S20. Six generation configurations

| Configuration | Positions | Decoding | Max. new tokens | Length incl. start |
| --- | --- | --- | --- | --- |
| greedy20_source | 1…n | Greedy | 19 | 20 |
| greedy20_all | 0…n | Greedy | 19 | 20 |
| greedy10_source | 1…n | Greedy | 9 | 10 |
| greedy10_all | 0…n | Greedy | 9 | 10 |
| sample10_source | 1…n | Sampling | 9 | 10 |
| sample10_all | 0…n | Sampling | 9 | 10 |

Source: R06. Each domain fixes 512 queries/groups matched across configurations. Sampling uses temperature=1, top_k=0 and top_p=1. SHA-256(seed=101, query_id, position) sets each query-position random stream; one candidate is generated per position. Greedy batch size is 8; sampling batch size is 1.

## Table S21. Eight training configurations with matched update budgets

| Arm | Optimizer | Schedule | Schedule horizon | Length | Weight decay | Seed |
| --- | --- | --- | --- | --- | --- | --- |
| L | AdamW | Constant | — | 512 | 0.01 | 101 |
| O | Adafactor | Constant | — | 512 | 0.01 | 101 |
| S | AdamW | Linear | 100,000 | 512 | 0.01 | 101 |
| N | AdamW | Constant | — | 128 | 0.01 | 101 |
| W | AdamW | Constant | — | 512 | 0.0 | 101 |
| P | Adafactor | Linear | 100,000 | 128 | 0.0 | 101 |
| PN | Adafactor | Linear | 1,093 | 128 | 0.0 | 101 |
| PS | Adafactor | Linear | 100,000 | 128 | 0.0 | 42 |

Sources: R15/R16/R30. All arms start from the same T5-base weights, share 32,768 training rows and data order, and run 1,024 updates. Microbatch 8; accumulation 4; effective batch 32; initial learning rate 0.001; gradient norm cap 1; BF16 computation with FP32 weights.
The linear schedule horizon differs from the actual 1,024-update budget. Adafactor fixes scale_parameter=false and relative_step=false. Each arm uses final weights without early-stopping selection. Evaluation separately retains own-generated pools and rescoring of the fixed historical pool.

## Table S22. Fixed-action adoption and rank-fusion settings

| Item | Setting | Source |
| --- | --- | --- |
| Sample and action | CoSQA D: 1,490 queries in 512 connected groups; the same ENTR Top1 action is fixed | R09 |
| Training target | Fixed-action RR@100 minus original-query RR@100; queries weighted equally; relevance labels do not enter features | R09 |
| Feature blocks | M0=27, MT=37, MR=61, MRT=71 dimensions; complete feature names and partitions remain in the protocol | R09 |
| Fitting | StandardScaler + Ridge; alpha=1; solver=svd; intercept included; no prediction clipping or tuning | R09 |
| OOF split | Reuse 5 folds of whole connected groups; fit standardization and models using training folds only | R09 |
| Eligible actions | Change both query text and the full Top100 ranking; all prediction heads share this observable requirement | R09 |
| Fixed20 | In each held-out fold adopt floor(eligible actions/5) with the highest predictions, including negative scores; break ties by ascending query_id | R09 |
| Zero threshold | Adopt only eligible actions with raw predicted gain >0; both policies use all D queries as denominator | R09 |
| Resampling scope | Conditional on fitted OOF models; do not refit or change quotas within bootstrap resamples | R09 |
| RRF inputs | RRF3 uses three ENTR lists; RRF4 additionally includes the original-query list; deduplicate complete input text before equal-weight fusion | R10 |
| RRF parameters | Constant 60; input and output depth 100; score Σ1/(60+rank) | R10 |
| RRF arithmetic and ties | Integer ranking equivalent to exact rational arithmetic; ascending document-ID ties; return top 100 from the retrieved-document union | R10 |
| Workload | Count deduplicated retrieval inputs; do not interpret counts as online latency; no risk gate in this control | R10 |

Fixed-action Ridge uses alpha=1. Full-H models in Table S23 use alpha=10. These are separate experiments with their original settings.

## Table S23. Frozen full-H models and decision rules

| Item | Setting | Source |
| --- | --- | --- |
| Three D roles | Gain fit: 953 queries/307 groups; verification-update fit: 288/103; risk fit: 249/102; mutually exclusive whole groups | R08 |
| Evaluation sample | Full H: 8,530 queries in 1,963 groups; inference follows freezing of fitting and thresholds | R07 |
| Gain prediction | 71 actual input dimensions; StandardScaler + Ridge alpha=10; weight each candidate by 1/candidate count within query | R07, R19 |
| Base harm prediction | StandardScaler + LogisticRegression; C=0.1, lbfgs, max_iter=2000, random_state=261019 | R19 |
| Gain-stage adoption | Choose the highest predicted gain within the same ENTR min(3,n) set; adopt only if prediction >0 and ranking changes; clip prediction to [−1,1] | R19 |
| Verification quota | Order positive base predicted gains; at most min(floor(N/4), eligible queries); break ties by query_id | R07, R19 |
| Verification evidence | 8 distinct codes per query; construct from original-query success anchors, current winner/competing candidates and rank shifts | R07, R19 |
| Verifier input | Original query paired with code; truncate separately to 64/160 tokens plus special tokens; inference batch 32 | R20 |
| Verifier precision | Frozen eval model; BF16 autocast with FP32 output; scoring uses the original query rather than the rewrite | R20 |
| Update model | 71+2+18=91 dimensions; weighted StandardScaler + Ridge alpha=10; add 0.5×predicted residual to the prediction, then clip to [−1,1] | R19 |
| Risk model | negative/zero/positive three-class model; zero is reference; base-probability offset; 20 inputs; mask duplicate verification-availability column | R07, R19 |
| Risk regularization and optimizer | Intercept penalty 1; verification-indicator coefficient penalty 50; other coefficients 10; L-BFGS-B, maxiter=2000, ftol=1e−11, gtol=1e−7 | R19 |
| Fallback | Threshold 0.075; when exceeded, restore the original full Top100 without choosing another candidate; retain incurred verification cost | R07, R19 |
| Numerical zero and small samples | TOL=1e−10; zero residual below 30 fitting rows; retain probability offset below 30 risk rows or with fewer than two classes | R19 |

Full-H Ridge alpha=10 differs from the fixed-action alpha=1 in Table S22. The risk threshold is an execution rule; observed harm counts do not certify population risk.
HGB secondary control (R07): squared_error; max_iter=200; learning_rate=0.05; max_leaf_nodes=7; max_depth=3; min_samples_leaf=30; l2_regularization=10; max_bins=63; early_stopping=false; random_state=20261019.

## Table S24. Comparison families, resampling and study samples

| Analysis | Queries/groups and corpus | Family | Resamples / seed | Primary setting |
| --- | --- | --- | --- | --- |
| Specified CoSQA replication | Capped H: 7,718 queries / 1,963 groups; at most 16 per group | 1 | 10,000 / 261019 | Two-sided 95% dual intervals; strictly within ±0.005 and at least 20 changed groups |
| Full-H stages | Full H: 8,530 queries / 1,963 groups | Descriptive | 10,000 / 261019 | Stage-mean intervals; retention-ratio seed 261020 |
| Same-pool B/G | AdvTest 2,048/594; CoSQA D 1,490/512; two retrievers | 24 | 10,000 / 20261020 | ENTR/PROB B/G and direct component differences |
| StaQC confirmation | 13,312 queries/groups; corpus of 16,384 codes | 2 | 10,000 / 202610202 | CodeBERT ENTR B/G; PROB/BM25 descriptive |
| UniXcoder extension | Reuse CoSQA D and the preceding StaQC sample | 12 | 10,000 / 202610203 | B/G in both domains and paired differences versus CodeBERT/BM25 |
| Matched weights | 512 queries/groups per AdvTest/CoSQA domain | 24 | 20,000 / 202610204 | Two domains × two selectors × B/G × within-weight effects and direct differences |
| New paired-weight confirmation | 25,600 new StaQC queries/groups; corpus of 32,768 codes | 6 | 20,000 / 202610215 | ENTR B/G under both weights and paired weight differences; no equivalence test |
| Fixed-action adoption | CoSQA D 1,490/512 | 8 | 10,000 / 20261020 | Two conditional increments × two adoption policies × two retrievers |
| RRF fusion | CoSQA D 1,490/512 | 4 | 10,000 / 20261020 | RRF3/RRF4 versus original query; two retrievers |
| Limited training | 512 queries/groups per AdvTest/CoSQA domain | 28 | 10,000 / 20261020 | Retain sealed primary family and both candidate-pool views |

Sources: R07/R09–R16/R21/R26. Queries are the weighting unit. Resampling retains whole connected or repository groups and shares paired resamples within each study. Multiplicity-adjusted comparisons use Bonferroni cluster-t and group-percentile-bootstrap intervals.
B/G direction requires both adjusted intervals to exclude zero on the same side and at least 20 nonzero groups. Study families and populations remain separate. CoSQA ±0.005 is its specified equivalence margin; 0.005 in new paired-weight planning is a precision target.
The limited-training resampling seed follows the actual analysis entry R35; other settings follow R07/R09–R16/R21/R26.

## Table S25. Human review and supplemental relevance judgments

| Item | Setting | Source |
| --- | --- | --- |
| Frame and strata | CodeBERT full H8530; 8 disjoint strata fixed before collecting labels; 16 queries sampled per stratum | R22 |
| Fixed sampling rule | SHA-256 pseudorandom ordering; salt rene-mechanism-review-20261020-v1; exact concatenation rule in source code | R22, R23 |
| Review size | 128 queries; each of two reviewers independently assesses 383 distinct rewrites and 327 code panels | R33 |
| Blinding | Hide method, score, ranking outcome, stratum and target role; separate semantics from code-relevance assessment | R33 |
| Semantic classes | Meaning-preserving, meaning-shifting, unreadable, indeterminate; rationale for every item | R33 |
| Code classes | 2/1/0 or unknown, relative to original query; unknown is not 0; displayed panels do not relabel the full corpus | R33 |
| Agreement | Semantic exact agreement and nominal Cohen κ; nominal code κ including unknown; linear-weighted κ and coverage on jointly judged panels | R33 |
| Weighted events | Original Nh/nh stratification weights; overlapping events; simultaneous stratum-wise hypergeometric bounds and fixed-H8530 95% intervals | R22, R33 |
| Top-3 supplemental review | Union of original-query and three-candidate Top-3 lists: 610 panels; reuse 266 judgments and add 344 per reviewer across 126 of the same queries | R25, R33 |
| Primary supplemental diagnostic | Fixed RIDGE_verified, RR@3, grade-2 relevance, original eight-stratum weights; share unknown relevance assignments for identical text | R25 |
| Sensitivity settings | Also report grades 1/2, unweighted analyses, each stage and RR@1; retain both reviewers' labels and disagreements | R25 |
| Identification and sampling envelopes | Sample bounds handle unjudged relevance; finite-H8530 sampling envelopes separately use hypergeometric inversion and Bonferroni; Neyman/Welch–Satterthwaite t is approximate | R24 |
| Coverage scope | Simultaneous envelopes for four reviewer × threshold primary-policy combinations; no claim of joint coverage for all 48 combinations; annotation error excluded | R24, R25 |

Sources: R22–R25 and Chinese-manuscript snapshot R33. Original independent labels, supplemental judgments and disagreements remain separate. Diagnostic RR@3, sample identification bounds and sampling envelopes do not replace full-H MRR@100.
Both reviewers gave informed consent before participating, after learning the study purpose and review tasks, and agreed to share de-identified annotations and rationales. The paper identifies reviewers as A/B and reports aggregate statistics; the reproduction materials retain anonymous item-level annotations and rationales without reviewer identities. No formal ethics review or exemption determination was sought for this study.
