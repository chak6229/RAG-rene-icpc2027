# Multi-candidacy integration: implementable baseline experiment

CoSQA D1490 Query/ 512 Associated group, CodeBERT/BM25, the same ENTR candidate for freezing local T5. Compare NoRewrite, ENTRTop1, RRF3 (three candidates) with RRF4 (original query + three candidates), ENTRBest3 only as a labeled opportunity indicator.

RRF Constant 60, enter Top100 to reweight the text by exactly the same input, equal, precise fractional sorting, document ID determine parallel rows, output and set Top100. No action qrels, reassembly after construction. No new generation, training, retrieval, CE or risk doors, no sweep parameters.

## CodeBERT

|Programme|MRR|ΔMRR|Group bootstrap95%|Four main effect correction compartments|Useful/harmful queries|Losses per query|
|---|---:|---:|---|---|---:|---:|
|NoRewrite|0.51890040|+0.00000000|[+0.00000000, +0.00000000]|—|0/0|0.00000000|
|ENTRTop1|0.50524189|-0.01365851|[-0.02269701, -0.00478947]|—|292/388|0.04377618|
|RRF3|0.50880074|-0.01009966|[-0.01971797, -0.00039499]|[-0.02279004, +0.00227612]|303/352|0.04089274|
|RRF4|0.51259246|-0.00630794|[-0.01356337, +0.00068063]|[-0.01556317, +0.00252055]|287/284|0.02826442|
|ENTRBest3|0.57927884|+0.06037845|[+0.05146242, +0.06938687]|—|528/125|0.00774419|

|Secondary pairing|Δ MRR|Group bootstrap95%|
|---|---:|---|
|RRF4-RRF3|+0.00379172|[-0.00116375, +0.00886707]|
|RRF3-ENTRTop1|+0.00355885|[-0.00776614, +0.01479171]|
|RRF4-ENTRTop1|+0.00735057|[-0.00181023, +0.01681903]|

|Only search input required for online policy|Total|Average value per query|Location generation (shared pool)|
|---|---:|---:|---:|
|NoRewrite|1490|1.0000|0|
|ENTRTop1|1490|1.0000|9915|
|RRF3|4440|2.9799|9915|
|RRF4|5930|3.9799|9915|

The total search input common to the four lists in the controlled comparison is 5930; Original query+Top1 is 2980. This is not proof of the same cost of strategy online. The original experiment was actually carried out in bulk/cachel and the table is the only input required per query; Do not delay, token or amount.

## BM25

|Programme|MRR|ΔMRR|Group bootstrap95%|Four main effect correction compartments|Useful/harmful queries|Losses per query|
|---|---:|---:|---|---|---:|---:|
|NoRewrite|0.22412467|+0.00000000|[+0.00000000, +0.00000000]|—|0/0|0.00000000|
|ENTRTop1|0.20377371|-0.02035096|[-0.02571306, -0.01522141]|—|45/555|0.02479017|
|RRF3|0.22055426|-0.00357041|[-0.00605671, -0.00109454]|[-0.00681099, -0.00040266]|92/411|0.00715884|
|RRF4|0.22210031|-0.00202436|[-0.00375980, -0.00041231]|[-0.00419819, +0.00002045]|81/349|0.00417705|
|ENTRBest3|0.23390096|+0.00977629|[+0.00518148, +0.01468792]|—|128/319|0.00499119|

|Secondary pairing|Δ MRR|Group bootstrap95%|
|---|---:|---|
|RRF4-RRF3|+0.00154605|[-0.00023031, +0.00338637]|
|RRF3-ENTRTop1|+0.01678055|[+0.01193765, +0.02195994]|
|RRF4-ENTRTop1|+0.01832660|[+0.01338007, +0.02354481]|

|Only search input required for online policy|Total|Average value per query|Location generation (shared pool)|
|---|---:|---:|---:|
|NoRewrite|1490|1.0000|0|
|ENTRTop1|1490|1.0000|9915|
|RRF3|4440|2.9799|9915|
|RRF4|5930|3.9799|9915|

The total search input common to the four lists in the controlled comparison is 5930; Original query+Top1 is 2980. This is not proof of the same cost of strategy online. The original experiment was actually carried out in bulk/cachel and the table is the only input required per query; Do not delay, token or amount.

## Interpretation of the boundary

Integration produces new rankings, which may exceed best3 and best3 cannot be the denominator of integration upward or yield retention rates. RRF4 The inclusion of original queries does not mean that the guarantees are intact; Harmful inquiries and loss ranges are reported. The three candidates are still screened for primary entropy, so the experiment does not require any extra UQ integration of the predictor head, rather than eliminating all the cross-references to entropy information.

The data have been consumed and the agreement has been frozen only before this new integration effect is calculated. (b) Inter-group descriptions of the differences in the development and do not constitute new independent confirmations; Four main effects provide Bonferroni sensitivity. Cross-retrieval is directly modulated at RESULTS.json, and no significant inferences are made by the obvious side. Standard RRF is not a new algorithm; See FINDINGINGS.md and PRIOR_WORK_COMPARISON.md.

Independent acceptance and inspection AUDIT.json, FINAL_ACCEPTANCE json, covering full sorting of Fraction, weighting, action/cost, by queryRR, indicator ranges and freezing of Hashi. Old A/B, old H confirmation and official five pointers.
