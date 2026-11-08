# SSQR 三项实现对齐

更新：2026-10-19T07:50:08.028369+00:00。状态：完整训练与最终验收通过。

| 项目 | 实现和验收 | 状态 |
|---|---|---|
| CoSQA 输入接口 | 真实查询只取 doc；code_tokens 字符串保持词项；官方 ID；public 查询/代码与 private qrels 分开；20,604配对、10,020正查询、6,267代码一致 | 完成 |
| T5 负熵位置评分 | 原始 teacher-forcing logits 的32,128全词表熵；包括哨兵/EOS、排除 padding；ENTR/PROB/RAND共享缓存；实际权重数值核对 | 完成 |
| CodeBERT 训练与推理 | 从base重新训练pooler，无归一化点积，query→code完整批内CE；32配对、256长度、两轮15,740步、linear10%warmup、weight_decay=0 | 完成 |

最终权重：`models/codebert-ssqr-pooler-2epoch-20261019/checkpoint-15740`；SHA-256 `0d533c04602916c22d1f13d282056f081b6bd04f9866ccc1ded46a0516e13e3d`。

19项契约测试通过。真实T5批量/单条最大差2.8610e-6，独立FP64熵最大差3.4758e-6。主熵包含全部非padding标签位置；span-only熵单列，不能混用。

## 复现设置

- 默认沿公开JSON源码1…n位置、显式现代贪心设置；另支持论文0…n位置。公开作者未记录完整解码选项，不能声称恢复了原解码状态。
- PROB定义为teacher-forced平均标签log概率；作者未公开PROB源码，此操作定义需在新实验前冻结。随机位置不放回，从同一缓存选择，候选不足不补原句。
- T5使用本地best92000权重；源家族T5Tokenizer legacy=True和base SentencePiece，词表逐项核对本地fast tokenizer。
- CodeBERT主权重为预定最终两轮checkpoint；另存只按CodeXGLUE valid选出的best。现代PyTorch、BF16和激活检查点是运行环境差异；完整批内负例保留。最后不足32的批次沿源实现保留。
- 新权重单独保存在models/codebert-ssqr-pooler-2epoch-20261019；旧权重和五处CURRENT/STABLE保护项均核对不变。

## 旧审查的一处纠正

此前冻结审查将本地完整CodeBERT的学习率调度写为constant有误；原完整本地脚本已经使用linear+10%warmup。需要纠正的是CLS/归一化训练目标。旧冻结证据保留，本报告追加纠正。

## 验收证据与下一步

- IMPLEMENTATION_VERIFICATION.json、TEST_RESULTS.json、T5_REAL_WEIGHT_SMOKE.json、TRAINING_BENCHMARK.json。
- FINAL_ACCEPTANCE.json：实际最终权重对作者向量/损失、独立重载、合成输入全链路排名全部通过。
- CoSQA方法效果尚未计算。下一步按既定计划冻结开发/确认分组、设置与统计比较，再开展开发试跑与功效估计；本任务不生成新的4/8seed候选。risk_certified=false、NoRewrite保持。
