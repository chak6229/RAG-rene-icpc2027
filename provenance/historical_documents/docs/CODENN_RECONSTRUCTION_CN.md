# CODEnn 输入与预处理复建

本地 CQC 训练使用 DeepCS 的 Keras 数据来源。上游 [Keras README](https://github.com/guxd/deep-code-search/blob/master/keras/README.md) 给出 [Google Drive 数据目录](https://drive.google.com/drive/folders/1GZYLT_lzhlVczXjD6dgwVUvDDPHMB6L7?usp=sharing) 和 [百度网盘替代入口](https://pan.baidu.com/s/1U_MtFXqq0C-Qh8WUFAWGvg)。下载真实数据，不能使用代码仓库中的小型 dummy 数据。

需要的文件为 `train.desc.h5` 和 `vocab.desc.pkl`。保留数据提供方的访问与使用条件。原始和解码后的注释不随小型源码库再分发；准确文件身份见 [CODENN_INPUTS.json](assets/CODENN_INPUTS.json)。当前上游网页提供的是获取路线，下载后必须再按校验值确认是否为历史输入。

## 路径与命令

将两个原始文件放在 `external_inputs/codenn/`，在源码库根目录执行：

```powershell
python tools/verify_codenn_inputs.py --raw-dir external_inputs/codenn
python prepare_codenn_cqc.py --h5 external_inputs/codenn/train.desc.h5 --vocab external_inputs/codenn/vocab.desc.pkl --output-dir data/codenn_cqc --train-samples 1000000 --validation-samples 10000 --seed 101
python tools/verify_codenn_inputs.py --raw-dir external_inputs/codenn --prepared-dir data/codenn_cqc
```

预处理使用 PyTables，属于模型数据准备环境，不包含在仅有 NumPy/SciPy 的便携统计重放依赖中。

| 交付/获取位置 | 对应历史位置 | 用途 |
| --- | --- | --- |
| external_inputs/codenn/train.desc.h5 | imports/codenn_keras/keras/train.desc.h5 | 依源顺序解码自然语言注释 |
| external_inputs/codenn/vocab.desc.pkl | imports/codenn_keras/keras/vocab.desc.pkl | token-to-id 词表 |
| data/codenn_cqc/train.txt | 同名历史相对路径 | 前 1,000,000 条可解码注释 |
| data/codenn_cqc/valid.txt | 同名历史相对路径 | 紧接训练区间的 10,000 条注释 |
| data/codenn_cqc/valid_fixed_cqc.jsonl | 同名历史相对路径 | 固定连续片段遮盖验证集 |

实际源索引为训练 `0—999999`、验证 `1000000—1009999`。读取前 1,010,000 行未丢弃记录。完整规则和原预处理清单见 [CODENN_PREPROCESSING_MANIFEST.json](assets/CODENN_PREPROCESSING_MANIFEST.json)。校验既检查两个原始输入，也检查三个预处理结果，不使用“文件名相同”代替输入身份相同。

这些文件复建的是本文独立训练使用的输入，不是未恢复的 SSQR 原作者训练 CSV。W1 的训练参数、完成 93,750 次更新及第 92,000 次更新权重的选取记录仍按 S17/S19 保留。
