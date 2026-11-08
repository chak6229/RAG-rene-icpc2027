# 自训模型附件与安装

本版本为 `2026.10.31`，仓库为 https://github.com/chak6229/RAG 。模型附件与小型源码库分别交付，以避免将大权重放入普通 Git 文件。投稿使用的附件名称、逐文件 SHA-256 和历史运行路径均记录在 [STUDY_ASSETS.json](assets/STUDY_ASSETS.json)。

## 获取与核对

三个附件可作为同名审稿附件提供，或从已获访问授权的私库 [rene-review-20261031 release](https://github.com/chak6229/RAG/releases/tag/rene-review-20261031) 下载。仅有私库链接不会自动授予审稿人权限，访问应按投稿安排单独配置。发布状态与实际在线核验结果见 [RELEASE_STATUS.json](../provenance/RELEASE_STATUS.json)。

| 附件 | 内容 | 历史模型路径 |
| --- | --- | --- |
| SSQR_RENE_W1_CQC_T5_20261031.zip | 第 92,000 次更新的 CQC T5 权重、配置、分词器及所需 SentencePiece 文件 | models/t5-cqc-codenn-1m/best/ |
| SSQR_RENE_W3_CodeBERT_20261031.zip | 两轮训练的 CodeBERT 最终权重、配置、分词器和表示契约 | models/codebert-ssqr-pooler-2epoch-20261019/ |
| SSQR_RENE_W5_CrossEncoder_20261031.zip | 冻结的 large_dense_4096 核验器及其基础模型配置、分词器 | results/uq_pilot_20261014/action_witness_uq/data/ce_large_dense_4096.pt 与 models/codebert-base/ |

附件保留实际使用的权重，不含优化器状态或未使用的中间 checkpoint。W1 额外补齐了历史流程从基础 T5 读取的 SentencePiece 辅助文件，没有修改训练权重。各文件与表 S19 的 W1、W3、W5 校验值对应。

下载三个附件到同一目录后，在源码库根目录运行：

```powershell
python tools/install_study_assets.py --asset-dir ../release-assets --check-only
python tools/install_study_assets.py --asset-dir ../release-assets
```

第一条逐一检查 ZIP 及其中全部文件的 SHA-256，不加载模型。第二条将文件安装到上表的相对路径；遇到已存在但散列不同的文件会停止。安装脚本检查路径，拒绝越出目标根目录的条目。

## 使用范围

自训资产安装后，历史生成、检索和核验源码能定位相应模型文件。完整输入还包括各研究冻结的查询、语料、标注和来源映射，应依 [数据获取说明](DATA_AVAILABILITY.md) 取得。模型环境与便携统计重放环境不同，分别见环境说明和补充表 S16—S25。此次交付验证资产身份、安装和缓存统计重放，未将这些检查表述为全量重训或全部原始排名的端到端复现。

W1 的基础模型来自 [T5-base](https://huggingface.co/google-t5/t5-base)，W3 和 W5 来自 [CodeBERT-base](https://huggingface.co/microsoft/codebert-base)。保留上游模型卡及许可条件；本次私密交付不赋予额外的公开再分发许可。
