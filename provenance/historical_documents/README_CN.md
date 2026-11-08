# SSQR 外部复现：GitHub 私密审稿材料

本包包含实验源码、实验设计与过程说明、完整汇总结果、核心逐查询数值复算输入、环境参数，以及两位审阅者的人工评审与补审数据。**不含论文正文、Word、LaTeX、参考文献库或论文 PDF。**

## 先看这些文件

- [安装与运行](INSTALL.md)：先安装依赖，再执行统一复算入口。
- [预期结果与复算范围](EXPECTED_RESULTS.md)：明确哪些结果重新计算、哪些表提供原结果与源代码。
- [实验来源索引](docs/EXPERIMENT_MAP.csv)及[实验过程说明](docs/EXPERIMENT_PROCESS.md)。
- [环境说明](docs/ENVIRONMENT.md)、[完整环境参数 S16–S25](docs/settings/experiment_environment_settings.md)。
- [人工评审说明](human_review/README_CN.md)：初评 A/B、补审 A/B、逐条理由及统计入口。
- [数据与权重获取](docs/DATA_AVAILABILITY.md)、[三份自训模型附件及安装](docs/MODEL_ASSETS_CN.md)、[CODEnn 输入复建](docs/CODENN_RECONSTRUCTION_CN.md)。
- [复现对象与结果对应](docs/REPLICATION_SCOPE_CN.md)、[冻结风险工作点](docs/RISK_WORKING_POINT_CN.md)、[RENE 要求对应](docs/RENE_REQUIREMENTS.md)。
- [GitHub 私密仓库上传说明](docs/PRIVATE_GITHUB_UPLOAD_CN.md)。

## 直接复算

使用 64 位 Python 3.12。安装依赖后，复算无需 GPU、模型下载或账号。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools/reproduce_all.py
```

核心数值入口重新计算 14 个分析单元、103 项比较及完整 bootstrap；人工评审入口重新计算一致性、10 类加权事件、24 类加权错误类型、6,144 条逐查询相关性识别界及 48 组抽样包络。两项 `ACCEPTANCE.json` 均须为 `passed: true`，数值项另须 `full_bootstrap: true`。

## 内容口径

`results/tables/` 保留 S1–S15 对应的历史 T1–T15 文件名。新 StaQC 两权重确认单列。所有表均供查看，但不是每张辅助表都已接入便携式完整重算，具体见预期结果说明。

人工评审保留 A/B 的标签、理由、unknown 和分歧。跨文件匿名编号已统一，不包含身份或过程时间。原始查询与代码语料依上游许可取得；自训 W1/W3/W5 以独立模型附件提供，公开基础权重按对应上游版本取得。指纹、来源、配置及逐文件哈希可供核对。源码包含历史训练、生成、检索与分析流程；便携复算从缓存数值开始。

此包按 RENE 对材料可用性、来源、方法与复现范围的要求组织。官网未提供独立的 GitHub 固定目录模板；IEEE 模板针对论文。本版本 `2026.10.31` 的仓库入口为 https://github.com/chak6229/RAG 。实际发布与远端核验状态见 [RELEASE_STATUS.json](provenance/RELEASE_STATUS.json)。
