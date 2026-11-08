# CoSQA 数据与 SSQR 复现审查

日期：2026-10-19。任务：核对查询来源、相关标注、许可、数据重叠、作者源码和 checkpoint，并判断其能否支持 RENE 主线。

**结论：CoSQA 可以作为新查询来源上的 SSQR 外部复现数据。采用作者的 6,267 文档检索语料和单目标相关性定义；主张限定为真实 Web 查询的迁移，不能写成完全未见代码上的验证。当前本地流程需要建立独立的 SSQR 对齐分支，尚不能直接作为作者原实现的重跑。**

本次已完成数据和来源审查、全量重叠扫描、实际 tokenizer 检查、权重身份与训练步数登记、实现差异表和一致性验收。没有运行 CoSQA 上的生成、检索排名、模型训练或方法效果评测；相关标注已用于来源审查，不能称为从未访问标签。正式 CURRENT/STABLE 的 5 个保护文件保持原哈希。

## 1. 数据版本和真实来源

| 来源 | 固定版本 / 证据 |
|---|---|
| CoSQA / CoCLR 作者仓库 | `Jun-jie-Huang/CoCLR@14ebcacf9e9bc3e7109102632bc63047876f27d2` |
| Microsoft 分发版本 | `microsoft/CodeXGLUE@ac74a62802a0dd159b3258c78a2df8ad36cdf2b9` |
| CodeSearchNet 来源 | `github/CodeSearchNet@106e827405c968597da938f6b373d30183918869` |
| SSQR 作者仓库 | `RedSmallPanda/SSQR@7fe5d780740eb4348500658ce6decf188a542d2b` |
| C-UDA 官方文本 | `microsoft/Computational-Use-of-Data-Agreement@a28ca06f6f8ecac0b5856ca6179ac49e55f00104` |

本地三份 CoSQA QA 文件同时与 CoCLR 作者版本和 CodeXGLUE 分发版本的 Git blob 完全一致。共核对 15 份本地 CodeXGLUE 文件、21 份选定作者源码/数据文件及 3 份官方 C-UDA 文本；原始文件、Git blob SHA-1 和 SHA-256 已留存。CoCLR dev500 首次网络读取失败，后续以原始 blob 接口成功取得并验证，最终状态见 `VERIFICATION.json`；原失败记录保留。

CoSQA 原论文 §§3–4、§6.1 给出的构造为：

1. 查询来自 Microsoft Bing 搜索日志，保留包含 “python” 的查询，再按代码搜索意图过滤。
2. 代码来自 CodeSearchNet 的 Python 函数，标注时可见函数及其文档。
3. 候选收集使用在 148K StaQC 对上微调的 CodeBERT，为查询选择相似度至少 0.5 的 Top-1 代码；同一代码出现次数受限。
4. 每对至少 3 名标注者，共 100 多名参与者；移除低一致性和非代码搜索意图样本，最终平均 Krippendorff α 为 0.63。

因此它回答的是**经过代码搜索意图与 CodeBERT 候选收集筛选的真实 Web 查询**问题。它不是任意开发者查询的随机样本，也不是点击成功日志。跨检索器结果需要结合这种收集方式解释。

## 2. 相关性和检索构造

### 2.1 两种任务必须分开

| 文件 / 切分 | 查询—代码对 | 正例 | 负例 | 不同代码 |
|---|---:|---:|---:|---:|
| QA all | 20,604 | 10,020 | 10,584 | 6,267 |
| QA train | 20,000 | 9,707 | 10,293 | 6,184 |
| QA dev | 604 | 313 | 291 | 552 |
| 官方 retrieval dev | 500 | 500 | 0 | 478 个目标 |
| 官方 retrieval test | 500 | 500 | 0 | 472 个目标 |

20,604 条原始查询在当前规范化规则下全部唯一，无重复配对、冲突标注或多正例查询。QA all 恰好等于 QA train + QA dev。**QA 的 20,000/604 切分不能当作检索的开发/测试切分。**

标签 0 只表示该配对代码不能回答该查询；它不表示查询在整个语料中无答案。没有已标注正例的 10,584 条查询不能直接拿来计算有意义的目标 RR。其他未标注查询—代码组合是未判断项，不能称作人工确认的不相关文档。

作者另外提供了检索 dev500/test500 及 `code_idx_map.txt`。已逐条验证：

- 语料恰好 6,267 个不同函数，ID 连续为 0–6266，与本地 CoSQA 函数集合完全相同。
- 两个检索切分全部 label=1；每行 `retrieval_idx` 等于其目标函数的语料 ID。
- 原始字段逐项等于 QA all 对应记录；dev/test 查询交集为 0。
- 官方 test 的不同目标函数实际为 **472**；源码注释里的 473 不作为统计依据。

检索 train 的 19,604 行身份由 all 减去官方两个 500 行切分恢复，包含 9,020 正例、10,584 负例。**没有下载并校验原检索 train JSON 的字节**，因此此项只声明行身份和标签推导一致。

建议保留官方 6,267 文档作为共同语料。CoSQA 原评测是全排名目标 RR；SSQR 的公开结果导出最多给 100 个答案。新实验应同时固定 cutoff：用完整排名生成明确的 MRR@100 和全排名 MRR，主指标在效果计算前写入协议，不能混用。原 SSQR 表 4 的 best-of-3 不含原查询回退；自动采纳、融合和包含原句的 oracle 分列。

### 2.2 数据接入中的关键字段

| 字段 | 本次核实含义 | 接入要求 |
|---|---|---|
| `doc` | Bing Web 查询 | T5 和检索器使用的原查询 |
| `code` | 完整函数，通常含文档 | 保持作者语料 ID 对应；不默认为纯代码索引文本 |
| `code_tokens` | 已提供的代码 token 字符串 | 对齐 SSQR 的代码表示时直接使用该字符串 |
| `docstring_tokens` | 配对函数的文档 token 字符串 | 不是 Web 查询，不得作为查询输入 |
| `label` / `retrieval_idx` | 配对相关性 / 检索目标 ID | 只进入标签/评测通道 |

CoCLR `code_search/run_siamese_test.py:68–88` 同样使用 `js['doc']` 作为查询。20,604 行中没有一行 `doc` 与 `docstring_tokens` 在空白归一后相同；后者 20,487 行非空。现有 `generate_codexglue_rewrites.py` 优先取 `docstring`、再把 `docstring_tokens` 当列表拼接，**直接套用会把目标函数文档逐字符拼成输入，不能用于 CoSQA**。

相同函数的 `code_tokens` 在各记录间完全一致，6,267 个索引字符串也唯一。建议主比较使用代码 token 索引；如果另跑 CoCLR 的含文档原始索引，需列为数据表示条件，不能与 SSQR 代码索引混为同一设置。相关标签仍对应同一个函数 ID。

### 2.3 实际 tokenizer 审查

已用本地实际 T5TokenizerFast 和作者源码所用 T5Tokenizer 类进行检查；后者用本地 t5-base 的 SentencePiece 词表重建，不冒充恢复了作者 tokenizer。当前 best 目录仅含 fast tokenizer 文件，直接加载 slow tokenizer 失败；已记录原因，未改环境或模型目录。

- 全部 20,604 条查询，原始空格词数 4–21；T5 纯查询最长 35 tokens。
- 论文 n+1 位置共 **156,603** 个带哨兵输入；公开 JSON 源码的 1…n 位置为 **135,999** 个。
- 两种 tokenizer 的插入输入最长均为 **36 tokens**；在 128 或 512 的长度设置下，没有哨兵缺失或输入截断。
- CodeBERT 查询最长 23 tokens。代码 token 索引在 256 限长下有 **27/6,267** 段被截断；若直接索引含文档的 raw code，则为 **287/6,267**。截断规则须随索引配置固定。

这些是输入与分词检查，尚未检验生成输出质量。

## 3. 许可核对与 artifact 处理

CoCLR 作者 README 明确声明：**代码 MIT，数据 Computational Use of Data Agreement（CUDA）**。CodeXGLUE 根目录的 MIT 不能把 CoSQA 数据重新标成 MIT。

已从 Microsoft 的官方 C-UDA 仓库取得并校验 v1.0 正文。其核心条款为：

- §1.1、§2.1：允许按条款使用、修改和分发数据，用途限定为 computational use。
- §3.1：重分发数据时保留收到的署名/归属信息，并将下游接收者约束于 C-UDA。
- §2.2、§5.5：不含超过微量原数据的分析结果、符合定义的模型等属于 Results，协议不对其另加分发限制。

CoCLR 只写 CUDA 名称，没有附明确版本号或全文哈希；本次保存的是与该名称对应的官方 v1.0 文本，不声称作者逐字附带了该版本。当前科研计算用途与其声明相容。

CodeSearchNet 的 README 另说明代码作为数据的原许可证在各语言 `_licenses.pkl` 中；本地导入目录未找到相应 sidecar。SSQR 仓库没有统一根 LICENSE，部分继承文件带 MIT 头。这些信息在发布时分别保留。

本项目 artifact 的明确处理为：**发布我们编写的适配/评测代码、数据获取说明、来源链接、划分 ID 与哈希；第三方数据和原作者源码由获取脚本按来源取得。** 若后续打包原始数据，必须随包保留适用数据条款和归属，不能统一贴 MIT。无需为当前只读审查或科研计算暂停推进。

## 4. 与训练数据和已消费数据的重叠

扫描已完成，包含原始代码、保留大小写的空白归一代码、已有 code tokens 三层；查询采用 NFKC、大小写和空白归一，并补充 Unicode 词项指纹。以下“重叠”是这些指纹的并集，**没有声称排除了语义克隆或所有基础模型预训练暴露**。

| 对照来源 | 对照行数 | 命中的 CoSQA 不同代码 | 以这些代码为正目标的查询 | 规范化/词项查询重复 |
|---|---:|---:|---:|---:|
| 本地 CodeXGLUE 检索器微调 train | 251,820 | **4,011** | **6,585** | 0 / 0 |
| CodeSearchNet Python 原 train | 412,178 | 5,404 | 8,743 | 0 / 0 |
| CodeSearchNet Python 原 valid | 23,107 | 335 | 532 | 0 / 0 |
| CodeSearchNet Python 原 test | 22,176 | 305 | 493 | 0 / 0 |
| 本地 AdvTest valid | 9,604 | 0 | 0 | 0 / 0 |
| 已消费 AdvTest test | 19,210 | 0 | 0 | 0 / 0 |
| 原 AdvTest test_code | 45,244 | 0 | 0 | — |
| 本地 T5 CODEnn train | 1,000,000 | 不适用 | 不适用 | 0 / 0 |
| 本地 T5 CODEnn valid | 10,000 | 不适用 | 不适用 | 0 / 0 |

AdvTest 包含标识符匿名化，指纹 0 命中不能进一步推成语义上完全不同。各 CSN 切分的代码命中数量也不能简单相加当作不同函数总量。

历史匿名化语料已通过来源索引、URL 或 repository+query 还原到原函数后对照：

| 已消费对照 | 命中代码 | 命中目标的正查询 | 查询重复 |
|---|---:|---:|---:|
| 历史 action 4096 | 66 | 95 | 未作该池查询表对照 |
| 历史 semantic 4096 | 59 | 90 | 未作该池查询表对照 |
| 严格确认 4096 | **88** | **128** | 不适用：文档语料 |
| 严格确认 1536 查询的目标 | 28 | 34 | 0 |
| 已消费 1024 查询的目标 | 25 | 37 | 0 |

这支持“查询来源变了”的外部复现设计，同时要求保留代码暴露分层。10,020 条有已知正例的查询按本地检索器微调暴露可分为：

| 层 | 正查询数 |
|---|---:|
| 目标代码匹配本地微调 train | 6,585 |
| 目标代码无匹配，但来源仓库在微调 train 出现 | 2,087 |
| 目标及已恢复仓库均未在本地微调 train 出现 | **1,096** |
| 目标无匹配、仓库未恢复 | 252 |

这些是**本地微调暴露层**，不能改名为基础模型完全未见层。CoSQA 官方 test500 中，317 条查询的目标代码已匹配本地微调 train。

## 5. 分组与研究问题适配

6,267 段代码中 **6,044** 段恢复了来源仓库，共 2,933 个仓库；**223** 段未恢复。按代码身份与已恢复仓库构成连通组，得到全部查询 3,156 组、有正查询 2,475 组，最大正查询组含 137 条。未知仓库仅按代码分组，不能据此保证其仓库独立。

| 官方检索切分对 | 查询交集 | 代码交集 | 已恢复连通组交集 |
|---|---:|---:|---:|
| dev500 / test500 | 0 | **70** | **108** |
| train 推导身份 / dev500 | 0 | 408 | 390 |
| train 推导身份 / test500 | 0 | 396 | 371 |

因此官方切分适合“新查询”评测，不能直接当作新仓库隔离。若固定官方 test500，其涉及 396 个正查询连通组；排除这些组后，仍有 **6,490 条正查询 / 2,079 组** 可供开发设计，其中 6,282 条 / 1,949 组有已恢复仓库。相比只排除测试 500 行，分组隔离还会移除另外 3,030 条相关查询。

此项只是可用规模审查，尚未分配 D/H、选择门限或证明统计功效。确认集 500 行不能预先承诺足以检出小增量、做等效检验或认证稀有有害事件。

| 研究问题 | 本数据的支持方式 | 必须固定的解释 |
|---|---|---|
| RQ1：SSQR ENTR 相对 RAND/PROB 能否迁移？ | 同一新查询、同一 6,267 文档、同位置级候选缓存比较 | 检验单个已标注目标的排名收益；原 best-of-3 与自动选择分开 |
| RQ2：自动采纳能实现多少收益？ | 对原句、候选、核验与风险采纳逐项计算目标 ΔRR | 损害定义为已标注目标排名下降，不等同人工确认生成内容有害；下游拟合与确认角色分组隔离 |
| RQ3：既有 UQ 证据如何定位？ | 新分布可检验同一预定信号；既有阳性/条件等效保留原范围 | 数据本身不提供“认知/偶然不确定性真值”，不能用熵相关性代替解耦正确性证明 |
| RQ4：是否依赖检索机制？ | CodeBERT、BM25 可共享查询、函数 ID、代码索引文本和 qrels | 结果解释须包含 CodeBERT 候选收集偏向；换 checkpoint 不等于第二检索机制 |

**数据验收决定：采用 CoSQA 作为首选新查询来源；主问题限定为查询迁移和单目标检索收益。** 不需要因为代码来自相同上游而弃用它，但要保留暴露分层和分组设计。

## 6. SSQR 源码、论文与本地实现差异

作者 checkpoint 不是单纯“本轮没找到”。[作者在 issue #1 的回复](https://github.com/RedSmallPanda/SSQR/issues/1#issuecomment-2007139087)（2024-03-19）明确表示毕业后无法再提供 checkpoint，可以根据数据重新训练。当前完整仓库树没有权重，release 列表为空。README 的 Google Drive 链接是预训练数据，不是权重；本地访问超时，未把它判成永久失效。

| 项目 | 论文 / 作者公开源码 | 当前本地实现 | 正式实验处理 |
|---|---|---|---|
| 权重身份 | 原权重已无法由作者提供 | 本地独立训练 T5 / CodeBERT | 标为独立复现，记录实际权重哈希 |
| T5 训练数据 | 论文 CODEnn 前 1M；源码读 `train_context.csv` 并清理/分段 | 本地 HDF5 解码前 1M，后续 10K 作固定验证 | 保留来源、顺序和处理差异；没有作者 CSV 字节，不能声称完全相同 |
| CQC 遮盖 | 论文一段连续 15%；源码 `ceil(.15*n/3)` 个长度 3 的 span、多 sentinel | 单一连续 `round(.15*n)` span | 本地较接近论文描述；对齐契约需注明采用论文还是源码，不能把两者写成一致 |
| T5 长度 | 论文称默认长度；公开 README/源码输入输出均 128 | 512 | 登记训练差异；本次 CoSQA 查询输入不受此截断影响 |
| T5 优化器 | 公开源码 Adafactor | AdamW | 两者不相同；当前权重只能称本地训练实现 |
| T5 学习率 | 源码 linear，强制 `t_total=100000`，warmup=0 | constant，93,750 steps | 写清来源差异，不能只比较初始 lr |
| T5 随机种子 | 论文 101；源码虽有 `seed=101` 配置，末尾实际调用 `set_seed(42)` | 101 | 训练 RNG 与推理 RNG 分列；源码冲突显式登记，不能仅抄配置字段 |
| T5 checkpoint 选择 | 原权重/实际选择不可恢复 | 训练完成 3 epoch；使用 best step92,000，final step93,750 另存 | 区分 best 和 final；本地早停规则未触发，不能写成训练提前停止 |
| 插入位置 | 论文 n+1（0…n）；公开 JSON 路径 1…n | 本地 0…n | 正式主设置与来源敏感性在看效果前固定，可共用位置缓存 |
| 解码与跨度 m | 论文描述采样、m=10；源码仅 `model.generate(input_ids)` | 本地采样、top-k50、最多 12 新 tokens 等显式配置 | 作者训练后 config 不可恢复；记录所选独立实现，不能推定作者调用等于本地采样 |
| ENTR 分布 | 论文全词表分布；源码先生成，再 teacher forcing、全词表 softmax 算熵 | 本地历史生成器用 `generation.scores` 的处理后分布 | **正式负熵比较必须恢复全词表 teacher-forcing 评分**；warper 后熵另命名 |
| ENTR 平均位置 | 源码重包 `<extra_id_0> span <extra_id_1>`，包含特殊标签位置 | 本地排除部分特殊 token，流程不同 | 冻结平均 mask；词面 span 与包含 sentinel/EOS 的范围分清 |
| 熵方向 | 论文和主 JSON 路径取最小熵；另一未使用 text helper 取最大熵 | 本地按负熵降序 | 以主路径为依据；不将未调用 helper 误判为作者主算法反号 |
| PROB / RAND | 表4说明存在，公开 generator 未包含对应实现 | 历史流程不能补足作者精确版本 | 独立实现二者并冻结 PROB 聚合定义、RAND RNG；歧义作预定敏感性，不按结果选 |
| 查询输入 | SSQR 原数据 `docstring_tokens` 是查询 token 列表 | 历史优先 raw docstring；CoSQA 同名字段是目标文档字符串 | 新 CoSQA adapter 显式取 `doc`，禁用旧字段回退 |
| CodeBERT 向量 / 损失 | 公开 `model.py` 用 pooler output、未归一化点积、batch CE | 本地训练和推理用归一化 CLS / 点积 | **仅推理时换 pooler 不构成忠实复现；主基线要有匹配的训练目标** |
| CodeBERT 训练完成度 | 论文 2 epoch；README batch32、block256、seed123456 | 当前指向 step3,000；另有完成 15,740 steps 的模型 | 已有两个 checkpoint 均登记；完整训练模型仍是 CLS 目标，不能自动升级成作者实现 |
| CodeBERT 学习率 | 论文 constant 5e-5；公开代码线性 warmup/decay（10% warmup） | 本地 constant 5e-5 | 论文和源码自身不一致；主设置在协议选择，另一路作必要敏感性 |
| 排名、截断与 tie | 作者代码先排序再按 `topK` 间隔导出，评测给出的答案序列；原生产 tie 不可恢复 | 历史模块有 CLS/pooler、MRR@100 与排名路径差别 | 新评测统一函数 ID、稳定 tie 与 cutoff，保存完整排名/目标 rank，不能照抄导出细节后假定同一指标 |
| 运行依赖与修复 | 老 transformers/Lightning；缺少未使用模块 import；`args.ouput` 拼写错误 | 现代 transformers/PyTorch | 只在新分支做必要兼容修复，保留原文件与补丁；不用修复后源码冒充原始发布文件 |

另一个公开训练实现问题是短文本 span 抽样可能无可行位置；此次没有执行作者训练脚本，后续复现应先做可行性边界验证。作者也已在 [issue #5](https://github.com/RedSmallPanda/SSQR/issues/5#issuecomment-2488160495) 确认缺失的未使用导入可删除。

这张表界定的是外部复现的实现条件，不否定历史同流程对照的相对效应；历史结果继续保留其原数据和处理范围。

## 7. 本地 checkpoint 实际身份

| 模型 | 实际状态 | 权重 SHA-256 |
|---|---|---|
| T5 best | step92,000；训练运行最终完成 93,750 steps / 3 epochs | `7acd4cb8379a1f5297bb978a748bc002b708ef5fd60a8b4737a6f8c7d2ebcaa5` |
| T5 final | checkpoint-93750 | `b543ed2878dfebe8c36acb87564d73ea50f8e05b98d6aa77d3b68785364c3335` |
| 当前 CodeBERT | checkpoint-3000；计划 15,740 steps，实际只到 3,000 | `c3bb2df27ddc777d5f0d6fb22525ca74cfc1c19b459630f60df2fddc97e42bd1` |
| 完整本地 CodeBERT | checkpoint-15740；训练两轮完成；仍是归一化 CLS 目标 | `802943bd77ea247fbe92e1ae2d26f520ef2a08ecdd79b76e1292f8de1a6da903` |

CodeBERT 完整运行另有 best_step8500，因此模型根目录、best 与 final 也不能仅凭目录名互换。基础 CodeBERT 实际文件是 `pytorch_model.bin`，其哈希已补到 `INPUT_CONTRACT.json`；未把不存在的 safetensors 判成基础模型丢失。

## 8. 验收与下一步

**本次来源审查完成；数据路线通过。正式主复现尚未就绪，因为对齐分支和新统计协议还未执行。**

`VERIFICATION.json` 的 91 项一致性检查通过，覆盖：作者 Git blob、本地副本、C-UDA 文本、标签守恒、官方目标 ID、逐条分组指纹、代码/仓库组一致性、暴露层、记录的 tokenizer 输入契约，以及 5 个正式指针。大数据输入哈希已在初次全量扫描时记录，本验收没有重复跑完整扫描或计算任何方法收益。

下一步按以下顺序推进：

1. 建立隔离的 CoSQA adapter 与 SSQR 复现入口：query=`doc`，语料 ID 固定，ENTR 全词表 teacher forcing；同一位置缓存派生 ENTR、PROB、RAND。先补实现差异对应的契约检查，再进少量开发查询。
2. 在新数据结果出现前，固定论文主导或源码主导的重建选择。以论文的 n+1、m10、k3 为主线较清楚；公开源码的排位域、遮盖、优化器和种子差异分别登记。CodeBERT 主基线须完成 pooler+未归一化点积的匹配训练；现有 CLS 模型先用于迁移/兼容诊断，不能直接贴上作者复现标签。训练预算在执行前实测估算。
3. 先做无方法结果的分组角色和功效草案，再在永久归入 D 的小批查询上运行 CodeBERT/BM25。依据 D 的方差、可独立组数和事件数确定确认规模；不得拿官方 test500 调策略后再宣称一次性确认。

本次没有更换 strong84、gain-priority、threeway、semantic-pairwise UQ 或旧模型；`risk_certified=false` 与 NoRewrite 部署状态保持。审查通过不代表复现出了正收益，也不代表新风险策略已经通过。

## 9. 证据入口

- `DATASET_STATISTICS.json`：全量配对、相关性、重复与 QA 切分统计。
- `RETRIEVAL_CONTRACT.json`：作者检索语料、qrels、切分重叠与本地微调暴露层。
- `INPUT_CONTRACT.json`：字段语义、实际 tokenizer、哨兵、代码截断与 tokenizer 哈希。
- `OVERLAP_AUDIT.json` / `OVERLAP_SUMMARY.json`：完整输入清单、分层重叠及摘要。
- `QUERY_GROUP_INVENTORY.jsonl` / `MATCHED_SOURCE_RECORDS.jsonl`：逐查询组与命中来源行。
- `ROLE_FEASIBILITY.json`：排除官方 test 关联组后的可用量，不是已经冻结的 D/H。
- `LOCAL_CHECKPOINTS.json`：权重、训练完成状态、元数据与正式指针快照。
- `CUDA_LICENSE_SOURCES.json`：官方条款及固定版本；网络失败和后续成功分别保留。
- `VERIFICATION.json` / `AUDIT_DECISION.json` / `DATA_ACCESS_LEDGER.json` / `AUDIT_DELIVERY.json`：验收、结论、访问范围及交付哈希。
- `selected_source/`：Git blob 校验后的作者文件；`sources/`：公开 API 与论文快照。

原始来源：[CoSQA 论文](https://aclanthology.org/2021.acl-long.442/)、[CoCLR](https://github.com/Jun-jie-Huang/CoCLR/tree/14ebcacf9e9bc3e7109102632bc63047876f27d2)、[SSQR 论文](https://arxiv.org/html/2307.00267)、[SSQR 源码](https://github.com/RedSmallPanda/SSQR/tree/7fe5d780740eb4348500658ce6decf188a542d2b)、[C-UDA v1.0 正文](https://github.com/microsoft/Computational-Use-of-Data-Agreement/blob/a28ca06f6f8ecac0b5856ca6179ac49e55f00104/C-UDA-1.0.md)、[CodeSearchNet 许可说明](https://github.com/github/CodeSearchNet/tree/106e827405c968597da938f6b373d30183918869#licenses)。
