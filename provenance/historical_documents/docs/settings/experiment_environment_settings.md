# SSQR 外部复现实验环境与参数

## 实验环境

本材料集中列出主稿三个研究问题及相关敏感性实验的环境和参数。表 S16—S25 接续既有补充结果表 S1—S15；同一参数在不同实验中的取值分别保留。依据列 R01—R35 对应随附来源索引。

### 表 S16 软件与硬件环境

| 项目 | 配置 | 依据 |
| --- | --- | --- |
| 操作系统 | Windows，运行记录标识 Windows-10-10.0.26200-SP0（build 26200） | R01、R02 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU；运行记录显存 8,151 MiB | R01、R02 |
| GPU 驱动 | 577.05 | R01 |
| CPU | AMD Ryzen 9 8945HX with Radeon Graphics；16 核、32 逻辑处理器（本次硬件核对） | R34 |
| 内存 | 32 GiB（本次硬件核对） | R34 |
| Python | 3.10.11，64 位 | R01、R02 |
| PyTorch / CUDA | 2.7.0+cu128 / 12.8（PyTorch 构建及运行库） | R01、R02 |
| Transformers / Tokenizers | 4.57.6 / 0.22.2 | R01 |
| SentencePiece / Safetensors | 0.2.2 / 0.8.0 | R01 |
| NumPy / SciPy | 2.2.6 / 1.15.3 | R01、R02 |
| Scikit-learn / Joblib | 1.7.2 / 1.6.0 | R07、R09 |
| Threadpoolctl / Matplotlib | 3.6.0 / 3.10.9 | R01 |
| 训练精度 | 主 T5、CodeBERT 及有限训练对照使用 BF16；有限对照模型权重为 FP32 | R03、R05、R15 |
| 生成与检索精度 | 主 T5 生成/评分及神经检索使用 FP32，关闭 TF32；交叉编码器核验采用 BF16 autocast，输出转 FP32 | R02、R06、R20 |
| 推断状态 | 模型使用 eval；生成、评分与检索关闭 dropout；训练阶段保留各模型训练设置 | R17、R20、R31 |

软件、GPU 与驱动来自实验当时的运行记录。CPU 与内存于本次整理时核对，单独标明记录时点。CUDA 12.8 表示 PyTorch 使用的运行库版本。

## 模型训练

### 表 S17 主改写器与神经检索及核验模型的训练参数

| 参数 | 本地 CQC T5 | CodeBERT 检索器 | 交叉编码器核验器 |
| --- | --- | --- | --- |
| 基础模型 | T5-base | CodeBERT-base | CodeBERT-base |
| 训练数据 | 本地 CODEnn 来源前 1,000,000 条注释 | CodeXGLUE 251,820 对查询—代码 | 历史 large_dense_4096：4,096 条查询、16,384 对 |
| 训练目标 | 恢复单个连续遮盖片段，长度 max(1, round(0.15n)) | 查询到代码的完整批内交叉熵 | 二元相关性 BCEWithLogits |
| 轮数 / 更新 | 3 轮 / 93,750 次 | 2 轮 / 15,740 次 | 1 轮 / 512 次 |
| 微批量 / 梯度累积 | 8 / 4 | 32 / 1 | 8 / 4 |
| 有效批量 | 32 | 32 | 32 |
| 优化器 | AdamW | AdamW | AdamW |
| 学习率 | 0.001 | 0.00005 | 0.00002 |
| 调度 | 恒定 | 10% 预热（1,574 次）后线性衰减 | 恒定 |
| 权重衰减 | 0.01（执行代码默认值） | 0 | 0.01 |
| 梯度裁剪 | L2 范数上限 1 | L2 范数上限 1 | L2 范数上限 1 |
| 输入长度 | 512 | 查询、代码各 256 | 查询 64、代码 160，另加特殊标记 |
| 随机种子 | 101 | 123456 | 141414 |
| 权重选取 | 源验证损失最佳，第 92,000 次更新 | 主检索使用预定两轮最终权重 | 历史固定训练完成权重，H 推断前冻结 |
| 训练精度 | BF16 | BF16 | BF16 autocast |

T5 依据 R03、R04、R29；CodeBERT 依据 R05；核验器依据 R27、R28。核验器沿用历史训练权重，其训练来源及负例构造保留在来源代码中，未使用 H 标注重新训练。

主 T5 完成 93,750 次更新，实际生成使用源验证最佳的第 92,000 次权重。该训练与八种 1,024 次更新的敏感性配置分别报告。

## 候选生成与检索

### 表 S18 生成评分选择及检索参数

| 环节 | 设置 | 依据 |
| --- | --- | --- |
| 主插入位置 | 按空白划分 n 个词，分别在第 1 至第 n 个词后插入；每位置生成 1 条 | R06、R17 |
| 主生成参数 | greedy；do_sample=false；num_beams=1；num_return_sequences=1；max_new_tokens=19 | R06、R14 |
| 长度解释 | 主解码器长度至多 20（含起始标记）；输入超过 512 时报错，主流程不静默截断 | R17 |
| 批量与种子 | 早期/CoSQA/有限训练生成批量 8；StaQC 与新两权重生成批量 32、查询块 32；种子均 101 | R06、R07、R11、R14、R16 |
| T5 分词 | T5Tokenizer legacy=True；本地 T5-base SentencePiece；核对保存分词器词表 | R17 |
| ENTR / PROB | 同一生成标签序列；ENTR 为完整词表预测熵的非填充位置均值，低分优先；PROB 为逐词元对数概率均值，高分优先 | R17 |
| 评分标签 | `<extra_id_0>` + 片段 + `<extra_id_1>`，并包含 EOS；同分按原插入位置排序 | R17 |
| 候选选择与聚合 | k=1、2、3，截至候选池大小；mean / 标注辅助 best-k；不额外加入原查询 | R26、R14 |
| 重复与空片段 | 保留原始位置及其抽取概率；不以原查询补齐短候选池 | R11、R14 |
| 精确随机 / RAND20 | 精确随机为位置上的均匀不放回期望；RAND20 使用种子 2026101900—2026101919 | R06、R07、R26 |
| CodeBERT 表示 | pooler_output；不归一化；FP32 原始点积；查询与代码长度上限均为 256 | R05、R11 |
| CodeBERT 批量 | CoSQA H 编码 16；新两权重 StaQC 编码 32；排名得分分块 128 | R20、R14 |
| BM25 | k1=1.2，b=0.75；IDF=ln(1+(N−df+0.5)/(df+0.5))；词法本地实现 | R07、R18 |
| BM25 分词与计分 | 拆分小写/数字到大写的驼峰边界，再提取小写 [a-z][a-z0-9]*；查询词去重；文档不截断；FP64 | R18 |
| UniXcoder | `<encoder-only>`；查询 128、代码 256；对非填充词元（含特殊标记）的末层表示取均值，L2 归一化后 FP32 点积；批量 16 | R12、R31 |
| 排名规则 | 取 Top100；按分数降序、原文档 ID 升序处理平分；所有同条件对照固定语料与标注 | R06、R11、R12、R14 |

采样参数 temperature=1、top_k=0、top_p=1 适用于采样敏感性配置；主贪心设置不使用这些采样截断参数。主流程中的 k 表示所选候选数，生成时仍枚举全部指定位置。

## 模型身份

### 表 S19 实际使用权重及文件校验值

| 模型 | 身份与用途 | 权重文件 SHA-256 |
| --- | --- | --- |
| W1 本地 CQC T5 | T5-base 上独立 CQC 训练；源验证最佳，第 92,000 次更新 | 7acd4cb8379a1f5297bb978a748bc002b708ef5fd60a8b4737a6f8c7d2ebcaa5 |
| W2 公开 T5-base | 公开基础权重；两权重对照及有限训练共同初始化 | a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4 |
| W3 CodeBERT 检索器 | SSQR pooler/点积目标；两轮最终权重 | 0d533c04602916c22d1f13d282056f081b6bd04f9866ccc1ded46a0516e13e3d |
| W4 UniXcoder-base | 公开零样本权重；revision 5604afdc964f6c53782a6813140ade5216b99006 | 6732b60037b086557e4c731c3748c99b4fec735d9ffe3c4c70061fd142e6b9d9 |
| W5 CodeBERT 交叉编码器 | 历史 large_dense_4096 交叉编码器；完整 H 前冻结 | 3a0f26dcae197b2e7cb97b62349ccd743e4211a835a8e2ab2842376a7c4c2f39 |

W1—W3 的封存身份见 R14，W4 见 R12，W5 见 R07；本次逐一计算实际文件哈希并与封存记录一致。精确文件位置见随附来源索引。W1/W3/W5 以独立模型附件提供，文件清单、校验值及安装入口见 [模型附件说明](../MODEL_ASSETS_CN.md)。

两权重分析比较各权重自己的完整候选池。相同查询、语料、标注及检索器形成配对；不将候选池假定为跨权重相同。

## 生成与训练敏感性

### 表 S20 六种生成配置

| 配置 | 位置 | 解码 | 新增词元上限 | 含起始总长 |
| --- | --- | --- | --- | --- |
| greedy20_source | 1…n | 贪心 | 19 | 20 |
| greedy20_all | 0…n | 贪心 | 19 | 20 |
| greedy10_source | 1…n | 贪心 | 9 | 10 |
| greedy10_all | 0…n | 贪心 | 9 | 10 |
| sample10_source | 1…n | 采样 | 9 | 10 |
| sample10_all | 0…n | 采样 | 9 | 10 |

依据 R06。每域固定 512 条查询/组，共用配置间匹配查询。采样 temperature=1、top_k=0、top_p=1；每查询—位置以 SHA-256(seed=101, query_id, position) 确定随机流，每位置生成 1 条。贪心批量 8，采样批量 1。

### 表 S21 八种预算匹配训练配置

| 臂 | 优化器 | 调度 | 调度总长 | 长度 | 权重衰减 | 种子 |
| --- | --- | --- | --- | --- | --- | --- |
| L | AdamW | 恒定 | — | 512 | 0.01 | 101 |
| O | Adafactor | 恒定 | — | 512 | 0.01 | 101 |
| S | AdamW | 线性 | 100,000 | 512 | 0.01 | 101 |
| N | AdamW | 恒定 | — | 128 | 0.01 | 101 |
| W | AdamW | 恒定 | — | 512 | 0.0 | 101 |
| P | Adafactor | 线性 | 100,000 | 128 | 0.0 | 101 |
| PN | Adafactor | 线性 | 1,093 | 128 | 0.0 | 101 |
| PS | Adafactor | 线性 | 100,000 | 128 | 0.0 | 42 |

依据 R15、R16、R30。全部从同一 T5-base 初始化，共用固定 32,768 行训练材料和数据顺序，进行 1,024 次更新；微批量 8、累积 4、有效批量 32、初始学习率 0.001、梯度范数上限 1；BF16 计算、FP32 权重。

线性调度总长独立于实际 1,024 次更新预算。Adafactor 固定 scale_parameter=false、relative_step=false；各臂使用最终权重，无早停选择。评估同时报告自身生成池与固定历史池重评分视角。

## 固定动作与结果融合

### 表 S22 固定动作采纳与融合设置

| 项目 | 设置 | 依据 |
| --- | --- | --- |
| 样本与动作 | CoSQA D：1,490 条查询、512 个连通组；固定同一 ENTR Top1 动作 | R09 |
| 训练目标 | 固定动作 RR@100 减原查询 RR@100；每查询等权；目标标注不进入特征 | R09 |
| 特征块 | M0=27 维；MT=37 维；MR=61 维；MRT=71 维；完整列名及分区保留在协议 | R09 |
| 拟合 | StandardScaler + Ridge；alpha=1；solver=svd；含截距，不裁剪预测，不调参 | R09 |
| OOF 划分 | 复用按完整连通组划分的 5 折；只在训练折拟合标准化与模型 | R09 |
| 合格动作 | 同时改变查询文本和完整 Top100 排序；各预测头共用该可观测门槛 | R09 |
| Fixed20 | 每留出折采纳 floor(合格动作数/5) 个最高预测动作；含负分；同分按 query_id 升序 | R09 |
| 零阈值 | 只采纳合格且原始预测收益 >0 的动作；两个策略均以全部 D 查询为分母 | R09 |
| 重抽样范围 | 以已拟合 OOF 模型为条件；不在 bootstrap 内重新拟合或调整配额 | R09 |
| RRF 输入 | RRF3 为三个 ENTR 列表；RRF4 再加原查询；先按完整输入文本去重，再等权融合 | R10 |
| RRF 参数 | 常数 60，输入和输出深度 100；分数为 Σ1/(60+rank) | R10 |
| RRF 数值与平分 | 用精确有理数的等价整数排序；同分按文档 ID 升序；取检索文档并集的前 100 | R10 |
| 工作量 | 融合按去重后检索输入数计量，不解释为在线耗时；本对照无风险门 | R10 |

## 完整 H 执行流程

### 表 S23 冻结执行阶段的模型与决策参数

| 项目 | 设置 | 依据 |
| --- | --- | --- |
| D 中三个角色 | 收益拟合：953 查询/307 组；核验更新拟合：288/103；风险拟合：249/102；按完整组互斥 | R08 |
| 评估样本 | 完整 H 为 8,530 条查询、1,963 组；拟合和阈值冻结后再推断 H | R07 |
| 收益预测 | 71 维实际输入；StandardScaler + Ridge alpha=10；查询内候选按 1/候选数加权 | R07、R19 |
| 基础损害预测 | StandardScaler + LogisticRegression；C=0.1、lbfgs、max_iter=2000、random_state=261019 | R19 |
| 收益采纳 | 在同一 ENTR min(3,n) 候选中选预测最高者；预测 >0 且排名改变才采纳；预测裁至 [−1,1] | R19 |
| 核验配额 | 按正向基础预测收益排序，最多 min(floor(N/4),合格查询数) 条；同分按 query_id | R07、R19 |
| 核验证据 | 每查询 8 个不同代码；按原查询成功锚点、当前赢家/竞争候选及排名位移构建 | R07、R19 |
| 核验输入 | 原查询与代码成对；分别截至 64/160 词元，另加特殊标记；推断批量 32 | R20 |
| 核验精度 | 冻结模型 eval；BF16 autocast，分数转 FP32；不用改写文本替代原查询评分 | R20 |
| 更新模型 | 71+2+18=91 维；加权 StandardScaler + Ridge alpha=10；预测加上 0.5×残差预测，再裁至 [−1,1] | R19 |
| 风险模型 | negative/zero/positive 三分类，zero 为参考类，含基础概率 offset；20 维输入，屏蔽重复核验可用性列 | R07、R19 |
| 风险正则与优化 | 截距惩罚 1；核验指示系数惩罚 50；其余系数 10；L-BFGS-B，maxiter=2000，ftol=1e−11，gtol=1e−7 | R19 |
| 回退 | 阈值 0.075；超过时恢复原查询完整 Top100，不重新挑候选；已付核验成本保留 | R07、R19 |
| 数值零与小样本 | TOL=1e−10；残差拟合不足 30 行时使用零残差；风险样本不足 30 或少于两类时保留概率 offset | R19 |

完整 H 流程的 Ridge alpha=10，与表 S22 的固定动作对照 alpha=1 分别保留。风险阈值沿用历史开发中的固定保守工作点 0.075，target_harm=0.1 是采用动作损害率的设计目标；二者不构成校准保证。来源及角色见 [风险工作点说明](../RISK_WORKING_POINT_CN.md)，经验损害计数不构成总体风险认证。

HGB 次要控制依据 R07：squared_error；max_iter=200，learning_rate=0.05，max_leaf_nodes=7，max_depth=3，min_samples_leaf=30，l2_regularization=10，max_bins=63，early_stopping=false，random_state=20261019。

## 配对推断与样本口径

### 表 S24 各项研究的比较族重抽样与样本

| 分析 | 查询/组与语料口径 | 比较族 | 重抽样次数 / 种子 | 主要设置 |
| --- | --- | --- | --- | --- |
| CoSQA 指定复现 | 限额 H：7,718 查询/1,963 组；每组最多 16 条 | 1 | 10,000 / 261019 | 双侧 95% 双区间；须严格落入 ±0.005 且至少 20 个变化组 |
| 完整 H 阶段 | 完整 H：8,530 查询/1,963 组 | 描述性 | 10,000 / 261019 | 阶段均值区间；收益保留率比值另用种子 261020 |
| 同池 B/G | AdvTest 2,048/594；CoSQA D 1,490/512；双检索器 | 24 | 10,000 / 20261020 | ENTR、PROB 的 B/G 及直接分量差 |
| StaQC 确认 | 13,312 查询/组；16,384 代码语料 | 2 | 10,000 / 202610202 | CodeBERT ENTR B/G；PROB、BM25 描述性 |
| UniXcoder 扩展 | 复用 CoSQA D 与前述 StaQC 样本 | 12 | 10,000 / 202610203 | 两域 B/G 及相对 CodeBERT、BM25 的配对分量差 |
| 匹配两权重 | AdvTest、CoSQA 每域 512 查询/组 | 24 | 20,000 / 202610204 | 两域×两选择器×B/G×权重内及直接差 |
| 新两权重确认 | 新 StaQC 25,600 查询/组；32,768 代码语料 | 6 | 20,000 / 202610215 | 两权重 ENTR B/G 及权重配对直接差；无等效检验 |
| 固定动作采纳 | CoSQA D 1,490/512 | 8 | 10,000 / 20261020 | 两条件增量×两采纳策略×两检索器 |
| RRF 融合 | CoSQA D 1,490/512 | 4 | 10,000 / 20261020 | RRF3/RRF4 相对原查询，两个检索器 |
| 有限训练对照 | AdvTest、CoSQA 每域 512 查询/组 | 28 | 10,000 / 20261020 | 保持已封存主要比较族和两候选池视角 |

依据 R07、R09—R16、R21、R26。均值以查询为权重单位；重抽样保留完整连通组或仓库组，在同一研究内共享配对重抽样。校正比较采用 Bonferroni 聚类 t 区间与组 percentile bootstrap 区间。

B/G 方向判断同时要求两种校正区间在同一方向排除零，并满足至少 20 个非零组。各研究比较族分别保留，不因统一展示合并总体。CoSQA ±0.005 为其指定等效界；新两权重规划中的 0.005 是精度目标。

有限训练对照的重抽样种子依据实际分析入口 R35；其余对应 R07、R09—R16、R21、R26。

## 人工评审与相关性诊断

### 表 S25 人工评审及补审的固定设置

| 项目 | 设置 | 依据 |
| --- | --- | --- |
| 抽样框与层 | CodeBERT 完整 H8530；在收集标签前固定 8 个互不重叠的层，每层抽取 16 条 | R22 |
| 固定抽样规则 | 按 SHA-256 伪随机顺序选取；盐值 rene-mechanism-review-20261020-v1；完整拼接规则见代码 | R22、R23 |
| 审阅规模 | 128 条查询；两人分别独立评估 383 条不同改写与 327 个代码面板 | R33 |
| 盲化信息 | 隐藏方法、分数、排名结果、层和目标角色；语义评估与代码相关性评估分开 | R33 |
| 语义类别 | 含义保持、含义偏移、不可读、无法判断；每项附理由 | R33 |
| 代码类别 | 相对原查询评为 2/1/0 或未知；未知不记为 0；展示面板不等于完整语料重标 | R33 |
| 一致性统计 | 语义精确一致率与名义 Cohen κ；代码含未知的名义 κ，以及双方可判子集的线性加权 κ 与覆盖率 | R33 |
| 加权事件 | 使用原始 Nh/nh 分层权重；事件可以重叠；构造同时层超几何界，报告固定 H8530 的 95% 区间 | R22、R33 |
| Top-3 补审 | 原查询与三候选 Top-3 并集 610 面板；266 沿用原判，每人新增 344 项，覆盖同批 126 条查询 | R25、R33 |
| 补审主诊断 | 固定 RIDGE_verified、RR@3、2 级相关、原八层权重；同文本共享未知相关性赋值 | R25 |
| 敏感性口径 | 另报告 1/2 级相关、未加权、各阶段及 RR@1；保留两位审阅者标签与分歧 | R25 |
| 识别界与抽样包络 | 样本识别界处理未判相关性；有限 H8530 的抽样包络另用超几何反演与 Bonferroni；Neyman/Welch–Satterthwaite t 为近似 | R24 |
| 覆盖范围 | 四个“审阅者×阈值”主策略组合另报同时包络；不声明全部 48 组合同时覆盖；区间不含标注误差 | R24、R25 |

依据 R22—R25 及当前中文稿 R33。原始独立标签、补充判断和分歧分别保留；相关性诊断的 RR@3、样本识别范围和抽样包络不替换完整 H 的 MRR@100。

## 来源索引

以下为工作区/复现包内的相对路径；完整 SHA-256 及配置对象另见配套 JSON。

- R01: `results/rene_factor_confirmation_20261020/RUNTIME_ENVIRONMENT.json`
- R02: `results/rene_weight_confirmation_20261021/RUNTIME.json`
- R03: `models/t5-cqc-codenn-1m/training_config.json`
- R04: `models/t5-cqc-codenn-1m/best/best_checkpoint.json`
- R05: `results/rene_ssqr_alignment_20261019/TRAINING_CONTRACT.json`
- R06: `results/rene_ssqr_bridge_20261019/PROTOCOL.json`
- R07: `results/rene_cosqa_confirmation_20261019/DESIGN.json`
- R08: `results/rene_cosqa_confirmation_20261019/STAGE_ROLES.json`
- R09: `results/rene_core_step_b_20261020/PROTOCOL.json`
- R10: `results/rene_multicandidate_fusion_20261020/PROTOCOL.json`
- R11: `results/rene_factor_confirmation_20261020/DESIGN.json`
- R12: `results/rene_third_retriever_20261020/DESIGN.json`
- R13: `results/rene_t5_weight_factors_20261020/DESIGN.json`
- R14: `results/rene_weight_confirmation_20261021/DESIGN.json`
- R15: `results/rene_cqc_training_fidelity_20261020/TRAIN_PROTOCOL.json`
- R16: `results/rene_cqc_training_fidelity_20261020/EVAL_PROTOCOL.json`
- R17: `experiments/rene_ssqr_alignment_20261019/t5.py`
- R18: `experiments/rene_cosqa_pilot_20261019/ranking.py`
- R19: `experiments/rene_cosqa_confirmation_20261019/policy.py`
- R20: `experiments/rene_cosqa_confirmation_20261019/cache.py`
- R21: `experiments/rene_cosqa_confirmation_20261019/statistics.py`
- R22: `results/rene_mechanism_review_20261020/PROTOCOL.json`
- R23: `experiments/rene_mechanism_review_20261020/common.py`
- R24: `paper/rene2027/relevance_sampling_uncertainty.md`
- R25: `paper/rene2027/relevance_sensitivity_results.md`
- R26: `results/rene_budget_aggregation_20261020/PROTOCOL.json`
- R27: `experiments/action_witness_uq_20261014/gpu_data.py`
- R28: `results/uq_pilot_20261014/action_witness_uq/data/ce_large_dense_4096_train_status.json`
- R29: `train_t5_cqc.py`
- R30: `experiments/rene_cqc_training_fidelity_20261020/train.py`
- R31: `experiments/rene_third_retriever_20261020/encoder.py`
- R32: `paper/rene2027/implementation_fidelity.md`
- R33: `paper/rene2027/ieee_submission/论文中文版_IEEE双栏.docx`
- R34: `results/rene_experiment_settings_20261026/HARDWARE_RECHECK.json`
- R35: `experiments/rene_cqc_training_fidelity_20261020/analyze.py`

### 权重文件

- W1: `models/t5-cqc-codenn-1m/best/model.safetensors`；SHA-256 `7acd4cb8379a1f5297bb978a748bc002b708ef5fd60a8b4737a6f8c7d2ebcaa5`
- W2: `models/t5-base/model.safetensors`；SHA-256 `a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4`
- W3: `models/codebert-ssqr-pooler-2epoch-20261019/model.safetensors`；SHA-256 `0d533c04602916c22d1f13d282056f081b6bd04f9866ccc1ded46a0516e13e3d`
- W4: `models/unixcoder-base-pinned-20261020/pytorch_model.bin`；SHA-256 `6732b60037b086557e4c731c3748c99b4fec735d9ffe3c4c70061fd142e6b9d9`
- W5: `results/uq_pilot_20261014/action_witness_uq/data/ce_large_dense_4096.pt`；SHA-256 `3a0f26dcae197b2e7cb97b62349ccd743e4211a835a8e2ab2842376a7c4c2f39`
