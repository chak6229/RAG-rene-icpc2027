# 最终动作条件风险校准：固定算法验证

问题：候选级harm学习总体与“收益筛选、核验更新、最高收益选择后真正执行”的动作总体不同。核验前strong84与核验后direct-harm的尺度也不同。本轮对最终动作做概率校准，不改变原候选选择，不以阈值搜索充当新模型。

固定基线为gain-priority、native tau=.1。每fold用既有三个fit128→held64完整pipeline产生的OOF192最终动作，保留实际非identity采用、一query一行、零收益保留。新增校准器在这些OOF动作上拟合；其自身训练预测不称OOF、不用于模型/门限选择。outer risk192与held128完全不参与训练。全部缓存属于历史开发数据，只进行开发验证，不追发独立证书。

固定方法：以原动作logit(harm)为offset的岭逻辑回归，包含stage、预测均值、均值竞争间隔、原排名位移/最坏幅度、核验证据对下降文档的支持等14维。每列以训练均值/尺度标准化；目标为非加权真实有害事件，禁止class balancing破坏概率基率。负对数似然求和加10/2倍系数平方，截距惩罚1/2；不调超参数。空集合用.5，单类用Jeffreys(nharm+.5)/(n+1)。概率计算截断仅防数值溢出。

主方法selected_structural；对照baseline、raw_tight（同动作raw harm<=.075）、stage_calibration（仅offset、stage）、selected_structural_uq（同结构附加3列共享member分歧，零新增UQ对照同维并置零）。新增门统一固定p<=.075且只过滤baseline已经选中的动作；拒绝即原query，禁止更换候选。所有query的原核验计划和成本保持，计数为已经执行核验的成本。

三个新增UQ列：最终实际chosen的member harm SD、member mean SD、候选共享member下chosen被实际allowed rival超过的正差均值；对核验后的动作，只用实际核验candidate更新成员。它们是共享bootstrap代理，不是完整T5后验、不声称新不确定性理论。

评价分三层：1同一baseline动作总体上的harm Brier/NLL/AUC及分阶段校准；2自然采用的正/负/零、DeltaMRR、固定baseline排序Top50采用、共享核验成本与成对repository置信区间；3主8fold/family精确CP目标10%、alpha=.05/8，失败即完整原排名回退。对照数值各自仅诊断，不聚合成共同证书。不合并跨fold重叠risk行发证。不读旧test，不生成T5/检索/CE、不调用旧神经网络。CPU仅拟合低维校准器和重放缓存。

即使本轮某项CP数值通过，historical_development强制certificate_issued=false。记录OOF128到outer192的上游训练规模差异；不将统计不显著直接解释成无效，也不把害数下降自动当有害率或收益提升。
