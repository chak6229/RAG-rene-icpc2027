# 阶段收缩与原排名锚点风险：事前固定缓存实验

主问题：前轮最终动作结构门降低损害但丢正收益；核验后的少量事件与训练规模偏移可能使阶段校正不稳定。本文不估计或校正已看的outer标签比例，直接用fit-only阶段系数强收缩和更具体的可见锚点证据检验。

固定原gain-priority tau=.1的实际选中动作、原核验计划、原候选/排名/成本，任何拒绝只恢复完整原排名，不重选。训练使用每fold完整inner128→held64三份OOF192中的非identity采用动作，每query一行，零收益保留。风险校准与held完全不拟合或选择参数。该OOF只形容上游预测，校准器训练分数不称其自身OOF。

预定七方法：baseline原方案；previous_structural读取旧去重模型；stage_shrunk同13列结构、仅stage标准化系数岭50（其余10、截距1）；anchor_shrunk主方案，在stage_shrunk上新增六列；anchor_shrunk_uq再开原三分歧列；threeway_anchor同非UQ输入的负/零/正offset softmax；threeway_positive使用同三分类模型及风险门，另要求fit-only符号全局幅度所得预测signed gain>0。不在这些方法中按outer结果选门或追认证书。

六新增列只读可见原/执行排名与实际已付费CE witness：原Top1 RR下降、原rank2–10 RR下降总量、全部降位RR质量未被CE覆盖比例、实际CE中同时存在升降位文档标记、Top1降位幅度乘其相对最佳升位文档CE支持sigmoid差、rank2–10按损失加权的同类冲突除以该区间全部损失质量。Top1/Top2之外缺核验很常见，未知质量明确编码，计划外的缓存CE禁止读取。文档ID只用于匹配，query ID/gold/原RR不作特征。CE无值置0并由stage/覆盖/配对标记表示缺失。

全部二元概率以原logit(harm)为offset。标准化只用fit动作，stage惩罚50作用于标准化系数，不声称原始0→1固定先验强度。主模型六锚点列均岭10，UQ三列均10。不类别重加权；空/单类二元用Jeffreys平滑全局harm。

三分类固定zero参考类。offset为native pneg、余量按fit-only正/零Jeffreys比例分配；所有行在看标签前用同一fit比例，不以该行真实符号建offset。负/正两套系数、同正则与标准化，参考系数为0。少于两类用Dirichlet .5全局频率；符号幅度只用该fit采用动作真实正/负幅度均值，缺符号时幅度0（因此positive规则会回退）。

所有新风险门统一P(harm)<=.075，主认证仍采用内有害率<=10%、单项alpha=.05/8；各方法各自八项只作历史诊断，不形成全七方法共同95%证书。固定采用50/100、同池Brier/NLL、配对repo CI及严重损害回退另行评价。所有历史数据均不能签发新证书。不新增T5/检索/CE/旧神经网络调用，不改正式strong84/UQ/gain-priority/V5配置。另有matched-scale独立诊断会改变上游路由/动作/成本，绝不混为本轮同动作同预算效果。

## 模型输入去重修正

自查发现六列中ce_up_down_pair_available在全部fit中与verified完全相同，造成阶段惩罚绕行。因此新目录统一固定mask[20]=0，将此列保留为诊断字段不入模，新增锚点有效列为5。其他输入、正则、门限、全部数据和评估不变。首轮anchor_stage_risk_20261016完整保留；以后以本去重实现为主，不按收益择版本。
