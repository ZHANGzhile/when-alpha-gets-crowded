# When Alpha Gets Crowded：实施设计 V2

版本：2.0  
日期：2026-09-05  
依据：原始项目报告与第二轮评审意见。V1 已归档至 `docs/archive/implementation_design_v1.md`。

## 0. 文档状态

这是正式实施前的设计基线，不是实证结果。当前尚未完成数据能力验证、数据下载、模型训练和回测，因此本文不预设 crowding 有效，也不把任何预期写成结论。

数据能力审计完成后，把本文中的主定义、例外规则和参数写入 `research_protocol.md`，生成协议哈希并冻结。冻结发生在查看确认性测试结果之前。若数据源无法满足某项定义，只能依据预先写明的降级规则调整，并留下版本记录。

## 1. 项目目标与证据边界

研究单位为 `factor × week`。股票日频价格、成交和估值数据是构建因子状态的底层输入，研究对象是一个既有 Alpha 的可靠性。

最终主问题为：

> 在普通市场风险、因子自身状态以及组合的一般风险特征之外，因子特有的异常集中结构是否能够提供额外的 Alpha 脆弱性信息？如果一个已经处于高集中状态的因子进一步遭遇流动性或交易压力，它是否更容易失效？

项目分为三个证据层次：

1. **测量**：真实因子组合相对同期匹配组合，是否表现出异常的共同运动与策略趋同，并且这种异常相对自身历史也处于高位？
2. **预测与机制**：结构代理能否在市场、因子和一般组合风险之外提高样本外预测；结构代理与压力触发的交互是否进一步提高预测；该信息是否在事件发生前仍然存在？
3. **应用**：用针对主动多头风险训练的样本外概率调整主动暴露，是否优于固定低暴露、波动控制和一般风险控制？

即使所有实验通过，也只能支持“因子特有的异常市场结构具有与潜在拥挤一致的预测信息”。公开数据不能直接观测机构持仓，本项目不声称测量了真实拥挤，也不作因果结论。

## 2. V1 研究边界

| 项目 | 冻结方向 |
| --- | --- |
| 市场 | 中国 A 股 |
| 股票池 | 各历史时点 CSI300 与 CSI500 成分并集，经当时可知状态过滤 |
| 原始数据区间 | 2012-01-01 至 2026-08-31；前期作为预热，不保证产生模型样本 |
| 决策频率 | 每周最后一个交易日，在当日所需数据可用后生成状态 |
| 因子 | Momentum、Short-Term Reversal、Low Volatility、Value |
| 主预测对象 | 动态 Long-Short 因子未来 20 个交易日 Crash Event |
| 主模型 | 合并四因子的正则化 Logistic，含因子固定效应 |
| 主评价指标 | 样本外 Log Loss 改善 |
| 主识别比较 | M3 对 M2 |
| 主机制比较 | M4 对 M3 |
| 提前预警 | 5、10 个交易日 lead；0 日为同期可得预测基线 |
| 应用 | 因子多头相对 CSI800 基准的主动暴露控制 |

VALUE 只有在 PB 的 point-in-time 可用性通过审计后进入主分析；否则按协议降为稳健性因子。V1 不增加因子池、不使用神经网络、不接券商，也不声称建成可直接交易的生产系统。

## 3. 三类信息必须分离

### 3.1 结构性拥挤代理 C

`C[f,leg,t]` 表示因子某一侧组合在 t 时点出现了相对匹配组合和自身历史均异常的共同持仓结构。它是结构代理，不是真实持仓拥挤。所有结构量先在 Long、Short 两侧独立生成；LS 主模型保留两侧字段，不先平均成一个会掩盖来源的分数。Active Long 模型只读取 Long 侧。

主组成项：

- `ExcessResidualSync`：真实组合行业残差收益同步性减去匹配组合均值。
- `ExcessEigenConcentration`：真实组合相关矩阵最大特征值占比减去匹配组合均值。
- `ExcessStrategyConvergence`：真实因子间持仓重叠减去匹配置换下的预期重叠。

每项经过两层校正：

\[
ExcessFeature_{k,f,t}=Actual_{k,f,t}-Mean(Placebo_{k,f,t})
\]

\[
CFeature_{k,f,t}=HistoricalZ_f(ExcessFeature_{k,f,t})
\]

主结构指数为方向统一后的等权均值：

\[
C_{f,t}=Mean_k(CFeature_{k,f,t})
\]

历史标准化仅使用 t 之前、过去 252 个交易日范围内的周末观测，约 50 个点，最低 40 个有效观测；当前点不进入基线。不得在周频表上直接使用 252 行并称为一年。若任一主组成项缺失，主指数记为缺失，不因缺项动态改变权重；分项结果始终保留。

同时冻结只含 Residual Sync 与 Eigen Concentration 的 `C_core`，用于检验只有四个因子时 Strategy Convergence 是否主导结果。模型可以对 HistoricalZ 按训练期规则裁剪，但原始值、placebo 均值与方差、Excess、PlaceboZ 和 HistoricalZ 均不可覆盖。

`CrossFactorOverlap` 模块正式命名为 **Strategy-Convergence Proxy**。主要输出 Excess Jaccard、Weighted Degree、Maximum Edge Weight 与 Stock Crowding Count。四节点网络图只用于展示。

### 3.2 一般组合风险 G

`G[f,t]` 描述这个组合当下的一般风险状态，不要求它相对 placebo 异常，包括：

- raw residual correlation；
- raw eigen concentration；
- raw turnover level / dispersion；
- raw illiquidity level与恶化程度；
- 组合覆盖率、集中度和可选的规模/行业暴露诊断。

G 的作用是建立更强的基准：即使已知该组合当前风险很高，C 是否仍提供额外信息。

### 3.3 压力或平仓触发 S

`S[f,leg,t]` 描述短期压力的创新或突变，而不是长期形成的结构。主成分先冻结为：

- `turnover_shock`：成员最新5日换手相对截至 t-5 的此前60日中位数/MAD，先在股票层标准化，再取组合中位数；
- `turnover_sync`：有效成员中 `turnover_shock > 1.5` 的比例，同时报告 coverage；
- `liquidity_shock`：股票当前 ILLIQ20 相对此前252日历史基线的异常值，当前20日不得进入基线，再取组合中位数；
- `factor_return_shock`：负向5日因子收益相对过去历史的异常值；
- 尚未被 Market State 覆盖的组合层短期压力。

市场收益、市场波动、市场流动性与 breadth 等仍属于 Market State。S 中不重复复制同一个市场变量。每个压力项按该因子自己的过去信息进行历史标准化，方向统一为“越高越紧张”，主 `S` 为预先冻结的透明等权组合，分项保留。

G 是风险水平，S 是近期变化或触发。部分原始数据来源相同，但字段、窗口和经济含义必须分开。M2 纳入 G 和 S 的全部主效应，保证 M4 的增量只来自交互，而不是遗漏的 S 主效应。

## 4. 数据正确性与时间语义

### 4.1 Point-in-Time 审计

BaoStock 是候选主数据源，不因接口支持 date 参数就默认历史数据合格。正式下载前抽查：

- 早期年份与指数定期调样前后；
- 分红除权、停牌复牌、ST、退市；
- 行业变更；
- PB 更新和缺失；
- 历史查询结果是否真的随查询时点变化。

成分、行业和估值记录至少保存 `requested_at`、`effective_from/to`、`source_update_at`、`available_at`、`retrieved_at` 与来源。抓取时间不能冒充历史发布时间。无法可靠恢复 PIT 的字段不得进入主模型。

原始响应按抓取批次只追加保存，并生成请求清单、行数、校验和和失败状态。空响应与成功的零行结果分开。数据不合格时先补充来源；无法修复时，在查看确认性结果前按协议缩短时期或降低结论范围，不用当前成分和当前行业回填过去。

### 4.2 时间字段与执行口径

任何样本均明确：

- `information_cutoff`：特征实际可知的最晚时点；
- `decision_at / issue_at`：生成预测的时点；
- `execution_at`：应用组合最早可以成交的时点；
- `label_start_at` 与 `label_end_at`：结果区间。

周末状态必须等所需日线数据可得后生成。理论研究序列可以采用下一交易日收盘到收盘口径，但必须标注理想成交假设。应用回测使用下一交易日开盘的日频执行近似：旧持仓承担成交前隔夜收益，新持仓获得成交后收益；目标权重、实际权重、拒绝/延迟原因和成本进入执行账本。

### 4.3 有效样本由依赖关系计算

动量需要约 252 个交易日预热，结构指数还需要自身历史，Crash 阈值再需要约三年成熟收益历史。完整主标签很可能约从 2016 年开始。系统逐因子输出首个信号日、首个结构指数日、首个阈值日与首个可训练日，不人工写死开始年份。

数据截止日之后尚未走完 20/60 日区间的标签必须为空。不同期限独立判断成熟度。

## 5. 因子与组合引擎

主定义：

\[
MOM_{i,t}=P_{i,t-21}/P_{i,t-252}-1
\]

\[
REV_{i,t}=-(P_{i,t}/P_{i,t-20}-1)
\]

\[
LOWVOL_{i,t}=-Std(r_{i,t-59:t})
\]

\[
VALUE_{i,t}=-log(PB_{i,t}),\quad PB>0
\]

处理顺序为：历史成分 → 当时可知的正常交易、非 ST、上市天数及数据充分性过滤 → 因子原值 → 横截面 MAD 处理 → 行业内 percentile rank → 全股票池选择前后 10%。并列、取整和小行业回退规则写入配置。

行业内排名只降低行业倾斜，不等于严格行业中性。输出行业暴露诊断；严格中性版本作为稳健性。

收益信号使用经公司行为核验的收益序列；交易金额和流通市值计算使用对应实际价格。若确认 `volume` 为股、`turn` 为百分数，则 `float_shares=volume/(turn/100)`；turn 为零、缺失或异常时不计算。该估计只有通过抽样核验后才可用于 size 匹配。

Long/Short 每个 leg 调仓时等权，周内权重随收益漂移。停牌证券保留持仓与可得估值，不因无法交易而从收益中消失；退市损失必须保留。研究多空序列明确 `+1 long / -1 short` 的名义约定，并在全链路保持同一尺度，不能声称在 A 股可直接执行。

## 6. 特征与 placebo 引擎

在决策时点 t 固定当时成员，读取截至 t 的历史窗口。不得读取未来成员或未来行情。Long 与 Short 分开计算，之后再按协议形成 factor 层聚合。

| 原始指标 | 计算规则 | 归属 |
| --- | --- | --- |
| Residual Sync | 60 日行业残差相关矩阵非对角均值 | raw→G；placebo excess→C |
| Eigen Concentration | 最大特征值 / trace(Corr) | raw→G；placebo excess→C |
| Turnover Level | 股票自身历史换手统计的组合聚合 | G |
| Turnover Shock | 当前换手相对此前 60 日中位数/MAD | S |
| Turnover Synchronization | 高换手冲击股票比例及共同变化 | S |
| Liquidity Level | 20 日 `abs(return)/amount` 聚合 | G |
| Liquidity Shock | 流动性相对历史基线的恶化 | S |
| Strategy Convergence | 因子多头的 Jaccard 等结构 | raw诊断；匹配超额→C |

换手基线不含当天；MAD=0 记无效，不造无穷值。主版本使用 raw MAD，是否乘 1.4826 仅作冻结的敏感性版本。ILLIQ 中 amount=0 既不设为零也不设为无穷，按停牌/异常规则单独处理。ILLIQ20 的历史基线建议为此前 252 个交易日、至少 126 个有效值。

60 日、约 80 只股票的相关矩阵可能秩不足。最大特征值仍可计算，但必须使用合法半正定矩阵；主版本预先选择收缩相关估计器，样本相关矩阵为稳健性。不能把 pairwise 缺失拼接出的非半正定矩阵直接用于特征值指标。

### Matched placebo

每个 factor × date × leg 生成 B=100 个同期随机组合：来自同一合格股票池、股票数相同、无放回抽样，匹配行业与滞后流动性；size 仅在流通市值通过核验后加入。流动性匹配使用截至 t-20 的历史水平，避免把当前待检验的流动性冲击一并匹配掉。

每次抽样记录 seed、实际成员、候选股票快照哈希、算法版本、交集、匹配误差、分组降级和有效样本。不能满足精确匹配时，只按协议中的层级合并规则降级，并显式标记。主版本允许随机组合抽到真实成员，排除真实成员作为敏感性。行业计数要求一致；流动性/规模平衡阈值在协议中冻结，未达标的 row 不进入主样本。每个可用 row 至少需要80个有效 draw 且 placebo 标准差大于零，不能用 epsilon 或填零修复。

Strategy Convergence 的 placebo 必须在同一 draw 中联合生成四个因子的匹配组合，再计算 pairwise Jaccard；不能把四个独立的单因子 null 均值拼成 overlap null。抽取10%的日期以 B=500 重算，检查 B=100 下的排名与符号稳定性。

所有指标保存 `raw`、`placebo_mean/std`、`actual-minus-placebo`、`placebo_z`、`historical_z`、有效样本数和窗口边界。

Placebo 有两种用途：

1. 同期横截面校正，用于构造 C；
2. 随机/打乱策略完整经过 `Membership → Return → Feature → 自身历史阈值与Outcome → Prediction` 流程，用于检验 `IncrementalValue_real > IncrementalValue_placebo`。每条伪策略必须有跨周连续的 `pseudo_strategy_id`，不能每周抽一个匿名组合后把它当作动态策略。

行业内打乱排名必须重新生成成员、特征和标签；时间错位检验保留连续时间块与四因子的同期依赖，不做逐行 IID shuffle。

## 7. 两个研究目标族

### 7.1 Research Target：因子可靠性

主目标是动态 Long-Short 策略：未来仍按固定规则每周再平衡，回答当前状态能否预测策略未来失效。

主确认性端点：

\[
Crash^{LS,dynamic}_{f,t}=1[R^{LS,20d}_{f,t}<Q^{past}_{10\%,f,t}]
\]

`Qpast` 按因子分别计算，只使用截至决策时点已经完全成熟的历史周度20日动态因子结果：历史标签结束日必须不晚于信息截止日。主窗口为此前756个交易日范围内的周度决策样本，至少104个成熟历史观察；不足时标签不可用。实际事件率不保证每年正好10%。

次要结果包括未来 20 日收益、MDD、RankIC breakdown、5/60 日期限及 5%/15% 阈值稳健性。

固定成员结果在 t 冻结股票及起始股数，下一交易日执行并持有20日，中途不重新选股或恢复等权，权重自然漂移；停牌、公司行动与退市沿用同一持仓账本。它回答当前 footprint 是否对应当前这批股票的未来风险，是 mechanism test。Dynamic 与 Fixed 各用自身成熟历史分布定义阈值，但使用相同 feature rows、fold 和模型配方，并同时比较连续收益与MDD。动态有效、固定无效时，应警惕模型捕获的是更广的因子 regime；固定有效、动态无效时，只能说明当前持仓脆弱。

Long 与 Short 分别输出未来收益、MDD、结构/压力特征和对 LS 损失的贡献，回答 breakdown 来自哪一侧。

### 7.2 Application Target：主动多头风险

应用对象为因子多头相对 CSI800 基准的主动倾斜。单日主动收益为：

\[
a_d=r^{Long}_d-r^{Benchmark}_d
\]

它用于 Tracking Error 和 Information Ratio。累计主动财富与主动回撤使用相对净值：

\[
RelativeNAV_h=\frac{\prod_{d=1}^{h}(1+r^{Long}_d)}{\prod_{d=1}^{h}(1+r^{Benchmark}_d)}
\]

`Active20dReturn=RelativeNAV_20-1`；`ActiveMDD` 在包含初始值 1 的 RelativeNAV 路径上计算非负回撤幅度；`ActiveCrash` 使用当时可知的历史 Active20dReturn 10% 分位数。

控制器优先使用针对 `ActiveCrash` 训练的概率。LS Crash 模型可作为迁移实验，但只有稳定预测 Active Long 风险时才可用于仓位控制，不能自动搬用。

## 8. 模型矩阵与确认性检验

所有主模型使用完全相同的样本行、标签、fold、预处理规则和调参预算。缺失处理不能让后续模型通过换样本获得优势。

| 模型 | 信息集合 | 回答的问题 |
| --- | --- | --- |
| M0 | Market State | 普通市场状态有多少预测力？ |
| M1 | M0 + Factor State | 因子自身状态是否增加信息？ |
| M2 | M1 + Generic Portfolio Risk G + Stress S 主效应 | 已知市场、因子、组合风险与当前压力后能预测多少？ |
| M3 | M2 + Structural Crowding C | 因子特有异常结构是否仍有增量信息？ |
| M4 | M3 + 预先指定的 C × S | 高结构集中遇到压力时是否额外脆弱？ |

Market State 至少包括市场20日收益、20/60日波动、换手/流动性、breadth 和横截面 dispersion。Factor State 至少包括 trailing factor return、factor volatility、current drawdown、RankIC state 和 signal dispersion。具体窗口在协议中冻结。

M4 的主交互只有一个：标准化复合 C × 标准化复合 S，以控制小样本自由度。分项交互属于次要分析。Logistic 交互系数只说明 log-odds 尺度，必须同时展示低/高 C 与低/高 S 组合下的样本外预测概率和边际变化。

确认性比较：

\[
\Delta LL_C=LogLoss(M2)-LogLoss(M3)
\]

\[
\Delta LL_{C\times S}=LogLoss(M3)-LogLoss(M4)
\]

正值表示后者改善。M3 对 M2 是唯一确认性主识别检验；M4 对 M3 是预先指定的关键机制次级检验，避免把多个比较同时称为 primary。主端点为动态 LS 20d Crash。每个日期先对当周可用因子的损失取平均，再对日期取平均得到 OOS Log Loss，避免缺失较少的因子获得更高权重；不按年份或因子挑选最好结果。计分前仅为数值稳定将概率截断到 `[1e-6,1-1e-6]`。

由于 `Excess=Actual-PlaceboMean`，在 M2 已含 Actual 时，M3 的改善有可能部分来自新增的 placebo context。必须同步运行诊断模型：

- `M2_context = M2 + 对应的PlaceboMean`；
- `M3_context = M2_context + C`。

主文同时报告 M3-M2 与 M3_context-M2_context。不得在同一个线性模型中同时放入完全共线的 Actual、PlaceboMean 和 Excess。只有结构增量超过 context baseline，且真实因子的增量强于完整伪策略分布时，才加强因子特有异常结构的解释。

辅助指标为 PR-AUC、Brier、Calibration、ROC-AUC、最高预测风险 10% 状态的实际 Crash Rate。Future MDD、CVaR、RankIC breakdown、未来平均收益及其他期限属于次要/稳健性结果。

主预测器为含因子固定效应、共同斜率的 L2 正则化 Logistic，不使用 class weighting，也不对主 Logistic 额外校准。正则强度只在训练期内部时间验证选择；各模型使用相同搜索网格与预算，并报告共同固定正则强度的敏感性。因子特定斜率和单因子模型只用于异质性分析。OLS 与 Quantile Regression 提供关联和尾部分布证据。LightGBM 仅在主结果之后探索非线性，若校准则只能用训练期内部OOS预测拟合校准器。PCA、survival 和外部持仓验证均为后置模块。

## 9. Walk-forward、purge 与提前量

可用训练起点由预热决定。建议 2020–2023 作为开发期走步评估，2024–2026-08-31 作为冻结后的最终时期；最终时期只纳入截至数据终点已经成熟的标签。

每年采用 expanding walk-forward，在该年首个预测日前重训一次。进入冻结测试期后可以在下一年重训时加入上一年已经成熟的标签，这是可执行的扩展训练，但协议和搜索空间不再改变。训练、验证和测试边界根据实际 `label_start_at/label_end_at` 清除：训练结果区间不能跨入验证决策时点，验证结果区间不能跨入测试；最终期起点前凡标签延伸进入最终期的开发样本也必须删除。20日与60日任务分别 purge。标准化、缺失填补、历史分箱、模型、PCA和校准器均只在相应历史段拟合。

### Lead-time experiment

定义 L=0、5、10 个交易日。对于目标窗口 `t+1:t+20`，L 日预警的 `issue_at=t-L`。整个信息集合 M0–M4、成员、历史标准化、模型与校准器都只能使用 issue_at 当时可知的数据，不是把最终特征表机械移动几行。

Lead 模型使用 issue_at 当时已知的历史阈值：

\[
Crash^{(L)}_{f,t}=1[R^{20d}_{f,t}<Q^{past}_{10\%,f,issue\_at}]
\]

这使信号在 issue_at 就可以形成完整决策。训练样本也按相同 lead 重新构造和 purge。若仅 L=0 有效，结论只能称 contemporaneous/near-term risk indicator；L=5 或10仍有效，才支持 early-warning 表述。

除同期交互 `C_t×S_t` 外，机制稳健性还检验 `C_{t-5}×S_t` 与 `C_{t-10}×S_t`，更接近“结构已存在，随后遇到压力”。这些仍是预测关联，不据此作因果平仓解释。

## 10. Crash Event Study

事件研究用于检查时间顺序，不能替代样本外比较。

保存每个正标签的真实结果区间；同一因子内所有重叠区间先合并为一个 episode，若两个 episode 的 `-8w:+4w` 观察窗仍重叠则继续合并，避免同一次行情重复计权。事件零点预先定义为该 episode 第一次实现20日历史尾部阈值跌破的日期，而不是事后最深谷底；以最大回撤谷底对齐仅作敏感性图。同期多个因子可各自形成 episode，但推断按共同 calendar-event cluster 处理。

对每个独立 episode 对齐 `-8w:+4w`，绘制 C、S、因子收益、因子波动、流动性压力及市场压力。同步匹配同因子、相近年份、相同市场波动五分位且前后12周无Crash的非事件日期；报告匹配率与 event-control 路径，避免把一般危机形态误当成拥挤演化。

重点检查 C 是否先升高、S 是否更接近事件发生时上升。若二者只在事件后上升，不支持 early-warning mechanism。置信区间按 episode 或日期块重采样，并报告独立 episode 数量。

## 11. 测量有效性

结构代理必须经过三层检查：

1. **Cross-sectional abnormality**：Actual 与同期 matched placebo 的 Excess/PlaceboZ。
2. **Time-series abnormality**：Excess 相对该因子自身过去的 HistoricalZ。
3. **Discriminant validity**：报告 C 与市场波动、因子波动、因子回撤、S、G 的相关性及条件关系。

Discriminant validity 不使用全样本残差化来制造独立性。描述性相关与回归必须标明限制；预测意义由 M3 对 M2 的严格样本外比较判断。训练期内另估计 `C ~ Market + Factor + G + S` 并记录相关系数与 R²；高度重合只触发解释降级，不允许据此重新改指标追求显著性。还需通过 M2_context/M3_context 分解改善究竟来自因子异常偏离，还是匹配参照本身带来的额外市场信息。

如能获得可靠 PIT 的基金持仓集中度、共同机构持仓或其他公开持仓信息，可作外部构念验证，但这些数据不进入主预测模型。无法可靠获得时直接省略，不削弱主流程。

## 12. 统计推断与结果报告

由于20日标签在周频上重叠、四个因子同期相关，独立信息量远低于表面行数。

- 主预测证据为 M2/M3/M4 在完全相同 OOS 预测上的成对差值；先形成每周跨因子平均的损失差，再对周进行推断。
- 对周损失差做10,000次成对移动块bootstrap，四个因子保持在同一日期块中；主块长13周，8/26周为敏感性，报告点估计、95%区间与差值不大于零的比例。
- 回归时序使用适当 HAC；四个因子不足以把普通 factor-cluster 标准误当作主要推断。
- 报告独立 Crash episode 数量、事件集中度和逐年表现，不删除不显著年份。
- 主假设与探索性结果分开；分项、因子异质性和多种阈值处理多重比较并展示失败结果。

只有 `ΔLL_C>0` 且主bootstrap的95%区间下界大于0，才称确认性识别成立。M4、lead-time与Fixed结果使用同样的成对区间但保持次级身份；分项C/S、多期限和多腿结果按分析族使用BH-FDR。完整placebo另报告真实增量在伪策略分布中的Monte Carlo rank。单因子报告回答异质性，不给每个因子独立搜索大量模型后挑最好者。若只有 M2 优于 M1，则结论是一般组合风险有效、因子特有拥挤没有增量信息；placebo 与真实因子近似则承认无法区分拥挤与一般组合脆弱性。

## 13. 应用控制器

控制器只读取 `oos_predictions_active`，禁止使用 in-sample 概率。M2、M3、M4均针对 Active Long Target 训练，并使用同一套仓位规则。

为隔离“信息更好”与“长期持仓更少”，确认性应用比较采用相同的主动暴露预算。根据每个模型在 issue_at 之前的 OOS 预测历史计算风险分位，冻结分层：

- 低于历史60%风险分位：`w=1.00`；
- 60%–80%：`w=0.75`；
- 80%–90%：`w=0.50`；
- 最高10%：`w=0.25`。

阈值只使用过去预测，不能用整个测试期分位数。M2/M3/M4使用相同层级，因而具有近似一致的长期暴露预算。报告原方案 `w=clip(1-p,0.25,1)` 作为概率型敏感性；它在约10%事件率下可能只产生轻微调整，不作为唯一应用证据。

比较六类组合：

1. 满主动暴露；
2. 固定低主动暴露`w=0.825`，等于主分位层级在均匀风险分位下的事前期望暴露；
3. 波动率控制：截至决策日收盘的最近60个交易日Active Long日收益，至少40项，年化目标10%，`w=clip(0.10/σ,0.25,1.00)`；
4. M2 一般风险控制；
5. M3 拥挤感知控制；
6. M4 拥挤×压力控制。

核心经济比较是 M3/M4 控制器对 M2。另给出每个因子与测试期M3实际平均暴露相同的事后固定暴露诊断；该路径使用全期M3权重均值，结果字段固定标记`ex_ante_executable=false`，不进入六类事前策略的应用成功判定。

固定低暴露与波动率控制都在下一交易日开盘生效，并使用与M2/M3/M4相同的股票目标、成交限制和成本路径。波动率只读取决策时已经实现的Factor Long减CSI800日收益；不足40项或窗口内缺失时不生成仓位，下游共同样本门会停止而不会补值。

应用主指标冻结为10bp成本后的净 Information Ratio；同时要求 Active MDD 与 Active CVaR 不恶化，并完整报告 Annual Return、Vol、Sharpe、Sortino、MDD、Worst20d、Active Return、Tracking Error、Information Ratio、Active MDD、Crash Episode Loss、平均/分布主动暴露与换手。5/20bp为敏感性。

应用组合以 `w × FactorLong + (1-w) × Benchmark` 表示。CSI800 指数收益本身不是可交易资产；未明确基准复制篮子或历史可用跟踪工具前，结果称为理想化基准配置实验。

先在股票层形成最终目标权重，再和收益漂移后的执行前权重比较。若 c 为单边成本：

\[
Cost_t=c\sum_i|w^*_{i,t}-w^-_{i,t}|
\]

同时保存 half-L1 但不拿它重复收费。因子换股、基准调整和控制器改变 w 引起的全部成交都进入账本；停牌/涨跌停导致的未成交保留旧仓，只对实际成交收费。日线无法充分模拟开盘冲击与容量，因此结果不是实盘可行性证明。

## 14. 工程与数据表

```text
config/
  data.yaml
  factors.yaml
  measurement.yaml       # C/G/S 定义、窗口、placebo
  outcomes.yaml          # LS、fixed、active、阈值
  experiments.yaml       # M0-M4、fold、lead、metrics
  controller.yaml
research_protocol.md
src/alpha_crowding/
  data/ factors/ measurement/ outcomes/
  experiments/ placebo/ backtest/ reporting/
data/raw/ data/interim/ data/processed/
runs/<run_id>/
tests/
report/
```

核心表：

| 表 | 关键字段 |
| --- | --- |
| universe_history | index、code、effective_from/to、available_at |
| industry_history | code、classification、effective_from/to、available_at |
| daily_bars | date、code、原始OHLC、收益字段、volume、amount、turn、状态、单位 |
| factor_membership | decision_at、factor、code、leg、signal、rank、target_weight、filter_reason |
| factor_returns | date、factor、leg、return_convention、daily_return、weights_before/after |
| structural_features | decision_at、factor、leg、feature、raw、placebo_mean/std、excess、historical_z、valid_n |
| generic_risk_features | decision_at、factor、leg、feature、value、window、valid_n |
| stress_features | decision_at、factor、leg、feature、raw、historical_z、value |
| placebo_diagnostics | decision_at、factor、leg、seed、match_balance、fallback_level |
| placebo_feature_values | decision_at、factor、leg、draw_id、feature、value、valid_n、quality_flags |
| pseudo_strategy tables | pseudo_strategy_id、membership、returns、outcomes、oos_predictions |
| factor_outcomes | issue_at、factor、target_family、membership_mode、horizon、label_start/end、threshold、value |
| model_dataset | issue_at、factor、target_id、M0…M4字段、fold_id、maturity_flag |
| oos_predictions | issue_at、factor、target_id、model_id、lead、probability、train_label_cutoff |
| model_design_manifest | fold_id、model_id、精确列清单、sample_mask_hash、scaler区间、train_label_cutoff |
| execution_ledger | date、code、target/actual_weight、trade_value、cost、reject_reason |

每次运行保存 `run_id`、时间戳、Git commit、配置与协议哈希、数据清单哈希、seed、数据截止日、fold与指标。任何图表都能回溯到对应运行与结果表。

## 15. 实施阶段与门槛

| 阶段 | 交付 | 放行条件 |
| --- | --- | --- |
| P0 数据能力审计 | 来源矩阵、20股/关键事件抽查、字段时点报告 | PIT、复权、停牌、估值和成员历史可用性明确 |
| P1 协议冻结 | `research_protocol.md`、配置、协议哈希 | 主端点、指标、M0-M4、C/G/S、LS/Active、事件和lead规则冻结 |
| P2 数据与因子 | 历史股票池、日线、四因子成员/收益/IC/换手 | 手算样本通过；无未来成分；权重漂移和停牌账本正确 |
| P3 测量引擎 | C/G/S分表、matched placebo、有效性诊断 | 至少10个factor-date逐项追溯；匹配误差在阈值内 |
| P4 标签与切分 | dynamic/fixed/active标签、fold、purge、lead数据集 | 时间区间断言和标签成熟性全部通过 |
| P5 确认性研究 | M0-M4、OOS比较、block bootstrap、lead、event study | M3/M2和M4/M3同样本；零结果完整保留 |
| P6 证伪与稳健性 | 完整随机策略、打乱、错位、窗口/阈值/股票池敏感性 | Real与placebo增量差异可报告；失败项不隐藏 |
| P7 应用与报告 | Active模型、六类控制器、成本、最终图表与正文 | 仅OOS预测；公平基准；主动风险指标齐全；可复现 |

先用少量年份与2个因子做工程冒烟，只验证数据和数值计算，不选择因子或事件年份。正式分析仍覆盖冻结的四因子与完整时期。

LightGBM、PCA、survival、外部持仓验证和高级网络图只有在 P0–P7 主闭环完成后再做。时间不足时首先删除这些模块，不能删除 M2/M3、M4、lead-time、placebo 或 Active Target。

## 16. 必须自动化的检查

- 扰动未来数据不改变历史特征与历史预测。
- 历史成分和行业使用正确的 as-of join。
- 周频一年窗口按252交易日范围而非252周。
- 调仓后权重随收益漂移，停牌与退市不消失。
- 除权收益和实际交易价格不混用。
- Future MDD 的 NAV 路径包含初始值1。
- 历史阈值只读取已经成熟的20日结果。
- Purge 根据真实标签区间而不是日期大小。
- L=5/10 的全部信息截止于 issue_at。
- M2/M3/M4使用完全相同的样本与fold。
- S 主效应在M2中，M4只新增预先定义的交互。
- M2_context/M3_context 使用相同 sample mask，并检查设计矩阵无完全共线列。
- Placebo 完成与真实因子相同的特征、标签和预测流程。
- 每个结构 row 的有效 placebo draw、匹配质量和 Monte Carlo 稳定性达到协议门槛。
- 控制器只读取OOS Active预测，成本按实际成交收费。

## 17. 结果解释规则

| 观察结果 | 允许的结论 |
| --- | --- |
| M3优于M2，M4优于M3，且5/10日lead有效 | 因子特有异常结构具有增量提前预警信息，结构与压力交互进一步提高脆弱性预测 |
| M3优于M2，M4没有改善 | 结构代理具有预测信息，但当前数据不支持明确的trigger交互机制 |
| M3不优于M2，M2优于M1 | 一般组合风险有预测力，因子特有结构没有增量信息 |
| 只有L=0有效 | 当前指标是同期或近端风险指标，不能称提前预警 |
| Fixed有效、Dynamic无效 | 能识别当前成员风险，不能预测持续再平衡策略失效 |
| Dynamic有效、Fixed无效 | 可能捕获更广的因子regime，而非当前成员集中结构 |
| 仅Momentum有效 | 机制具有因子异质性，不是普遍规律 |
| Placebo与真实因子近似 | 当前公开市场代理无法区分潜在因子拥挤与一般组合脆弱性 |
| M3/M4 Active控制器不优于M2 | 拥挤信息没有提供额外应用价值，即使研究端可能成立 |

工程成功、测量成功、研究成功、机制成功与应用成功是五个独立判断。项目的最低成功标准不是高 Sharpe，而是能够在固定数据、代码与协议下，清楚回答异常结构是否具有超越一般风险的增量信息，并如实报告被证伪的情况。
