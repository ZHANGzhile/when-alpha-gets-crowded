# 当 Alpha 变得拥挤

[English](README.md) · [简体中文](README.zh-CN.md)

这是一个可审计、可复现的量化研究系统，用于检验：股票因子组合内部出现的异常共同结构，能否在普通市场风险和组合风险之外，提前解释未来因子脆弱性。

项目研究历史 CSI 300/CSI 500 股票池中的动量、反转、低波动，以及通过时点审计后才允许进入主分析的价值因子。系统严格区分三类信息：

- **结构性拥挤 `C`**：经同期匹配组合校正的行业残差同步性、特征值集中度和跨因子策略趋同。
- **一般组合风险 `G`**：组合本身的相关性、集中度、换手和非流动性水平。
- **平仓压力 `S`**：近期换手、流动性和因子收益冲击。

这些指标只是市场隐含结构代理，不能解释成真实机构持仓。确认性问题是：模型已经包含 Market State、Factor State、`G` 和 `S` 后，`C` 是否仍能提高严格样本外预测能力。

## 已实现内容

- 带原始响应检查点的 CSI 300/CSI 500 历史周频成分采集。
- 对五个并购换股相关的 499 只成分截面进行有官方来源、逐行留痕的修复。
- 全历史股票日线下载、数据契约、质量验收和失败后串行重试。
- 周频因子信号、确定性 Long/Short 选股、下一交易日生效及周内权重漂移。
- 精确 60 个交易日窗口的 Ledoit-Wolf 行业残差相关与特征值集中度。
- 每个 `factor × date × leg` 生成 100 个行业计数一致、匹配滞后流动性的同期 placebo 组合。
- 在同一 draw 中联合生成全部因子的 Strategy Convergence 空分布。
- 只读取过去 252 个真实交易日范围、排除当前值的 Historical Z-score。
- Market、Factor、`C`、`G`、`S` 状态表，并保留有效性、覆盖率和窗口边界。
- Dynamic LS、Long/Short 分解、Active Long/CSI800 以及 Fixed-membership 前瞻结果路径。
- 年度扩展窗 M0–M4 Logistic 比较、训练期内净化时间验证和成对移动块 bootstrap。
- M2/M3/M4 严格按同一因子的历史 OOS 概率生成仓位，预热期为52周，并在下一交易日生效。
- 可交易 CSI 800 复制权重的 PIT 数据契约；只有成分集合或简单等权会在股票级控制器前被拒绝。
- 零主动暴露的CSI 800可执行复制对照；使用相同开盘约束和成本引擎，分别审计毛/净Tracking Error与毛主动收益偏差，超过预设阈值时硬停止。
- 股票级控制器执行引擎：成本后目标求解、每日持仓漂移、现金、买卖方向开盘限制、实际成交成本和逐证券拒单账本。
- 开盘调仓的正确收益归属：旧持仓承担隔夜收益，成交后的目标持仓承担日内收益，两段连乘重建收盘到收盘收益。
- P7绝对与相对CSI 800指标，包括Sharpe、Sortino、MDD、最差20日、Tracking Error、Information Ratio、Active MDD/CVaR、Active目标Crash Episode Loss、暴露、换手和拒单。
- 协议哈希门：P0 数据门未关闭时，不允许生成确认期结果。

完整困难、判断过程和解决办法记录在
[`docs/implementation_journal.md`](docs/implementation_journal.md)。研究协议位于
[`research_protocol.md`](research_protocol.md)，详细架构位于
[`implementation_design.md`](implementation_design.md)。

## 当前研究状态

代码主路径已经连接到 M0–M4 样本外比较，但项目仍处于研究实施阶段，以下数据和证伪门尚未关闭：

- 退市证券的最终结算收益必须接入真实来源，不能前向填充；
- PB 的历史发布时间通过 PIT 审计前，Value 不进入主分析；
- 历史行业分类切换还需完成实际指标影响量化；
- 历史成分事件还需独立来源交叉核验；
- 5/10 日提前量、Crash 事件研究、M2/M3 placebo-context、26/52周错位、判别效度和完整连续伪策略路径已经具备受协议门保护的生产入口。100条伪策略各自拥有稳定成员、收益账本、leave-one-out结构C、同口径G/S与因子状态、逐策略成熟LS Outcome，以及独立调参的年度OOS M2/M3比较。
- 控制器的OOS暴露日程、基准数据门、纯基准复制质量门和受限股票级会计引擎已经实现。P7生产运行仍等待Tushare参考数据通过真实验收；固定低暴露和波动率控制的参数也仍需在协议中冻结。

原始数据、运行清单、日志、模型输出、本地报告和来源 PDF 不进入 Git。仓库保存可复现代码、候选冻结配置和审计记录。

## 安装

要求 Python 3.11 或更高版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

P7参考数据下载器使用可选的Tushare客户端：

```powershell
python -m pip install -e ".[controller-data]"
$env:TUSHARE_TOKEN = "仅保存在本机环境中的token"
```

token只从进程环境读取，不会写入仓库文件或运行清单。账户需要拥有`index_weight`和`stk_limit`接口权限。

如果 Python 位于其他目录，可在使用 PowerShell 脚本前把环境变量 `ALPHA_CROWDING_PYTHON` 指向对应可执行文件。

## 验证

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_checks.ps1
```

行为检查覆盖 PIT 连接、精确交易日窗口、组合账本、matched placebo、标签成熟、时间净化和 walk-forward 训练。检查只用于验证实现正确性，不能替代真实研究结果。

## 生产流水线

按阶段运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_production.ps1 -Stage membership
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_production.ps1 -Stage daily
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_production.ps1 -Stage measurements
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_production.ps1 -Stage analysis
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_production.ps1 -Stage falsification
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_production.ps1 -Stage controller
```

按依赖顺序运行完整流水线：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_production.ps1 -Stage all
```

网络任务和长任务会在 `data/` 下写入原子检查点。Outcome 和模型阶段只有在 `protocol_freeze.json` 有效且其中登记的文件哈希未改变时才运行。

## 目录结构

```text
config/                  候选冻结定义与阈值
docs/                    数据验收报告与实施日志
scripts/                 审计脚本和编号生产阶段
src/alpha_crowding/      数据、因子、测量、结果、实验和回测代码
tests/                   行为与时间完整性检查
implementation_design.md 完整实施架构
research_protocol.md     确认性研究协议（当前为 DRAFT）
```

## 可复现规则

1. 本地保留原始响应和校验和，清洗数据不得覆盖原始数据。
2. 每个决策时点只能使用当时已经可知的信息。
3. Long 与 Short 的结构量在模型层之前保持分离。
4. 结构性拥挤、一般风险和短期压力必须属于不同特征族。
5. 不得用常数修补缺失退市结算、零 MAD 或无效 placebo 分布。
6. 构造确认期结果前必须冻结协议并校验哈希。

本仓库是研究系统，不是实盘交易或券商接入系统。
