# BaoStock 数据能力审计（第一轮小样本）

审计状态：**BOUNDED PROBE PASS; FULL PIT AUDIT OPEN**  
接口版本：`00.9.30`  
查询完成时间：`2026-09-05T16:28:44.194564+00:00`

本报告只说明有限样本的接口行为，不构成完整数据源验收。原始响应的路径、行数与SHA-256保存在
`data/raw/manifests/baostock_capability_audit.json`；CSV位于被Git忽略的`data/raw/audit/`。

## 历史指数成分

| index | requested | source_update | rows | unique | count_check | asof_check |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | 2012-06-29 | 2012-06-25 | 300 | 300 | PASS | PASS |
| CSI300 | 2016-06-30 | 2016-06-27 | 300 | 300 | PASS | PASS |
| CSI300 | 2020-06-30 | 2020-06-29 | 300 | 300 | PASS | PASS |
| CSI300 | 2024-06-28 | 2024-06-24 | 300 | 300 | PASS | PASS |
| CSI500 | 2012-06-29 | 2012-06-25 | 500 | 500 | PASS | PASS |
| CSI500 | 2016-06-30 | 2016-06-27 | 500 | 500 | PASS | PASS |
| CSI500 | 2020-06-30 | 2020-06-29 | 500 | 500 | PASS | PASS |
| CSI500 | 2024-06-28 | 2024-06-24 | 500 | 500 | PASS | PASS |

同日CSI300/CSI500集合关系：

| date | intersection | union |
| --- | --- | --- |
| 2012-06-29 | 0 | 800 |
| 2016-06-30 | 0 | 800 |
| 2020-06-30 | 0 | 800 |
| 2024-06-28 | 0 | 800 |

跨样本日期确有成员变化，说明接口没有简单返回当前成分：

| index | interval | added | removed |
| --- | --- | --- | --- |
| CSI300 | 2012-06-29 → 2016-06-30 | 118 | 118 |
| CSI300 | 2016-06-30 → 2020-06-30 | 126 | 126 |
| CSI300 | 2020-06-30 → 2024-06-28 | 96 | 96 |
| CSI500 | 2012-06-29 → 2016-06-30 | 271 | 271 |
| CSI500 | 2016-06-30 → 2020-06-30 | 269 | 269 |
| CSI500 | 2020-06-30 → 2024-06-28 | 272 | 272 |

当前判断：行数、唯一性和`updateDate <= requested_date`均需通过；仍要对照独立来源抽查具体调入/调出事件，
并覆盖指数调整日前后相邻交易日，之后才能关闭PIT成分审计。

## 历史行业

| code | requested | source_update | industry | asof_check |
| --- | --- | --- | --- | --- |
| sh.600000 | 2012-06-29 | 2012-06-25 | 金融保险业-银行业 | PASS |
| sh.600000 | 2020-06-30 | 2020-06-29 | J66货币金融服务 | PASS |
| sz.000001 | 2016-06-30 | 2016-06-27 | J66货币金融服务 | PASS |
| sz.000001 | 2024-06-28 | 2024-06-24 | J66货币金融服务 | PASS |

样本显示历史查询会返回不晚于请求日的更新日期，但2012年的行业名称与后期编码体系不同。这是明显的
分类口径断点。同期真实组合与placebo仍可使用当日分类，但HistoricalZ可能在分类制度切换时发生机械跳变。
正式实现必须定位全部分类口径切换日期，并比较“当日原生分类”与稳定映射/切断窗口的稳健性。

后续专项审计已将代码前缀切换夹定在2012-12至2013-01，并形成可执行处理规则；详见
`docs/industry_taxonomy_audit.md`。Residual Sync影响仍需在收益数据接入后量化，因此该项尚未关闭。

## 行情、复权和单位

- adjustflag=2与不复权数据的成交量、成交额、换手率、涨跌幅、PB和交易状态一致：
  `{"volume": true, "amount": true, "turn": true, "tradestatus": true, "pctChg": true, "pbMRQ": true, "isST": true}`。
- `amount / volume`得到的VWAP位于不复权日内最低价和最高价之间：`PASS`。
- 2017-05-25公司行为前后，前复权/不复权价格比发生变化：`PASS`。

实施含义：收益序列可以使用经核验的复权价格或BaoStock涨跌幅；成交额、VWAP、流通股数估计和执行模拟
必须配合不复权价格。`volume`与`turn`可用于`float_shares=volume/(turn/100)`，但还需跨更多证券和公司行为验证。

## 尚未关闭的问题

1. 历史成分的外部事件级交叉核验。
2. 行业分类体系切换的全市场日期与稳定映射。
3. PB是否严格按财报真实公开日进入历史日线；在此之前VALUE不进入确认性主研究。
4. 退市股票、长期停牌、ST切换、上市日和异常零成交覆盖。
5. 20只股票的复权、换手和流通股估算抽查。
6. 全量抓取速度、重试、断点续传与服务端限速。
7. P7需要每个控制器决策日已经可知、可交易且合计为1的CSI 800复制篮子权重。当前只有
   CSI 300/CSI 500历史成员和CSI 800价格指数收益；二者不能反推出复制权重。中证800等权指数
   另有独立代码000842与独立编制方案，因此不能用800只成分简单等权来替代000906基准。

第7项的输入契约是`decision_at、available_at、code、weight`，要求`available_at <= decision_at`、
同一决策日证券唯一、权重非负且严格合计为1。入口脚本为
`scripts/40_validate_benchmark_replication.py`；数据到位前，股票级含成本控制器必须停止，不能输出
看似可交易的结果。公开口径依据见[中证800等权指数事实表](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000842factsheet.pdf)
和[编制方案](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/186_000842_Index_Methodology_cn.pdf)。

## 放行决定

本轮允许继续开发数据适配器和小样本工程流程；**不允许冻结最终研究协议，也不允许开始确认性结果分析**。
只有上列问题按预设验收规则完成后，P0数据能力审计才可关闭。
