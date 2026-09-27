# 四项展示改进：实现与验证记录

本文件只记录本轮实际实现的能力，不把离线测试替身视为真实大模型效果。知识库与优惠均为项目演示内容，不是真实运营商政策。

## 1. 数据来源隔离

训练 Notebook `02_ml_pipeline.ipynb` 使用 `test_size=0.2`、`random_state=42`、按 `Churn` 分层划分。`business_tools/data_provenance.py` 读取原 CSV 的 `customerID` 与 `Churn`，按相同参数复现客户 ID 划分，不执行模型训练。Agent 页面把公开客户标为训练集或独立测试集，把其他 ID 标为自定义/未知。原来的五条 Agent 演示用例均为训练集客户；新增高、中、低风险各一条测试集客户。

注意：三条测试集用例只验证单客户操作链路。完整测试集 ROC-AUC 等指标仍以 Notebook 的全量评估为准。若 CSV 顺序、标签或 Notebook 划分逻辑改变，来源标记必须重新校验。

## 2. 自定义客户

`agent_app.py` 增加“添加自定义客户”模式，输入现有模型所需的 19 个字段，并核对电话/网络服务的关联取值。保存前拒绝不合格式或与原 CSV 冲突的客户 ID。`business_tools/customer_store.py` 把字段 JSON 存进独立 SQLite；`CustomerLookupTool` 先查该库，再查原公开 CSV。其余预测、RAG、优惠和最终事实校验仍沿用四个原工具，不另建第五个 Agent 工具。

数据库路径：`storage/customers/demo_customers.sqlite3`。它被 `.gitignore` 排除，不会进入训练集或向量库。当前只适用于本机演示，没有真实客户身份核验、修改/删除界面、权限与加密；请勿录入真实个人资料。

## 3. 评估扩充

- `evaluation/agent_tasks.json`：5 条训练集流程用例 + 3 条独立测试集高/中/低风险用例；`scripts/test_agent.py` 断言四工具顺序、风险等级和测试集来源。
- `evaluation/rag_cases.json`：扩为 13 条，覆盖合同、网络、技术支持、投诉、高中低风险、优惠限制与 2 条无依据问题；`scripts/test_rag.py` 断言来源与拒答。
- `tests/test_portfolio_upgrade.py`：验证自定义客户持久化、重复 ID 拒绝、完整离线 Agent 四工具链、来源复现和统一入口加载。
- 原 `evaluation/final_acceptance_cases.json` 的 15 项仍覆盖客户不存在、优惠金额边界、工具异常及人工确认/拒绝。此次不把旧结果自动改称“在线评估”。

曾出现的失败：问题“光纤客户反映网速不稳定时应先核查什么？”未达原 RAG 相似度门槛。2026-09-24 后续改进在 `rag/query_normalization.py` 加入受限的口语同义扩展，保留原相似度门槛与原文来源；该原句及“网络很卡”现均通过，股票价格反例仍拒答。该规则只覆盖已识别问法，不能宣称所有口语表达都已解决。

## 4. 统一演示入口

运行 `streamlit run main_app.py` 后，侧栏导航加载原 `app.py` 和 `agent_app.py`。这只是统一入口，没有复制业务实现；两个旧脚本仍可独立运行。原 `app.py` 不承担新增客户的存储或 Agent 编排。

## 数据流

```text
公开 CSV ──→ 客户查询 ──→ 已训练 Pipeline ──→ 风险等级
自定义 SQLite ─┘        │                         │
                       └──→ 单 Agent ←── 演示政策 RAG
                                      ←── Python 优惠规则
                                          ↓
                                  结构化方案 → 人工确认 → 审计 SQLite
```

## 本轮可复现命令

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe scripts/test_agent.py
.\.venv\Scripts\python.exe scripts/test_rag.py
.\.venv\Scripts\python.exe -m streamlit run main_app.py
```

真实 DeepSeek 调用需要环境变量 `DEEPSEEK_API_KEY`，且可能产生费用。本轮新增能力的回归使用离线替身，不声称已验证在线新客户流程。

当前复验（2026-09-24）：单元与页面测试 40/40、Agent 完整任务 8/8、RAG 问题 16/16、最终验收清单 15/15；`pip check` 无依赖冲突。统一入口已启动，`/_stcore/health` 返回 HTTP 200，并通过自动化测试从默认页面切换到 Agent 页。`scripts/verify_custom_customer.py` 的默认离线模式跑通虚拟新客户 `9000-NEWAA` 的保存、四工具方案、政策来源和拒绝审计。页面测试和健康检查不替代人工浏览器完整交互验收。

真实在线验收随后在 Windows 用户账户环境下完成。第一次模型成功生成方案，工具顺序是“客户查询→政策检索→风险预测→优惠计算”；旧验收脚本错误地要求预测必须早于政策检索，因此脚本判失败。这不是业务依赖错误，现改为只要求客户查询早于预测、预测早于优惠，且四类工具各调用一次。第二次真实在线调用通过：客户 `9000-NEWAA`，四工具均成功，预测高风险、流失概率约 `0.7077`，引用 `knowledge/demo_telecom/retention_policies.md`，优惠代码 `HIGH_M2M_HIGH_USAGE`、合计 `38.25`，人工拒绝状态写入临时 SQLite。模型两次工具顺序不同，说明在线路径存在选择波动；本验证仅覆盖这两次虚拟客户调用，不代表大规模稳定性。
