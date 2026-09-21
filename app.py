# ============================================================
# 电信客户流失预测与智能挽留分析系统
#
# 本文件是 Streamlit 应用入口。运行方式：
# python -m streamlit run app.py
#
# 页面流程：读取数据 -> 加载 Pipeline -> 展示 EDA -> 评价模型
# -> 预测单个客户 -> 生成 AI 挽留建议 -> 批量筛选并导出高风险客户。
# ============================================================

# ---------- 1. 导入依赖 ----------
import os
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError
)
from streamlit.errors import StreamlitSecretNotFoundError
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# set_page_config 必须在其他 Streamlit 页面命令之前调用。
# layout="wide" 让指标、表格和表单在桌面浏览器中拥有更充足的横向空间。
st.set_page_config(
    page_title="电信客户流失预测与智能挽留分析系统",
    layout="wide"
)

# ---------- 2. 读取数据并加载模型 ----------
# 使用 app.py 所在目录拼接文件路径，而不是依赖终端当前目录。
# 这样即使从其他目录启动 Streamlit，也能正确找到数据和模型文件。
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "telco_customer_churn.csv"
MODEL_PATH = PROJECT_DIR / "churn_pipeline.joblib"


@st.cache_data
def load_customer_data(data_path):
    """读取客户 CSV，并在文件不变时复用缓存结果。

    Streamlit 调整任何控件时都会重新执行 app.py。使用 cache_data 后，应用
    不需要在每次重新执行时重复读取同一份 CSV。data_path 发生变化或缓存被
    清除时，函数会重新读取文件。
    """
    return pd.read_csv(data_path)


@st.cache_resource
def load_model(model_path):
    """加载训练好的 Pipeline，并在后续页面重跑时复用同一个模型对象。

    cache_resource 适合缓存模型这类需要反复使用、但不应在每次交互时重新
    创建的资源。模型文件由 02_ml_pipeline.ipynb 训练并保存。
    """
    return joblib.load(model_path)


# 原始 CSV 用于展示客户数据、计算业务指标，以及进行批量预测。
df = load_customer_data(DATA_PATH)

# 完整 Pipeline 内含类别编码、数值标准化和逻辑回归分类器。
loaded_pipeline = load_model(MODEL_PATH)

# 页面开头先说明系统用途，帮助用户理解后续各模块。
st.title("电信客户流失预测与智能挽留分析系统")
st.write(
    "本系统用于分析电信客户流失情况、预测客户流失概率、筛选高风险客户，"
    "并通过 DeepSeek 生成供人工参考的客户挽留建议。"
)
st.success("模型加载成功")

# ---------- 3. 清洗数据并准备模型输入 ----------
# Churn 是原始标签（Yes/No）。ChurnFlag 是用于统计流失率的 0/1 标签。
df["ChurnFlag"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# 为模型准备清洗后的数据副本。
# TotalCharges 原始数据中有少量空白值；这些客户的 tenure 为 0，按 0 处理。
# TotalCharges 中有少量空字符串。先去除首尾空格，再用布尔值定位这些行。
total_charges_clean = df["TotalCharges"].astype(str).str.strip()
is_blank_total_charges = total_charges_clean.eq("")

# 复制原始数据，清洗时不直接修改 df，便于保留原始数据用于展示。
df_clean = df.copy()
df_clean.loc[is_blank_total_charges, "TotalCharges"] = 0
# 转成数值类型，保证模型和后续统计可以进行数值运算。
df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"])

# 数值特征：数值大小本身有意义，Pipeline 会对它们进行标准化。
numeric_features = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
]

# 类别特征：表示客户属于哪一类，Pipeline 会对它们进行独热编码。
categorical_features = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

# X 是模型输入特征。customerID 只是客户标识，不代表行为，因此不参与训练。
# Churn 和 ChurnFlag 是标签或统计字段，也不能放入 X，否则会造成信息泄漏。
feature_columns = numeric_features + categorical_features
X = df_clean[feature_columns]

# y 是模型要预测的标签：Yes 表示 1，No 表示 0
y = df_clean["Churn"].map({"Yes": 1, "No": 0})

# ---------- 统一管理分类阈值和风险等级边界 ----------
# CLASSIFICATION_THRESHOLD：把流失概率转换成 0/1 预测类别时使用的阈值。
# 例如某位客户的流失概率是 0.45：
# - 使用默认阈值 0.50 时会预测为未流失（0）；
# - 使用本项目选定的阈值 0.40 时会预测为流失（1）。
# 本项目在阈值实验中选择 0.40，原因是客户挽留场景更关注 Recall，
# 即希望尽可能识别真正可能流失的客户，并愿意接受一定的额外联系成本。
CLASSIFICATION_THRESHOLD = 0.40

# HIGH_RISK_THRESHOLD：风险等级中“高风险”的起点。
# 低于 0.40 为低风险，0.40（含）到 0.60（不含）为中风险，
# 0.60（含）及以上为高风险。风险等级仅用于项目演示，并非企业真实标准。
HIGH_RISK_THRESHOLD = 0.60

# 以下两个常量用于识别需要提供给 AI 的客户风险信息。
# 它们只是当前项目的演示规则，不是模型自己学习得到的分类阈值，
# 也不代表真实电信企业已经采用的业务标准。
#
# NEW_CUSTOMER_TENURE_MONTHS：使用时长不超过 12 个月时，视为仍处于新客户阶段。
# HIGH_MONTHLY_CHARGES：月费达到 75 时，标记为“月费较高”这一待关注信息。
# 75 的参考依据是本项目 EDA 中流失客户的平均月费约为 74.44。
NEW_CUSTOMER_TENURE_MONTHS = 12
HIGH_MONTHLY_CHARGES = 75.0

# DeepSeek 官方接口兼容 OpenAI SDK，因此可以使用 OpenAI 客户端发送请求，
# 只需要把 base_url 指向 DeepSeek，并使用 DeepSeek 平台生成的 API Key。
# 模型名称集中写成常量，后续模型升级时只需要修改这一处。
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"


def get_risk_level(probability):
    """根据一位客户的流失概率返回演示用风险等级。

    参数
    ----
    probability : float
        模型输出的流失概率，通常位于 0 到 1 之间。
        例如 0.72 表示模型估计该客户有 72% 的流失风险。

    返回值
    ------
    str
        返回“低风险”“中风险”或“高风险”之一。

    使用方法
    --------
    单个客户可以直接调用：get_risk_level(0.72)。
    整列客户概率可以通过 Series.apply(get_risk_level) 逐行调用。

    备注
    ----
    这里的边界来自项目演示规则，不代表真实企业的业务标准。
    """
    if probability < CLASSIFICATION_THRESHOLD:
        return "低风险"
    elif probability < HIGH_RISK_THRESHOLD:
        return "中风险"
    else:
        return "高风险"


def extract_risk_factors(customer):
    """从一位客户的资料中提取可供 AI 参考的风险信息。

    这个函数只负责识别客户当前具备哪些“需要关注的特征”，不负责计算
    流失概率，也不直接生成挽留建议。流失概率仍然由训练好的机器学习
    Pipeline 计算；DeepSeek 会综合概率、风险等级和这里提取的风险信息，
    再生成自然语言建议。

    参数
    ----
    customer : pandas.Series
        一位客户的特征数据。调用时使用 new_customer.iloc[0]，即可从只有
        一行的 DataFrame 中取出该客户，并变成一维 Series。

    返回值
    ------
    list[str]
        风险信息组成的列表。每命中一条项目规则，就向列表中加入一段说明；
        如果一条规则都没有命中，则返回空列表。

    当前使用的五条演示规则
    ----------------------
    1. 合同类型是 Month-to-month（月付合同）；
    2. 没有开通技术支持，即 TechSupport 等于 No；
    3. 付款方式是 Electronic check（电子支票）；
    4. 使用时长不超过 12 个月；
    5. 月费不低于 75。

    注意
    ----
    这些规则主要来自项目 EDA 中观察到的相关关系。命中规则不代表该特征
    一定导致客户流失，因此页面会称其为“风险信息”而不是“流失原因”。
    性别和是否为老年客户等敏感信息不会用于生成差异化挽留策略。
    """
    risk_factors = []

    # 月付合同客户在本项目数据中的流失率相对较高，因此将其作为关注信息。
    if customer["Contract"] == "Month-to-month":
        risk_factors.append("合同类型为月付合同，该群体在项目数据中的流失率相对较高")

    # 这里只识别真正的“No”。“No internet service”表示没有互联网服务，
    # 其业务含义和“有互联网但没有技术支持”不同，不能混在一起判断。
    if customer["TechSupport"] == "No":
        risk_factors.append("客户已使用互联网服务，但未开通技术支持服务")

    # EDA 中使用电子支票的客户群体具有相对较高的流失率。
    if customer["PaymentMethod"] == "Electronic check":
        risk_factors.append("付款方式为电子支票，该群体在项目数据中的流失率相对较高")

    # tenure 表示客户已经使用服务的月数。较短的使用时长用于提示后续 AI
    # 关注新客户的使用体验，但不能据此断言客户一定会流失。
    if customer["tenure"] <= NEW_CUSTOMER_TENURE_MONTHS:
        risk_factors.append(
            f"客户使用时长为 {int(customer['tenure'])} 个月，仍处于项目定义的新客户阶段"
        )

    # 75 是项目演示边界，参考了流失客户平均月费约 74.44 的 EDA 结果。
    if customer["MonthlyCharges"] >= HIGH_MONTHLY_CHARGES:
        risk_factors.append(
            f"客户月费为 {customer['MonthlyCharges']:.2f}，达到项目定义的较高月费标准"
        )

    return risk_factors


def build_retention_prompt(probability, risk_level, risk_factors):
    """把客户预测结果整理成发送给 AI 的完整提示词。

    这个函数只负责准备文本，不会连接网络，也不会调用任何大模型 API。
    将“准备提示词”和“调用 AI”拆开，可以先在页面检查输入内容是否准确，
    也便于以后更换不同的大模型服务，而不必重写客户风险分析逻辑。

    参数
    ----
    probability : float
        机器学习模型计算出的流失概率，例如 0.6429 表示 64.29%。

    risk_level : str
        get_risk_level() 返回的风险等级，即“低风险”“中风险”或“高风险”。

    risk_factors : list[str]
        extract_risk_factors() 返回的风险信息列表。

    返回值
    ------
    str
        一段包含客户预测结果、风险信息、AI 输出任务和安全限制的完整文本。
        调用 DeepSeek 时，可以把这个返回值直接作为用户提示词发送。

    使用示例
    --------
    prompt = build_retention_prompt(
        probability=0.6429,
        risk_level="高风险",
        risk_factors=["合同类型为月付合同", "未开通技术支持服务"]
    )
    """
    # AI 接收的是一段文本，不能直接把 Python 列表原样当作自然语言使用。
    # "\n" 代表换行；join() 使用换行符连接列表中的每一项。
    # 例如 ["因素A", "因素B"] 会转换成：
    # - 因素A
    # - 因素B
    if risk_factors:
        risk_factors_text = "\n".join(
            f"- {factor}" for factor in risk_factors
        )
    else:
        risk_factors_text = "- 暂未命中当前项目的风险信息规则"

    # f-string 会把 {} 中的变量值填写到多行字符串中。
    # {probability:.2%} 会将 0 到 1 之间的小数转换为百分比，并保留两位小数。
    # 提示词同时规定 AI 的角色、输入事实、输出任务和限制条件，
    # 目的是降低 AI 虚构优惠政策或把统计相关误写成因果关系的可能性。
    prompt = f"""
你是一名电信客户运营辅助分析助手。

下面是一位客户的流失风险预测结果：

流失概率：{probability:.2%}
风险等级：{risk_level}

识别到的风险信息：
{risk_factors_text}

请根据以上信息生成客户挽留建议，输出内容包括：

1. 用一段简短文字说明该客户目前的风险情况。
2. 提供最多三条具有针对性的挽留建议。
3. 提供一段客服可以参考的沟通话术。
4. 给出建议的跟进优先级。

生成要求：

- 使用简洁、专业的中文。
- 只能把上述特征描述为相关风险信息，不能断言它们一定导致客户流失。
- 不得虚构具体优惠金额、企业政策或客户没有提供的信息。
- 如建议提供优惠，必须注明需要符合企业实际政策。
- 不使用性别、年龄等敏感信息制定差异化挽留策略。
- 不得保证客户接受建议后一定不会流失。
"""

    # 三引号字符串为了排版会在开头和结尾产生空行。
    # strip() 删除这些多余空白，让最终发送给 AI 的文本更加整洁。
    return prompt.strip()


def get_deepseek_api_key():
    """从安全配置中读取 DeepSeek API Key。

    读取顺序
    --------
    1. 优先读取操作系统环境变量 DEEPSEEK_API_KEY；
    2. 如果环境变量不存在，再读取 Streamlit 的 st.secrets；
    3. 两处都没有配置时返回 None，由页面显示配置提示。

    本地密钥保存在被 Git 忽略的
    .streamlit/secrets.toml 中。

    返回值
    ------
    str | None
        找到有效配置时返回去除首尾空格后的 Key，否则返回 None。
    """
    # os.getenv() 用于读取操作系统环境变量。第二个参数 "" 是找不到时的默认值。
    environment_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if environment_key:
        return environment_key

    try:
        # st.secrets 的使用方式与字典类似。get() 在字段不存在时返回空字符串，
        # 不会因为某个字段漏填而直接触发 KeyError。
        secrets_key = str(st.secrets.get("DEEPSEEK_API_KEY", "")).strip()
    except StreamlitSecretNotFoundError:
        # 项目还没有 secrets.toml 时，Streamlit 会抛出该异常。
        # 配置缺失属于可恢复情况，所以返回 None，让页面告诉用户如何处理。
        return None

    return secrets_key or None


def generate_ai_retention_advice(prompt, api_key):
    """调用 DeepSeek API，生成一位客户的智能挽留建议。

    参数
    ----
    prompt : str
        build_retention_prompt() 创建的完整提示词。

    api_key : str
        从环境变量或 Streamlit Secrets 安全读取的 DeepSeek API Key。

    返回值
    ------
    str
        DeepSeek 返回的最终自然语言建议。

    异常
    ----
    网络不可用、Key 无效、余额不足、请求过于频繁或服务异常时，OpenAI SDK
    会抛出异常。本函数不隐藏异常，而是交给页面中的 try/except 统一显示
    友好的失败提示。
    """
    # OpenAI() 在这里是一个“兼容客户端”，实际请求发送到 DEEPSEEK_BASE_URL，
    # 并不是在调用 OpenAI 模型。timeout 防止网络异常时页面无限等待；
    # max_retries=1 表示遇到临时问题最多自动重试一次，避免重复消耗过多时间。
    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=30.0,
        max_retries=1
    )

    # messages 是对话消息列表。system 规定模型的总体职责，user 放入该客户
    # 的具体预测信息和生成要求。stream=False 表示等待整段回答后一次性返回。
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是谨慎、专业的电信客户运营辅助分析助手。"
                    "请遵守用户提示词中的事实边界和安全限制。"
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=False
    )

    # choices[0] 表示取模型返回的第一个候选答案；message.content 是正文。
    advice = response.choices[0].message.content
    if not advice:
        raise ValueError("DeepSeek 返回了空内容")

    return advice.strip()


def get_ai_error_message(error):
    """把 OpenAI 兼容 SDK 的异常转换成适合页面展示的中文说明。

    DeepSeek 使用 OpenAI 兼容接口，因此不同 HTTP 或网络问题会表现为
    OpenAI SDK 中的异常类型。与只显示“调用失败”相比，分类提示能帮助用户
    快速判断应该检查密钥、余额、网络、请求频率还是模型配置。

    参数
    ----
    error : Exception
        调用 generate_ai_retention_advice() 时捕获到的异常对象。

    返回值
    ------
    str
        不包含 API Key 和完整请求内容的安全中文提示。
    """
    # APITimeoutError 属于连接类问题，应在 APIConnectionError 之前判断，
    # 否则超时可能被提前归入普通连接失败。
    if isinstance(error, APITimeoutError):
        return "DeepSeek 响应超时，请稍后重试或检查当前网络状态。"
    if isinstance(error, APIConnectionError):
        return "无法连接 DeepSeek，请检查网络、代理和本地依赖后重试。"
    if isinstance(error, AuthenticationError):
        return "DeepSeek API Key 无效或已被吊销，请重新检查本地密钥配置。"
    if isinstance(error, PermissionDeniedError):
        return "DeepSeek 拒绝了请求，请检查账户余额以及模型访问权限。"
    if isinstance(error, RateLimitError):
        return "DeepSeek 请求过于频繁，请稍等片刻后再试。"
    if isinstance(error, NotFoundError):
        return "当前 DeepSeek 模型不存在或不可用，请检查模型名称。"
    if isinstance(error, BadRequestError):
        return "发送给 DeepSeek 的请求参数不符合要求，请检查提示词和模型配置。"
    if isinstance(error, InternalServerError):
        return "DeepSeek 服务暂时异常，请稍后重试。"

    return "AI 建议生成失败，请稍后重试并查看技术错误类型。"


# ---------- 4. 计算页面需要展示的 EDA 指标 ----------
# 这些指标来自第一阶段 EDA；应用这里只负责展示，不重新训练模型。
customer_count = len(df)
churn_rate = df["ChurnFlag"].mean()
avg_monthly_charges = df["MonthlyCharges"].mean()
avg_tenure = df["tenure"].mean()

# ---------- 5. 项目概览与 EDA 分析结果 ----------
# 概览指标放在页面前面，让用户先了解数据总体情况。
st.subheader("核心数据指标")
st.metric("客户总数", f"{customer_count:,}")
st.metric("整体流失率", f"{churn_rate:.2%}")
st.metric("平均月费", f"{avg_monthly_charges:.2f}")
st.metric("平均使用时长（月）", f"{avg_tenure:.2f}")

# 下面的图表用于观察不同客户群体之间的流失率或平均值差异。
# 按合同类型计算流失率
# ---------- 5.1 EDA 图表 ----------
# groupby 按类别分组，mean 计算每组 ChurnFlag 的平均值，也就是该组流失率。
contract_churn_rate = (
    df.groupby("Contract")["ChurnFlag"].mean() * 100
)

st.subheader("合同类型与客户流失率")
st.bar_chart(contract_churn_rate)

# 技术支持服务与流失率的关系。
tech_support_churn_rate = (
    df.groupby("TechSupport")["ChurnFlag"].mean() * 100
)

st.subheader("技术支持服务与客户流失率")
st.bar_chart(tech_support_churn_rate)

# 比较流失与未流失客户的平均月费。
monthly_charges_by_churn = (
    df.groupby("Churn")["MonthlyCharges"].mean()
)

st.subheader("流失状态与平均月费")
st.bar_chart(monthly_charges_by_churn)

# 比较流失与未流失客户的平均使用时长。
tenure_by_churn = (
    df.groupby("Churn")["tenure"].mean()
)

st.subheader("流失状态与平均使用时长")
st.bar_chart(tenure_by_churn)

# 原始表格用于辅助查看字段，默认折叠，避免页面过长。
with st.expander("查看原始客户数据"):
    st.dataframe(df.head(10))

# ---------- 6. 在测试集上评价已加载的 Pipeline ----------
# 重新进行同样的分层划分，只用于复现 Notebook 中的测试集评价。
# stratify=y 保证训练集和测试集的流失比例大致一致。
st.info("以下指标来自固定测试集，用于了解当前逻辑回归基线模型的表现。")
X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# predict_proba(X) 的作用：输出每个样本属于各个类别的概率。
# 对二分类模型来说，返回结果的形状通常为 (样本数, 2)：
# - 第 0 列是“未流失（类别 0）”的概率；
# - 第 1 列是“流失（类别 1）”的概率。
# 因此 [:, 1] 表示取所有客户的第 1 列，也就是流失概率。
y_eval_proba = loaded_pipeline.predict_proba(X_test_eval)[:, 1]

# 使用项目选定的 0.40 阈值把概率转换为预测类别。
# 比较表达式会先得到 True/False，再通过 astype(int) 转换为 1/0：
# - 概率 >= 0.40 -> True  -> 1（预测流失）；
# - 概率 <  0.40 -> False -> 0（预测未流失）。
# 这里不使用 loaded_pipeline.predict()，因为 predict() 对逻辑回归默认采用约 0.50 阈值，
# 会导致页面指标与 Notebook 已选择的 0.40 阈值口径不一致。
y_eval_pred = (
    y_eval_proba >= CLASSIFICATION_THRESHOLD
).astype(int)

# 开发阶段用于核对数据规模的信息保留在折叠区域中。
# 普通用户默认看不到这些细节，页面会比直接逐行输出更加简洁。
with st.expander("查看模型调试信息", expanded=False):
    st.write("模型输入数据形状：", X.shape)
    st.write("标签数量：", y.shape)
    st.write("评价测试集形状：", X_test_eval.shape)
    st.write("评价标签形状：", y_test_eval.shape)
    st.write("当前分类阈值：", f"{CLASSIFICATION_THRESHOLD:.2f}")
    st.write("预测类别数量：", len(y_eval_pred))
    st.write("预测概率数量：", len(y_eval_proba))

# sklearn.metrics 中的这些函数都遵循相同的基本用法：
# 指标函数(真实标签, 预测结果)。
# y_test_eval 是测试集的真实标签，y_eval_pred 是阈值转换后的预测类别。

# accuracy_score：准确率，表示全部测试客户中预测正确的比例。
accuracy = accuracy_score(y_test_eval, y_eval_pred)

# precision_score：精确率，表示“被预测为流失”的客户中，实际流失的比例。
# 精确率较低意味着运营人员可能会联系较多实际不会流失的客户。
precision = precision_score(y_test_eval, y_eval_pred)

# recall_score：召回率，表示所有“实际流失”客户中，被模型成功识别的比例。
# 客户挽留场景通常比较关注 Recall，因为漏掉真实流失客户会失去挽留机会。
recall = recall_score(y_test_eval, y_eval_pred)

# f1_score：Precision 和 Recall 的综合指标。
# 当我们不希望只关注其中一个指标时，可以使用 F1 进行平衡比较。
f1 = f1_score(y_test_eval, y_eval_pred)

# roc_auc_score：使用真实标签和连续概率计算 ROC-AUC。
# ROC-AUC 衡量模型整体区分流失/未流失客户的排序能力，不依赖某一个固定分类阈值。
roc_auc = roc_auc_score(y_test_eval, y_eval_proba)

st.subheader(
    f"模型评价指标（分类阈值 = {CLASSIFICATION_THRESHOLD:.2f}）"
)

st.metric("Accuracy", f"{accuracy:.3f}")
st.metric("Precision", f"{precision:.3f}")
st.metric("Recall", f"{recall:.3f}")
st.metric("F1", f"{f1:.3f}")
st.metric("ROC-AUC", f"{roc_auc:.3f}")


# confusion_matrix(真实标签, 预测类别) 用二维表展示实际类别与预测类别的对应关系。
# sklearn 二分类混淆矩阵的默认排列为：
# [[TN, FP],
#  [FN, TP]]
# TN：实际未流失且预测未流失；FP：实际未流失但预测流失；
# FN：实际流失但预测未流失；TP：实际流失且预测流失。
cm = confusion_matrix(y_test_eval, y_eval_pred)

cm_display = pd.DataFrame(
    cm,
    index=["实际未流失", "实际流失"],
    columns=["预测未流失", "预测流失"]
)

st.subheader(
    f"混淆矩阵（分类阈值 = {CLASSIFICATION_THRESHOLD:.2f}）"
)
st.dataframe(cm_display)

st.subheader("模型局限")

st.info(
    "当前模型是逻辑回归基线模型，评价结果受数据划分、特征选择和分类阈值影响。"
    "模型输出的是风险概率，不代表客户一定会流失；实际业务还需要结合联系成本、客户价值和运营策略。"
)


# ---------- 7. 单个客户预测表单 ----------
# 所有输入字段的名称和取值必须与训练 Pipeline 使用的字段保持一致。
st.subheader("单客户流失预测")
st.info("请填写客户信息，点击‘预测新客户’后查看流失概率和风险等级。")

st.markdown("#### 合同与费用信息")
contract = st.selectbox(
    "合同类型",
    ["Month-to-month", "One year", "Two year"]
)

input_tenure = st.number_input(
    "客户使用时长（月）",
    min_value=0,
    max_value=100,
    value=6
)

input_monthly_charges = st.number_input(
    "客户月费",
    min_value=0.0,
    value=85.0
)

input_total_charges = st.number_input(
    "客户累计费用",
    min_value=0.0,
    value=510.0
)

# 付款方式与其他服务字段没有依赖关系，可以单独输入。
st.markdown("#### 付款信息")
payment_method_input = st.selectbox(
    "付款方式",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

st.markdown("#### 客户基本信息")
gender_input = st.selectbox(
    "性别",
    ["Female", "Male"]
)

partner_input = st.selectbox(
    "是否有伴侣",
    ["Yes", "No"]
)

dependents_input = st.selectbox(
    "是否有家属",
    ["Yes", "No"]
)

st.markdown("#### 电话、网络和附加服务")
phone_service_input = st.selectbox(
    "是否开通电话服务",
    ["Yes", "No"]
)

# 多线路依赖电话服务。没有电话服务时，训练数据中的对应取值必须是
# "No phone service"，不能再让用户选择 Yes 或 No。
if phone_service_input == "No":
    multiple_lines_input = st.selectbox(
        "是否使用多条线路",
        ["No phone service"],
        disabled=True,
        help="未开通电话服务时，多线路状态会自动设为 No phone service。"
    )
else:
    multiple_lines_input = st.selectbox(
        "是否使用多条线路",
        ["Yes", "No"]
    )

internet_service_input = st.selectbox(
    "互联网服务类型",
    ["DSL", "Fiber optic", "No"]
)

# 在线安全、备份、设备保护、技术支持和流媒体服务都依赖互联网服务。
# 如果互联网服务为 No，就统一锁定为训练数据使用的 "No internet service"；
# 这样可以避免“没有互联网但开通在线服务”等不可能的输入组合。
if internet_service_input == "No":
    st.info("未开通互联网服务，相关附加服务已自动设为 No internet service。")

    online_security_input = st.selectbox(
        "在线安全服务", ["No internet service"], disabled=True
    )
    online_backup_input = st.selectbox(
        "在线备份服务", ["No internet service"], disabled=True
    )
    device_protection_input = st.selectbox(
        "设备保护服务", ["No internet service"], disabled=True
    )
    tech_support_input = st.selectbox(
        "技术支持服务", ["No internet service"], disabled=True
    )
    streaming_tv_input = st.selectbox(
        "流媒体电视", ["No internet service"], disabled=True
    )
    streaming_movies_input = st.selectbox(
        "流媒体电影", ["No internet service"], disabled=True
    )
else:
    online_security_input = st.selectbox(
        "在线安全服务", ["Yes", "No"]
    )
    online_backup_input = st.selectbox(
        "在线备份服务", ["Yes", "No"]
    )
    device_protection_input = st.selectbox(
        "设备保护服务", ["Yes", "No"]
    )
    tech_support_input = st.selectbox(
        "技术支持服务", ["No", "Yes"]
    )
    streaming_tv_input = st.selectbox(
        "流媒体电视", ["Yes", "No"]
    )
    streaming_movies_input = st.selectbox(
        "流媒体电影", ["Yes", "No"]
    )

paperless_billing_input = st.selectbox(
    "是否使用无纸化账单",
    ["Yes", "No"]
)

senior_citizen_input = st.selectbox(
    "是否为老年客户",
    ["No", "Yes"]
)

# 将页面控件的结果组装成一行 DataFrame。
# Pipeline 接收的就是这种“列名与训练特征一致”的表格。
new_customer = pd.DataFrame([{
    "SeniorCitizen": 1 if senior_citizen_input == "Yes" else 0,
    "tenure": input_tenure,
    "MonthlyCharges": input_monthly_charges,
    "TotalCharges": input_total_charges,
    "gender": gender_input,
    "Partner": partner_input,
    "Dependents": dependents_input,
    "PhoneService": phone_service_input,
    "MultipleLines": multiple_lines_input,
    "InternetService": internet_service_input,
    "OnlineSecurity": online_security_input,
    "OnlineBackup": online_backup_input,
    "DeviceProtection": device_protection_input,
    "TechSupport": tech_support_input,
    "StreamingTV": streaming_tv_input,
    "StreamingMovies": streaming_movies_input,
    "Contract": contract,
    "PaperlessBilling": paperless_billing_input,
    "PaymentMethod": payment_method_input
}])

# 只有点击预测按钮后才执行预测，避免调整控件时反复显示结果。
# 点击预测后把结果保存到 session_state。
# Streamlit 每次点击按钮都会从头重新执行 app.py；如果只使用普通局部变量，
# 再点击“生成 AI 挽留建议”时，上一次预测结果就会消失。
if st.button("预测新客户"):
    new_customer_probability = loaded_pipeline.predict_proba(
        new_customer
    )[0, 1]

    # 调用统一的风险等级函数。以后如果调整阈值，只需修改文件顶部的常量，
    # 不需要分别修改单客户预测和批量预测，避免两个功能使用不同标准。
    risk_level = get_risk_level(new_customer_probability)

    # DataFrame 的 iloc[0] 表示“按位置取第 1 行”。这里将只有一行的
    # new_customer 转换为 Series，再交给风险信息提取函数逐项判断。
    risk_factors = extract_risk_factors(new_customer.iloc[0])

    # 将模型概率、风险等级和风险信息组合成一段完整的 AI 提示词。
    # 这里使用“参数名=变量”的形式，能够直观看出每个实参传给哪个形参。
    retention_prompt = build_retention_prompt(
        probability=new_customer_probability,
        risk_level=risk_level,
        risk_factors=risk_factors
    )

    # 使用一个字典统一保存本次预测的相关内容，方便页面重新运行后继续读取。
    st.session_state["single_customer_result"] = {
        "probability": new_customer_probability,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "retention_prompt": retention_prompt
    }

    # 客户资料改变并重新预测后，旧的 AI 建议已经不再对应新客户，应当清除。
    st.session_state.pop("ai_retention_advice", None)


# 只有完成过一次预测后才展示结果和 AI 按钮。
# 因为数据保存在 session_state 中，点击第二个按钮后这些结果仍然存在。
if "single_customer_result" in st.session_state:
    prediction_result = st.session_state["single_customer_result"]
    new_customer_probability = prediction_result["probability"]
    risk_level = prediction_result["risk_level"]
    risk_factors = prediction_result["risk_factors"]
    retention_prompt = prediction_result["retention_prompt"]

    # 输出概率而不是只输出 0/1，运营人员可以据此排序和分级。
    st.metric(
        "流失概率",
        f"{new_customer_probability:.2%}"
    )

    st.write("风险等级：", risk_level)

    st.caption(
        "风险等级划分为项目演示规则，并非真实企业业务标准。"
    )

    # 这一部分用于预览即将发送给 DeepSeek 的输入信息。
    # 只有点击后面的生成按钮时才会产生 API 请求和费用。
    st.markdown("#### AI 挽留建议：客户信息准备")
    st.write("以下信息将作为本次 DeepSeek 请求的分析依据：")
    st.write(f"- 流失概率：{new_customer_probability:.2%}")
    st.write(f"- 风险等级：{risk_level}")

    if risk_factors:
        st.write("- 识别到的风险信息：")
        for factor in risk_factors:
            st.write(f"  - {factor}")
    else:
        st.write("- 识别到的风险信息：暂未命中当前项目的五条演示规则")

    st.caption(
        "以上内容来自模型结果和项目 EDA 规则，仅表示统计关联，"
        "不代表这些特征一定导致客户流失。点击下方按钮后才会调用 AI。"
    )

    # expander 创建一个可以展开和收起的页面区域。
    # expanded=False 表示默认收起，避免较长的提示词占用过多页面空间。
    # st.code 使用纯文本格式展示 Prompt，方便在正式调用 AI 前人工检查。
    with st.expander("查看将发送给 AI 的完整提示词", expanded=False):
        st.code(retention_prompt, language="text")

    # ---------- 7.1 调用 DeepSeek 生成智能挽留建议 ----------
    st.markdown("#### AI 智能挽留建议")
    st.caption(
        "只有点击生成按钮后才会调用 DeepSeek API；每次调用都可能产生少量费用。"
    )

    deepseek_api_key = get_deepseek_api_key()

    if deepseek_api_key is None:
        # 没有密钥时不发送请求，并明确告诉用户应在哪个文件中配置。
        st.warning(
            "尚未检测到 DeepSeek API Key，请将有效 Key 填入 "
            ".streamlit/secrets.toml。如果 Key 曾经公开，请先吊销并重新生成。"
        )
    else:
        # 按钮必须位于预测结果的持久化区域内，才能在 Streamlit 重新运行时
        # 继续读取 retention_prompt。spinner 用于提示网络请求正在处理中。
        if st.button("生成 AI 挽留建议"):
            with st.spinner("DeepSeek 正在生成挽留建议，请稍候……"):
                try:
                    ai_advice = generate_ai_retention_advice(
                        prompt=retention_prompt,
                        api_key=deepseek_api_key
                    )
                except Exception as error:
                    # 不在页面打印 API Key，也不展示可能包含请求细节的完整异常。
                    # 错误类型足以帮助判断是认证、网络、限流还是其他问题。
                    st.session_state.pop("ai_retention_advice", None)
                    st.error(get_ai_error_message(error))

                    # 只展示异常类型和最底层原因的类型，不打印可能包含请求内容的
                    # 完整异常字符串，也不会在页面中暴露 API Key。
                    cause = error.__cause__
                    cause_name = type(cause).__name__ if cause else "无"
                    st.caption(
                        f"技术信息：{type(error).__name__}；底层原因：{cause_name}"
                    )
                else:
                    # 保存回答，保证页面发生下一次重新运行时结果不会立即消失。
                    st.session_state["ai_retention_advice"] = ai_advice

    # API 调用成功后显示模型回答。st.markdown 可以正常渲染标题和列表。
    if "ai_retention_advice" in st.session_state:
        st.success("AI 挽留建议生成成功")
        st.markdown(st.session_state["ai_retention_advice"])
        st.caption(
            f"生成模型：{DEEPSEEK_MODEL}。AI 内容仅供项目演示和人工参考。"
        )

# ---------- 8. 批量预测并筛选高风险客户 ----------
# predict_proba 返回每位客户属于“流失=1”的概率；[:, 1] 取流失概率这一列。
st.subheader("现有客户批量风险筛选（项目数据演示）")
st.write(
    "本模块对项目原始数据中的全部客户计算流失概率并筛选高风险客户。"
    "真实使用时应改为上传一份当前、没有 Churn 标签的新客户名单。"
)
all_customer_probabilities = loaded_pipeline.predict_proba(X)[:, 1]

# 复制客户数据，避免修改原始 df_clean
all_customer_predictions = df_clean.copy()

# 添加流失概率列
all_customer_predictions["churn_probability"] = all_customer_probabilities

# 将概率转换为演示用风险等级。
# 这里复用文件顶部的 get_risk_level，不再重复定义函数或写死阈值。
all_customer_predictions["risk_level"] = (
    all_customer_predictions["churn_probability"]
    .apply(get_risk_level)
)

top_n = st.selectbox(
    "显示高风险客户数量",
    [10, 20]
)


# 先筛选“高风险”，再按流失概率降序排列并取 Top N。
# 这样可以保证页面标题和表格内容一致，不会把中风险客户混入高风险名单。
high_risk_customers = all_customer_predictions[
    all_customer_predictions["risk_level"] == "高风险"
]

top_customers = (
    high_risk_customers[
        ["customerID", "churn_probability", "risk_level", "Contract", "MonthlyCharges"]
    ]
    .sort_values("churn_probability", ascending=False)
    .head(top_n)
)

st.subheader(f"高风险客户 Top {top_n}")
st.dataframe(top_customers)


# 将当前 Top N 表格转换为 CSV 文本，供浏览器下载。
csv_data = top_customers.to_csv(
    index=False,
    encoding="utf-8-sig"
)

# 提供下载按钮
st.download_button(
    label="下载高风险客户 CSV",
    data=csv_data,
    file_name=f"high_risk_customers_top_{top_n}.csv",
    mime="text/csv"
)
