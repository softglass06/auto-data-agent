# -*- coding: utf-8 -*-
"""
=============================================================================
Project: Auto Data Agent (自动数据分析智能体)
Tech Stack: Streamlit + Pandas + Plotly + LLM API (OpenAI-compatible)
License: MIT License
Description:
    An intelligent end-to-end data analytics and visualization agent.
    - Automated Exploratory Data Analysis (Auto-EDA) with Plotly heatmaps
    - Natural language to Pandas & Plotly code generation (NL2Code)
    - Safe execution environment (restricted exec sandbox)
    - Dual-column interactive dashboard with structured insight generation
    - Built-in benchmark datasets (Titanic, E-commerce) & Offline Demo Mode
=============================================================================
"""

import os
import sys
import re
import json
import traceback
from typing import Dict, Any, Tuple, Optional
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration & UI Styles
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Auto Data Agent - 自动数据分析智能体",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Data Science Dashboard
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid #DBEAFE;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #1E40AF;
    }
    .insight-card {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #DBEAFE;
        color: #1E40AF;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = ""
if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# -----------------------------------------------------------------------------
# 3. LLM API Client (OpenAI-compatible HTTP Gateway)
# -----------------------------------------------------------------------------
def call_llm_api(
    prompt: str,
    system_prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float = 0.2
) -> str:
    """
    Call OpenAI-compatible Chat Completions endpoint via HTTP POST.
    Compatible with DeepSeek, OpenAI, SiliconFlow, Qwen, Moonshot, Ollama, vLLM, etc.
    """
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        if not url.endswith("/v1"):
            url = f"{url}/v1/chat/completions"
        else:
            url = f"{url}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}"
    }

    payload = {
        "model": model.strip(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 2048
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(
            f"API Request Failed [HTTP {response.status_code}]: {response.text}"
        )
    data = response.json()
    return data["choices"][0]["message"]["content"]


# -----------------------------------------------------------------------------
# 4. Code Extraction & Sanitization
# -----------------------------------------------------------------------------
def extract_python_code(raw_response: str) -> str:
    """
    Extract pure Python code inside ```python ... ``` markdown block.
    """
    code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", raw_response, re.DOTALL | re.IGNORECASE)
    if code_match:
        code = code_match.group(1).strip()
    else:
        code = raw_response.strip()

    # Block dangerous system-level function calls
    forbidden_terms = ["sys.exit", "os.system", "shutil.rmtree", "subprocess.", "__import__"]
    for term in forbidden_terms:
        if term in code:
            raise ValueError(f"Dangerous call detected in generated code: {term}")

    return code


# -----------------------------------------------------------------------------
# 5. Safe Code Execution Sandbox
# -----------------------------------------------------------------------------
def execute_generated_code(
    code_str: str,
    df: pd.DataFrame
) -> Tuple[bool, Optional[go.Figure], Any, str]:
    """
    Execute generated Python code in a restricted scope.
    The code must store the Plotly Figure in `fig` and key metrics in `summary_data`.
    """
    safe_builtins = {
        "len": len, "range": range, "str": str, "int": int, "float": float,
        "bool": bool, "list": list, "dict": dict, "set": set, "tuple": tuple,
        "sum": sum, "min": min, "max": max, "round": round, "abs": abs,
        "sorted": sorted, "enumerate": enumerate, "zip": zip, "print": print,
        "isinstance": isinstance, "type": type
    }

    exec_globals = {"__builtins__": safe_builtins}
    exec_locals = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "fig": None,
        "summary_data": None
    }

    try:
        exec(code_str, exec_globals, exec_locals)

        fig = exec_locals.get("fig")
        summary_data = exec_locals.get("summary_data")

        if fig is None:
            return False, None, summary_data, "执行完成但未找到变量 `fig`，请确认代码中已生成 Plotly 图表对象。"

        return True, fig, summary_data, "执行成功"

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return False, None, None, error_msg


# -----------------------------------------------------------------------------
# 6. Fallback / Offline Rule-Based Demo Engine
# -----------------------------------------------------------------------------
def generate_mock_analysis(
    user_query: str,
    dataset_type: str,
    df: pd.DataFrame
) -> Tuple[str, str]:
    """
    Rule-based intelligent template engine for offline demonstration
    without requiring external API keys.
    """
    q = user_query.lower()

    if dataset_type == "titanic":
        if "性别" in q or "sex" in q:
            code = """# 统计不同性别的幸存率与总人数
summary = df.groupby('Sex').agg(
    总人数=('PassengerId', 'count'),
    幸存人数=('Survived', 'sum'),
    幸存率=('Survived', 'mean')
).reset_index()
summary['幸存率(%)'] = (summary['幸存率'] * 100).round(2)
summary_data = summary.to_dict(orient='records')

# 绘制 Plotly 交互柱状图
fig = px.bar(
    summary,
    x='Sex',
    y='幸存率(%)',
    color='Sex',
    text='幸存率(%)',
    title='泰坦尼克号乘客不同性别的生还率对比',
    labels={'Sex': '性别', '幸存率(%)': '生还率 (%)'},
    color_discrete_map={'female': '#EC4899', 'male': '#3B82F6'},
    template='plotly_white'
)
fig.update_traces(textposition='outside')
fig.update_layout(yaxis_range=[0, 100])
"""
            insight = """### 📌 核心结论与数据事实
- **女性生存优势极其显著**：女性生还率约为 74.2%，而男性生还率仅为 18.9%，女性生还概率约为男性的 4 倍。
- **样本基数**：男性乘客总数远超女性（占比约 60% 以上），但幸存绝对人数反而显著低于女性。

### 🔍 深度业务/数据机理解析
- 这一数据强有力地印证了海难撤离时严格执行的“女士和儿童优先 (Women and children first)”救援准则。
- 救生艇名额分配上，女性和母亲获得了优先登艇的权利，社会契约与伦理规范在极端危机下发挥了主导作用。

### 💡 针对性决策与行动建议
1. **应急演练规范化**：在极端应急疏散规约制定中，清晰的分流规则对于减少拥挤踩踏、提高弱势群体存活率具有决定性价值。
2. **多维交叉探索**：建议进一步结合“舱位等级 (Pclass)”进行双重交叉分析（如 `Sex + Pclass`），探索高舱位男性与低舱位女性的生还差异。
"""
            return code, insight

        elif "舱位" in q or "pclass" in q or "票价" in q or "fare" in q:
            code = """# 分析不同船舱等级的平均票价与生还率
summary = df.groupby('Pclass').agg(
    总乘客数=('PassengerId', 'count'),
    平均票价=('Fare', 'mean'),
    生还率=('Survived', 'mean')
).reset_index()
summary['平均票价'] = summary['平均票价'].round(2)
summary['生还率(%)'] = (summary['生还率'] * 100).round(2)
summary['Pclass_Label'] = summary['Pclass'].map({1: '头等舱(1st)', 2: '二等舱(2nd)', 3: '三等舱(3rd)'})
summary_data = summary.to_dict(orient='records')

# 绘制双维度柱状图
fig = px.bar(
    summary,
    x='Pclass_Label',
    y='生还率(%)',
    color='平均票价',
    text='生还率(%)',
    title='不同船舱等级 (Pclass) 的生还率与票价分布',
    labels={'Pclass_Label': '船舱等级', '生还率(%)': '生还率 (%)', '平均票价': '平均票价 (英镑)'},
    color_continuous_scale='Tealgrn',
    template='plotly_white'
)
fig.update_traces(textposition='outside')
fig.update_layout(yaxis_range=[0, 100])
"""
            insight = """### 📌 核心结论与数据事实
- **阶层与生还率呈现极强正相关**：头等舱 (Pclass 1) 生还率高达 62% 以上；二等舱约为 47%；而三等舱 (Pclass 3) 生还率仅约 24%。
- **经济成本差异巨大**：头等舱平均票价是三等舱的近 7 倍，反映出极大的经济阶层分化。

### 🔍 深度业务/数据机理解析
- **地理空间与舱位布局**：头等舱通常位于游轮上层甲板，距离救生艇甲板最近；而三等舱位于游轮底层，撤离通道曲折且部分闸门在事故初期处于关闭状态。
- **信息通达性**：高等级船舱乘客第一时间得到了船员通知与疏导服务，信息获取的滞后性严重削弱了底层乘客的逃生窗口期。

### 💡 针对性决策与行动建议
1. **通道平等设计**：现代大型邮轮必须建立无障碍、无特权的均等化应急撤离逃生廊道。
2. **物联网感知预警**：对底层舱室部署智能应急照明和定向广播指引系统。
"""
            return code, insight

        else:
            code = """# 探索年龄分布与票价散点关系
plot_df = df.dropna(subset=['Age', 'Fare']).copy()
plot_df['生存状态'] = plot_df['Survived'].map({1: '幸存', 0: '遇难'})
summary_data = {
    '平均年龄': round(plot_df['Age'].mean(), 1),
    '年龄中位数': round(plot_df['Age'].median(), 1),
    '平均票价': round(plot_df['Fare'].mean(), 2)
}

fig = px.scatter(
    plot_df,
    x='Age',
    y='Fare',
    color='生存状态',
    marginal_x='histogram',
    marginal_y='box',
    title='乘客年龄与票价分布关系 (按生存状态分组)',
    labels={'Age': '年龄 (岁)', 'Fare': '票价 (英镑)', '生存状态': '生存结果'},
    color_discrete_map={'幸存': '#10B981', '遇难': '#EF4444'},
    template='plotly_white'
)
"""
            insight = """### 📌 核心结论与数据事实
- 乘客年龄主要集中在 18-40 岁之间；高票价（>100 英镑）乘客中，幸存绿点比例明显高于低票价区域。
- 5 岁以下的幼童群体幸存比例显著偏高，体现了对弱小儿童的优先救助原则。

### 🔍 深度业务/数据机理解析
- 票价作为经济地位与舱位等级的直接代理变量，强有力地影响了生死结果。
- 年龄两端（极低龄与极高龄）中，低龄由于受到家庭监护人与群体的倾斜性保护，获得了极高的存活窗口。

### 💡 针对性决策与行动建议
- 建立基于年龄特征的分类保障机制，重点关注老人与儿童等行动受限群体的专项疏散方案。
"""
            return code, insight

    else:
        # E-commerce Dataset Analysis
        if "品类" in q or "类别" in q or "category" in q or "销售额" in q:
            code = """# 各大类销售额与利润对比
summary = df.groupby('商品大类').agg(
    总销售额=('销售额', 'sum'),
    总利润=('净利润', 'sum'),
    平均折扣率=('折扣率', 'mean')
).reset_index()
summary['销售额(万元)'] = (summary['总销售额'] / 10000).round(2)
summary['净利润(万元)'] = (summary['总利润'] / 10000).round(2)
summary['利润率(%)'] = ((summary['总利润'] / summary['总销售额']) * 100).round(2)
summary_data = summary.to_dict(orient='records')

# 绘制柱状图
fig = px.bar(
    summary,
    x='商品大类',
    y=['销售额(万元)', '净利润(万元)'],
    barmode='group',
    title='各商品大类的销售额与净利润规模对比',
    labels={'value': '金额 (万元)', 'variable': '指标类别', '商品大类': '品类'},
    color_discrete_map={'销售额(万元)': '#3B82F6', '净利润(万元)': '#10B981'},
    template='plotly_white'
)
"""
            insight = """### 📌 核心结论与数据事实
- **电子数码**品类的总销售额遥遥领先，贡献了超过 40% 的总体营收，是驱动大盘的核心支柱。
- **服饰鞋包**与**家居生活**展现出更高的利润率弹性，虽然客单价适中，但整体净利润表现健康。

### 🔍 深度业务/数据机理解析
- 电子数码具有高客单价、高流转特征，但供应链成本与促销折让较大，导致其综合销售利润率略低于高毛利品类。
- 家居生活类产品复购周期虽长，但单笔附加值和毛利稳定，属于典型的“利润压舱石”业务。

### 💡 针对性决策与行动建议
1. **品类组合策略**：以电子数码作为前台引流型“爆品”，配套捆绑高毛利的数码配件与生活文教产品，拉升全单综合净利润率。
2. **折扣刚性管控**：避免在数码大件上进行过深幅度的无差别满减，转向会员权益返券等温和促销方式。
"""
            return code, insight

        elif "区域" in q or "region" in q:
            code = """# 各区域销售额分布与对比
summary = df.groupby('区域').agg(
    订单总数=('订单编号', 'count'),
    总销售额=('销售额', 'sum'),
    平均客单价=('销售额', 'mean'),
    平均利润=('净利润', 'mean')
).reset_index()
summary['总销售额(万元)'] = (summary['总销售额'] / 10000).round(2)
summary['平均客单价'] = summary['平均客单价'].round(2)
summary_data = summary.to_dict(orient='records')

fig = px.pie(
    summary,
    names='区域',
    values='总销售额(万元)',
    title='全国各销售区域营收贡献占比',
    hole=0.4,
    color_discrete_sequence=px.colors.qualitative.Set2,
    template='plotly_white'
)
fig.update_traces(textposition='inside', textinfo='percent+label')
"""
            insight = """### 📌 核心结论与数据事实
- **华东与华南**两大经济发达区域贡献了大盘近 60% 的总销售额，属于核心粮仓市场。
- **西南与华中**市场虽然当前销售占比较低（合计约 20%），但订单增速和客单价稳步上升，具备高成长潜力。

### 🔍 深度业务/数据机理解析
- 沿海区域的数字化消费渗透率高，物流履约时效快（次日达/隔日达），消费者的线上信任度高。
- 中西部市场目前受制于区域前置仓密度和配送半径，长尾商品的需求尚未被充分激活。

### 💡 针对性决策与行动建议
1. **仓网布局升级**：在华中与西南建设枢纽前置仓，提升区域履约速度。
2. **本地化选品**：针对中西部消费偏好，制定适配二三线城市家庭的性价比产品矩阵。
"""
            return code, insight

        else:
            code = """# 客户类型与销售额箱线图
summary = df.groupby('客户类型')['销售额'].describe().round(2).reset_index()
summary_data = summary.to_dict(orient='records')

fig = px.box(
    df,
    x='客户类型',
    y='销售额',
    color='客户类型',
    points='outliers',
    title='不同客户类型的单笔订单销售额分布 (箱线图)',
    labels={'客户类型': '客户细分', '销售额': '单笔订单金额 (元)'},
    template='plotly_white'
)
"""
            insight = """### 📌 核心结论与数据事实
- **企业客户**的单笔订单销售额中位数与上限均显著高于个人消费者与小型商户，高价值大额订单密集。
- **个人消费者**样本量庞大，但订单金额方差较小，主要集中在中低频小额高频场景。

### 🔍 深度业务/数据机理解析
- 企业的采购行为具有计划性、批量化和集中结算特征，决策链较长但粘性高。
- 个人消费者易受促销折扣刺激，退换货率相对更高。

### 💡 针对性决策与行动建议
- 推进 B 端与 C 端差异化运营：为企业客户开通专属账期与大宗采购合同折扣；对 C 端强化会员积分与个性化推荐。
"""
            return code, insight


# -----------------------------------------------------------------------------
# 7. Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")

    # 7.1 Dataset Selection
    st.markdown("### 1. Data Source")
    data_source = st.radio(
        "选择数据模式",
        options=["内置基准数据集 (一键体验)", "上传自定义数据 (CSV / Excel)"],
        index=0
    )

    df_loaded = None
    dataset_name = ""

    if data_source == "内置基准数据集 (一键体验)":
        preset_choice = st.selectbox(
            "选择数据集",
            options=["🚢 泰坦尼克号乘客数据 (Titanic)", "🛒 全渠道电商销售数据 (E-commerce)"]
        )
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "data")

        if "泰坦尼克号" in preset_choice:
            file_path = os.path.join(data_dir, "titanic.csv")
            dataset_name = "titanic"
            if os.path.exists(file_path):
                df_loaded = pd.read_csv(file_path)
            else:
                st.error(f"Data file not found: {file_path}")
        else:
            file_path = os.path.join(data_dir, "ecommerce_sales.csv")
            dataset_name = "ecommerce"
            if os.path.exists(file_path):
                df_loaded = pd.read_csv(file_path)
            else:
                st.error(f"Data file not found: {file_path}")

    else:
        uploaded_file = st.file_uploader(
            "上传 CSV 或 Excel 文件",
            type=["csv", "xlsx", "xls"]
        )
        if uploaded_file is not None:
            dataset_name = uploaded_file.name
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_loaded = pd.read_csv(uploaded_file)
                else:
                    df_loaded = pd.read_excel(uploaded_file)
                st.success(f"成功加载文件: {uploaded_file.name}")
            except Exception as e:
                st.error(f"文件读取失败: {e}")

    if df_loaded is not None:
        st.session_state.current_df = df_loaded
        st.session_state.dataset_name = dataset_name

    st.markdown("---")

    # 7.2 LLM Configuration
    st.markdown("### 2. LLM Configuration")

    use_demo_mode = st.toggle(
        "🎯 离线演示模式 (无需 API Key)",
        value=True,
        help="开启后无需配置 API Key 即可体验内置智能分析与自动代码渲染，便于快速演示与离线体验。"
    )

    if not use_demo_mode:
        provider = st.selectbox(
            "服务提供商预设",
            options=[
                "DeepSeek (推荐: 高性价比/强推理)",
                "OpenAI (官方 GPT 系列)",
                "硅基流动 SiliconFlow (Qwen/DeepSeek)",
                "通义千问 DashScope",
                "月之暗面 Kimi (Moonshot)",
                "自定义 OpenAI 兼容接口"
            ]
        )

        default_url = "https://api.deepseek.com/v1"
        default_model = "deepseek-chat"

        if "OpenAI" in provider:
            default_url = "https://api.openai.com/v1"
            default_model = "gpt-4o-mini"
        elif "硅基流动" in provider:
            default_url = "https://api.siliconflow.cn/v1"
            default_model = "Qwen/Qwen2.5-72B-Instruct"
        elif "通义千问" in provider:
            default_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            default_model = "qwen-plus"
        elif "月之暗面" in provider:
            default_url = "https://api.moonshot.cn/v1"
            default_model = "moonshot-v1-8k"
        elif "自定义" in provider:
            default_url = "http://localhost:11434/v1"
            default_model = "llama3"

        api_url = st.text_input("API Base URL", value=default_url)
        api_key = st.text_input("API Key", type="password", placeholder="sk-...")
        model_name = st.text_input("模型名称 (Model)", value=default_model)
        temperature = st.slider("多样性 (Temperature)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    else:
        st.info("💡 当前已启用 **离线演示模式**。系统自动激活内置模式匹配与可视化生成引擎。")
        api_url = ""
        api_key = ""
        model_name = "demo-agent"
        temperature = 0.2

    st.markdown("---")
    st.markdown("""
    **Project Info**:
    - **Name**: Auto Data Agent
    - **Tech**: Streamlit + Pandas + Plotly + LLM
    - **Open Source**: MIT License
    """)


# -----------------------------------------------------------------------------
# 8. Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🤖 Auto Data Agent (自动数据分析智能体)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">基于 <b>Streamlit + Pandas + Plotly + LLM API</b> 的端到端数据自动化画像与自然语言交互分析系统</div>', unsafe_allow_html=True)

df = st.session_state.current_df

if df is None:
    st.warning("⚠️ 请在左侧侧边栏选择加载内置数据集或上传您的 CSV/Excel 数据文件。")
    st.stop()


# -----------------------------------------------------------------------------
# 9. Module 1: Automated Exploratory Data Analysis (Auto-EDA)
# -----------------------------------------------------------------------------
with st.expander("📌 模块一：数据集基础概况与自动画像 (Auto-EDA)", expanded=True):
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    total_rows, total_cols = df.shape
    total_missing = df.isnull().sum().sum()
    missing_rate = (total_missing / (total_rows * total_cols)) * 100 if (total_rows * total_cols) > 0 else 0
    duplicate_rows = df.duplicated().sum()
    mem_usage = df.memory_usage(deep=True).sum() / 1024

    with col_m1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">总样本行数</div><div class="metric-value">{total_rows:,}</div></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">特征字段数</div><div class="metric-value">{total_cols}</div></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">缺失值单元格</div><div class="metric-value">{total_missing} ({missing_rate:.2f}%)</div></div>', unsafe_allow_html=True)
    with col_m4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">重复行总数</div><div class="metric-value">{duplicate_rows}</div></div>', unsafe_allow_html=True)
    with col_m5:
        st.markdown(f'<div class="metric-card"><div class="metric-title">内存占用</div><div class="metric-value">{mem_usage:.1f} KB</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_preview, tab_missing, tab_stats, tab_corr = st.tabs([
        "📋 数据预览与字段字典",
        "⚠️ 缺失值诊断",
        "📈 描述性统计 (Describe)",
        "🔥 数值型特征相关性热力图"
    ])

    with tab_preview:
        col_t1, col_t2 = st.columns([1.6, 1.0])
        with col_t1:
            st.markdown("**前 5 行样本预览**:")
            st.dataframe(df.head(5), use_container_width=True)
        with col_t2:
            st.markdown("**字段类型明细字典**:")
            schema_df = pd.DataFrame({
                "字段名": df.columns,
                "数据类型 (dtype)": [str(t) for t in df.dtypes],
                "非空数量": df.notnull().sum().values,
                "非空率": [(v / total_rows) * 100 for v in df.notnull().sum().values]
            })
            schema_df["非空率"] = schema_df["非空率"].map("{:.1f}%".format)
            st.dataframe(schema_df, height=220, use_container_width=True)

    with tab_missing:
        null_counts = df.isnull().sum()
        null_df = pd.DataFrame({
            "字段名": null_counts.index,
            "缺失数量": null_counts.values,
            "缺失比例(%)": (null_counts.values / total_rows * 100).round(2)
        }).sort_values(by="缺失数量", ascending=False)

        if null_df["缺失数量"].sum() == 0:
            st.success("🎉 数据集质量优良！当前数据集无任何字段存在缺失值。")
        else:
            col_null_1, col_null_2 = st.columns([1.2, 0.8])
            with col_null_1:
                fig_null = px.bar(
                    null_df[null_df["缺失数量"] > 0],
                    x="字段名",
                    y="缺失比例(%)",
                    text="缺失比例(%)",
                    title="各字段缺失值比例分布",
                    color="缺失比例(%)",
                    color_continuous_scale="Reds",
                    template="plotly_white"
                )
                fig_null.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                st.plotly_chart(fig_null, use_container_width=True)
            with col_null_2:
                st.markdown("**缺失字段明细清单**:")
                st.dataframe(null_df[null_df["缺失数量"] > 0], use_container_width=True)

    with tab_stats:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        if num_cols:
            st.markdown("**数值型字段统计摘要 (Numerical Describe)**:")
            desc_df = df[num_cols].describe().T
            desc_df["skew"] = df[num_cols].skew()
            st.dataframe(desc_df.round(2), use_container_width=True)

        if cat_cols:
            st.markdown("**类别型字段概览 (Categorical Summary)**:")
            cat_summary = []
            for col in cat_cols:
                top_val = df[col].mode().iloc[0] if not df[col].dropna().empty else "N/A"
                unique_cnt = df[col].nunique()
                cat_summary.append({
                    "字段名": col,
                    "唯一定义值数": unique_cnt,
                    "众数(Top Value)": str(top_val),
                    "众数出现频次": (df[col] == top_val).sum() if top_val != "N/A" else 0
                })
            st.dataframe(pd.DataFrame(cat_summary), use_container_width=True)

    with tab_corr:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] >= 2:
            corr_matrix = numeric_df.corr().round(3)
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="数值型特征 Pearson 相关性矩阵热力图",
                template="plotly_white",
                zmin=-1,
                zmax=1
            )
            fig_corr.update_layout(height=480)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("当前数据集中数值型特征少于 2 个，无法计算相关性矩阵。")


st.markdown("---")


# -----------------------------------------------------------------------------
# 10. Module 2: Natural Language Conversational Query
# -----------------------------------------------------------------------------
st.markdown("### 💬 模块二：自然语言交互数据探索")
st.markdown("输入您的分析需求，智能体将自动提取元数据生成 Prompt，驱动大模型生成数据处理与 Plotly 交互图表代码。")

dataset_key = "titanic" if ("titanic" in st.session_state.dataset_name.lower() or "survived" in [c.lower() for c in df.columns]) else "ecommerce"

if dataset_key == "titanic":
    preset_prompts = [
        "📊 分析不同性别的生还率差异，绘制对比柱状图",
        "🚢 探究船舱等级 (Pclass) 与生还率、平均票价的关系",
        "📈 探索乘客年龄与票价的散点分布（按幸存与否上色）",
        "🥧 分析不同登船港口 (Embarked) 的乘客数量分布饼图"
    ]
else:
    preset_prompts = [
        "📊 各商品大类的总销售额与总利润规模对比柱状图",
        "🥧 全国各销售区域的营收贡献占比饼图",
        "📦 分析不同客户类型的单笔订单销售额分布箱线图",
        "📈 探究折扣率与净利润之间的相关关系与散点图"
    ]

st.markdown("**💡 快速体验提问示例 (点击填入):**")
btn_cols = st.columns(4)
for idx, prompt_text in enumerate(preset_prompts):
    with btn_cols[idx]:
        if st.button(prompt_text, key=f"quick_btn_{idx}", use_container_width=True):
            st.session_state.last_query = prompt_text

user_input = st.text_area(
    "请输入您的数据分析需求：",
    value=st.session_state.last_query,
    placeholder="例如：按不同性别统计生还率并绘制柱状图，同时给出关键结论...",
    height=85
)

run_button = st.button("🚀 启动智能体分析", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# 11. Module 3: Prompt Assembly, Safe Execution & Dual-column Rendering
# -----------------------------------------------------------------------------
if run_button and user_input.strip():
    query = user_input.strip()
    st.session_state.last_query = query

    with st.status("🤖 智能体正在解析需求并执行全流程分析...", expanded=True) as status:
        st.write("🔍 **步骤 1/3**: 提取数据 Schema 元数据并构建严格约束 Prompt...")

        dtypes_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
        sample_records = df.head(3).to_dict(orient="records")

        system_prompt = """你是一个世界顶尖的数据科学与商业智能可视化专家。
你的任务是根据用户给出的数据集 Schema 元数据和自然语言分析需求，编写出高质量、可直接运行的 Python 数据分析与 Plotly 可视化代码。

【重要约束与规范】：
1. 你的 Python 代码必须直接使用已有变量 `df`（即当前 Pandas DataFrame），绝不能重新读取文件。
2. 必须生成一个 Plotly 图表对象并严格将其赋值给变量 `fig`（例如 `fig = px.bar(...)` 或 `fig = go.Figure(...)`）。
3. 必须将代码中计算出的关键核心聚合指标、数值或汇总表格，赋值给变量 `summary_data`（可以是 dict 或 DataFrame），以便后续下游大模型生成深度业务洞察。
4. 图表风格要求专业优雅：使用 `template='plotly_white'`，带有清晰的图表标题、轴标签、图例说明。
5. 绝不能破坏原有的 `df` 数据；严禁包含 `os`, `sys`, `subprocess`, `open` 等任何涉及文件读写或操作系统的高危库。
6. 输出格式：直接输出包裹在 ```python ... ``` 块内的 Python 代码，不要输出任何非代码的闲聊。"""

        user_prompt = f"""【数据集信息】：
- 数据维度：{df.shape[0]} 行, {df.shape[1]} 列
- 字段与类型：{json.dumps(dtypes_info, ensure_ascii=False)}
- 前 3 行数据样例：{json.dumps(sample_records, ensure_ascii=False)}

【用户分析需求】：
{query}

请根据上述数据集特征和分析需求，生成 Python 分析与 Plotly 绘图代码。记得严格赋值给 `fig` 和 `summary_data`。"""

        generated_code = ""
        structured_insight = ""

        st.write("⚡ **步骤 2/3**: 大模型推理生成 Python 代码与绘图逻辑...")
        if use_demo_mode or not api_key.strip():
            generated_code, mock_insight = generate_mock_analysis(query, dataset_key, df)
            structured_insight = mock_insight
        else:
            try:
                raw_response = call_llm_api(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    api_key=api_key,
                    base_url=api_url,
                    model=model_name,
                    temperature=temperature
                )
                generated_code = extract_python_code(raw_response)
            except Exception as e:
                st.error(f"大模型代码生成失败: {e}。将自动降级使用规则智能生成。")
                generated_code, structured_insight = generate_mock_analysis(query, dataset_key, df)

        st.write("🛡️ **步骤 3/3**: 安全沙箱 (exec) 执行代码并捕获渲染结果...")
        success, fig_result, summary_res, exec_log = execute_generated_code(generated_code, df)

        if not success:
            status.update(label="❌ 代码执行遇到错误", state="error", expanded=True)
            st.error(f"代码执行异常:\n\n{exec_log}")
            st.markdown("**尝试执行的生成代码如下**:")
            st.code(generated_code, language="python")
            st.stop()

        if not use_demo_mode and api_key.strip() and not structured_insight:
            st.write("📝 正在根据精确计算数据生成结构化决策洞察...")
            insight_system_prompt = """你是一个资深商业数据分析师与战略决策顾问。
根据用户的提问、生成的图表逻辑以及代码执行所计算出的真实数据统计结果 (summary_data)，生成一份高度结构化、专业、洞察深刻的数据分析报告。

报告必须包含三个明确部分，使用 Markdown 格式：
### 📌 核心结论与数据事实 (Key Findings)
- 用数据说话，明确列出最关键的统计事实、占比、极值或均值。

### 🔍 深度业务/数据机理解析 (Deep Insights)
- 结合业务背景分析为什么会出现这种现象，背后的深层原因是什么。

### 💡 针对性决策与行动建议 (Actionable Recommendations)
- 给出 2-3 条具有可落地性、前瞻性的管理或运营行动建议。"""

            insight_user_prompt = f"""【用户提问】：{query}
【代码计算出的核心统计数据 (summary_data)】：
{json.dumps(summary_res, ensure_ascii=False, default=str)}

请输出结构化的分析报告。"""
            try:
                structured_insight = call_llm_api(
                    prompt=insight_user_prompt,
                    system_prompt=insight_system_prompt,
                    api_key=api_key,
                    base_url=api_url,
                    model=model_name,
                    temperature=0.3
                )
            except Exception as e:
                structured_insight = f"调用大模型生成洞察报告时遇到网络波动: {e}。直接呈现统计计算结果。"

        status.update(label="✅ 分析完成！可视化图表与数据洞察已联动呈现", state="complete", expanded=False)

    # -------------------------------------------------------------------------
    # 12. Dual-Column Presentation
    # -------------------------------------------------------------------------
    st.markdown("### 📊 智能体分析结果联动呈现")

    col_chart, col_report = st.columns([1.15, 0.85])

    # Left Column: Interactive Plotly Chart
    with col_chart:
        st.markdown("#### 📈 交互式数据可视化 (Plotly)")
        st.plotly_chart(fig_result, use_container_width=True)

        with st.expander("💻 查看生成的 Python 代码与数据处理逻辑"):
            st.code(generated_code, language="python")

        with st.expander("🔢 查看代码计算生成的中间统计数据 (summary_data)"):
            if isinstance(summary_res, pd.DataFrame):
                st.dataframe(summary_res, use_container_width=True)
            else:
                st.write(summary_res)

    # Right Column: Structured Insights
    with col_report:
        st.markdown("#### 🧠 结构化数据分析与决策洞察")
        st.markdown(f'<div class="insight-card">{structured_insight}</div>', unsafe_allow_html=True)

        st.download_button(
            label="📥 导出本条分析结论 (Markdown)",
            data=f"# 数据分析报告: {query}\n\n{structured_insight}\n\n## 生成代码\n```python\n{generated_code}\n```\n",
            file_name=f"analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )


# -----------------------------------------------------------------------------
# 13. Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9CA3AF; font-size: 0.85rem;">
    Auto Data Agent · Automated Exploratory Data Analysis & Natural Language Agent · Powered by Streamlit + Plotly + LLM
</div>
""", unsafe_allow_html=True)
