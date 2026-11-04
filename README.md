# 🤖 Auto Data Agent (自动数据分析智能体)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **一句话项目介绍**：  
> **Auto Data Agent 是一款基于 Streamlit、Pandas、Plotly 与大模型 API 构建的端到端自动化数据探索与自然语言对话分析智能体，支持零门槛数据体检画像、代码安全沙箱执行与“图表-决策洞察”双栏联动呈现。**

---

## 📌 项目做什么 (What It Does)

面对复杂的数据集，传统数据分析流程常常需要耗费大量时间编写重复的探索性数据分析（EDA）代码，同时要求使用者精通 Python、Pandas 与可视化库。

**Auto Data Agent** 旨在打破这一门槛，实现**“即席上传、秒级画像、对话即分析、图表即洞察”**的完整数据分析闭环：

1. **自动探索性数据画像 (Auto-EDA)**
   - **全量指标体检**：自动呈现样本行数、特征列数、缺失值总数与百分比、重复行数与内存占用。
   - **多标签诊断面板**：
     - 📋 **数据预览与字段字典**：快速查看前 5 行样本与每个字段的数据类型（`dtypes`）与非空率；
     - ⚠️ **缺失值诊断**：动态绘制缺失比例条形图；
     - 📈 **描述统计**：数值字段的均值、标准差、分位数、偏度，类别字段的众数与唯一定义值分布；
     - 🔥 **相关性热力图**：自动识别数值型特征，一键生成 Pearson 相关系数矩阵热力图。
2. **自然语言对话分析 (NL2Code)**
   - 用户只需输入中文问题（例如：“*分析不同船舱等级的生还率与票价分布*”），程序自动提取表结构元数据组装 Prompt，引导大模型直接生成标准的 Pandas 统计与 Plotly 交互图表代码。
3. **安全沙箱执行与双栏联动展现**
   - **安全沙箱**：内置白名单过滤高危系统调用（`os`, `sys`, `subprocess` 等），在受限命名空间内安全执行。
   - **左栏（图表与源码）**：交互式 Plotly 矢量图表，支持悬浮看数、缩放，并可展开审查生成的 Python 源码及计算出的聚合数据（`summary_data`）。
   - **右栏（结构化深度洞察）**：自动调用大模型针对真实计算结果生成**【核心事实】、【机理解析】与【行动建议】**三段式决策报告，拒绝大模型“数据幻觉”。
4. **开箱即用与双模式运行**
   - 内置经典泰坦尼克号数据（`titanic.csv`）与电商销售模拟数据（`ecommerce_sales.csv`），自带推荐提问按钮。
   - 内置**离线演示模式 (Demo Mode)**，无需配置 API Key 即可一键体验完整功能！

---

## 🚀 怎样运行 (How to Run)

### 1. 环境准备
确保本机已安装 Python 3.10 或更高版本。建议使用虚拟环境（可选）：
```bash
python -m venv venv
# Windows 激活虚拟环境:
venv\Scripts\activate
# Linux/macOS 激活虚拟环境:
source venv/bin/activate
```

### 2. 克隆仓库与安装依赖
```bash
# 克隆仓库
git clone https://github.com/<your-username>/auto-data-agent.git
cd auto-data-agent

# 安装依赖
pip install -r requirements.txt
```

### 3. 启动应用
方式一（推荐：Windows 用户）：直接双击根目录下的 **`run.bat`** 启动。  
方式二：在终端中运行：
```bash
streamlit run app.py
```
启动成功后，控制台将输出本地访问地址（默认：`http://localhost:8501`），浏览器将自动弹出主界面。

---

## ⚙️ 模型配置与使用模式

系统侧边栏提供了灵活的双运行模式：

| 模式 | 适用场景 | 配置方法 |
| :--- | :--- | :--- |
| **🎯 离线演示模式** (默认开启) | 随堂答辩、快速测试、无 API Key | 保持侧边栏“离线演示模式”开关开启即可，系统采用内置规则与模板引擎即时响应。 |
| **🌐 云端大模型模式** | 自定义复杂提问、生产级深度分析 | 关闭“离线演示模式”，填入您的 API Key（支持 DeepSeek、OpenAI、硅基流动、通义千问、Kimi 等标准 OpenAI 接口规范）。 |

---

## 📂 项目结构

```text
auto-data-agent/
├── app.py                 # Streamlit 单文件核心应用
├── requirements.txt       # Python 依赖清单
├── run.bat                # Windows 一键启动脚本
├── LICENSE                # MIT 开源许可证
├── README.md              # 项目技术与运行说明
├── .gitignore             # Git 忽略配置
└── data/                  # 内置基准测试数据集
    ├── titanic.csv        # 泰坦尼克号乘客数据
    └── ecommerce_sales.csv# 全渠道电商销售模拟数据
```

---

## 🛠️ 技术栈

- **前端交互框架**：[Streamlit](https://streamlit.io/)
- **数据科学与计算**：[Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **交互式可视化**：[Plotly](https://plotly.com/python/)
- **网络与接口交互**：[Requests](https://requests.readthedocs.io/) (兼容 OpenAI RESTful 规范)

---

## 📄 开源许可证

本项目基于 [MIT License](LICENSE) 开源。欢迎 Star、Fork 与提 Issue！
