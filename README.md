
# Social Media Audience Analysis System
# 社交媒体受众分析系统

## 1. 项目简介

MediaInsight 是一个基于 Python Flask 开发的社交媒体受众分析系统，结合人工智能与自然语言处理技术，对社交媒体评论中的用户情感、关注重点和讨论话题进行分析。

在数字媒体环境中，用户评论不仅是内容消费后的反馈，也是了解受众态度、内容传播效果和公众关注趋势的重要数据来源。本项目将 AI 与媒体内容分析相结合，通过预训练情感分析模型识别评论的情感倾向，利用 TF-IDF 提取高频关键词，并借助 LDA 主题模型发现用户讨论的潜在话题。同时，系统通过数据可视化呈现情感分布、关键词和主题分析结果，帮助使用者从大量非结构化文本中提取有价值的信息。

项目体现了 AI 与媒体数据分析的结合：将自然语言处理和机器学习技术应用于数字媒体评论数据，把分散的用户反馈转化为可观察、可分析的受众洞察，为理解媒体内容的受众反应、用户兴趣及讨论趋势提供辅助。

项目使用新浪微博评论数据集作为示例数据，主要用于验证和展示完整的数据处理与分析流程。

**项目定位：** NLP 数据分析与可视化应用。

## 2. 核心功能

### 2.1 数据上传与预处理

- 支持 CSV 文件上传。
- 检查必要的数据字段。
- 支持 UTF-8、UTF-8 BOM 和 GB18030 编码。
- 对评论数据进行清洗和预处理。
- 处理空评论、重复数据等异常情况。

### 2.2 情感分析

- 使用预训练多语言情感分析模型。
- 将评论分类为正面、负面和中性。
- 记录模型预测结果及置信度。
- 统计不同情感类别的数量和占比。

### 2.3 关键词提取

- 使用 Jieba 进行中文分词。
- 使用 TF-IDF 方法提取评论关键词。
- 展示关键词及其权重。

### 2.4 话题发现

- 使用 LDA 主题模型进行话题发现。
- 提取不同话题的代表性关键词。
- 辅助理解评论数据中的主要讨论内容。

### 2.5 数据可视化与分析报告

- 使用 ECharts 展示情感分布。
- 展示不同情感类别的数量。
- 可视化关键词提取结果。
- 展示 LDA 话题分析结果。
- 自动生成情感分析摘要。
- 展示模型平均预测置信度及置信度等级分布。

### 2.6 评论筛选与数据导出

- 支持按评论关键词搜索。
- 支持按情感类别筛选。
- 支持搜索与情感类别组合筛选。
- 支持导出情感分析结果 CSV 文件。

## 3. 技术栈

| 分类 | 技术 |
|---|---|
| 后端框架 | Python、Flask |
| 数据处理 | Pandas |
| 中文分词 | Jieba |
| 机器学习 | Scikit-learn |
| 情感分析 | Transformers、PyTorch |
| 关键词提取 | TF-IDF |
| 话题发现 | LDA |
| 前端 | HTML、CSS、JavaScript |
| 数据可视化 | ECharts |
| 数据格式 | CSV |

## 4. 系统处理流程

1. 用户上传 CSV 评论文件。
2. 系统检查文件格式和必要字段。
3. 对评论数据进行清洗和预处理。
4. 使用预训练模型进行情感分类。
5. 使用 TF-IDF 提取关键词。
6. 使用 LDA 发现潜在话题。
7. 汇总分析结果并生成摘要。
8. 使用 ECharts 展示分析结果。
9. 用户可以搜索、筛选评论并导出分析结果。

## 5. 项目结构

```text
social-media-analysis/
├── backup_day10/
│   ├── keywords.csv
│   ├── processed_comments.csv
│   ├── sentiment_comments.csv
│   ├── sentiment_summary.csv
│   └── topics.csv
│
├── data/
│   ├── sample_comments.csv
│   ├── processed_comments.csv
│   ├── sentiment_comments.csv
│   ├── sentiment_summary.csv
│   ├── keywords.csv
│   ├── topics.csv
│   └── stopwords.txt
│
├── services/
│   ├── analysis_summary.py
│   ├── nlp_preprocess.py
│   ├── preprocess.py
│   ├── sentiment_analysis.py
│   └── topic_analysis.py
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   ├── index.html
│   └── upload.html
│
├── app.py
├── requirements.txt
├── README.md
│
├── check_analysis_summary.py
├── generate_test.py
├── test_data.py
├── test_nlp.py
├── test_sentiment.py
├── test_topic_analysis.py
└── test_visualization.py
```

## 6. 环境配置与运行

### 6.1 克隆项目

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd social-media-analysis
```

### 6.2 创建虚拟环境

```bash
python -m venv .venv
```

Windows PowerShell 激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 6.3 安装依赖

```bash
pip install -r requirements.txt
```

### 6.4 启动项目

```bash
python app.py
```

启动成功后，根据终端显示的本地地址，在浏览器中访问系统。

首次运行情感分析功能时，预训练模型可能需要下载，请确保网络环境能够访问模型仓库。

## 7. 数据说明

本项目使用新浪微博评论数据集进行分析与功能测试。

数据来源： 新浪微博评论数据集，原始数据集下载地址为百度网盘分享链接。

数据规模： 原始数据集包含 10 万多条评论，带有情感标注；本项目从中选取了 10,000 条评论作为分析样本。

情感标注： 原始数据集包含正向和负向评论。

数据类型： 中文社交媒体用户评论文本及其情感标注。

获取方式： 通过数据分享者提供的百度网盘链接：https://github.com/SophonPlus/ChineseNlpCorpus.git 下载数据集，再从原始数据中选取 10,000 条评论用于项目分析，并整理为 CSV 格式供系统读取。

数据主要用于测试和展示评论数据清洗、情感识别、关键词提取、话题发现及可视化分析流程。

数据下载地址：通过网盘分享的文件：weibo_comments_10k.csv
链接: https://pan.baidu.com/s/1je0PmigbQUbOOCYTOzUWpQ?pwd=1111 提取码: 1111

## 8. 模型与结果说明

- 情感分析使用预训练多语言模型进行预测。
- 模型置信度反映模型对预测类别的估计把握程度，不等于实际预测准确率。
- TF-IDF 关键词权重用于衡量词语在文本集合中的相对重要程度。
- LDA 话题结果是基于当前数据集和模型参数得到的主题分布，可能随数据变化而变化。

## 9. 项目局限性

- 情感分类结果可能受到文本语境、反讽表达和领域词汇的影响。
- 话题发现效果受到数据规模、文本质量和模型参数影响。
- 当前项目以 CSV 文件作为数据输入和结果存储形式。

## 10. 项目成果与后续改进

目前已经完成一个可以在本地运行的 Web 分析原型，实现了从评论数据上传、文本预处理到情感分析、关键词提取、话题发现和结果导出的基本流程。

后续希望进一步扩展数据来源和分析维度，例如比较不同话题下用户的情感倾向，观察评论内容随时间发生的变化，并尝试优化模型对中文网络用语和复杂表达的识别能力。

通过这个项目，我希望进一步探索人工智能和数据分析在媒体内容理解、用户反馈分析等场景中的实际应用。
