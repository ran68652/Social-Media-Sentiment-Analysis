# Social Media Audience Analysis System
# 社交媒体受众分析系统

## 1. 项目简介
本项目是一个基于 Python Flask 开发的社交媒体受众分析系统，主要用于对中文社交媒体评论进行数据处理、情感分析、关键词提取和话题发现。
系统支持 CSV 文件上传与数据清洗，使用预训练多语言情感分析模型识别评论情感倾向，并结合 TF-IDF 和 LDA 技术提取关键词、发现潜在讨论话题。
同时，系统通过 ECharts 实现数据可视化，并提供情感分析摘要、评论筛选和分析结果导出功能，帮助用户了解评论数据中的情感分布与讨论内容。
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
- 展示模型平均预测置信度。

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
├── __pycache__/
│   └── app.cpython-313.pyc
├── backup_day10/
│   ├── keywords.csv
│   ├── processed_comments.csv
│   ├── sentiment_comments.csv
│   ├── sentiment_summary.csv
│   └── topics.csv
├── data/
│   ├── empty_test.csv
│   ├── keywords.csv
│   ├── missing_content_test.csv
│   ├── nlp_comments.csv
│   ├── processed_comments.csv
│   ├── sample_comments.csv
│   ├── sentiment_comments.csv
│   ├── sentiment_summary.csv
│   ├── stopwords.txt
│   ├── test_duplicate_id.csv
│   ├── test_empty_comments.csv
│   ├── test_long_comment.csv
│   ├── test_missing_content.csv
│   ├── test_upload.csv
│   ├── tfidf_keywords.csv
│   ├── topics.csv
│   └── word_frequency.csv
├── services/
│   ├── __pycache__/
│   │   ├── analysis_summary.cpython-313.pyc
│   │   ├── nlp_preprocess.cpython-313.pyc
│   │   ├── preprocess.cpython-313.pyc
│   │   ├── sentiment_analysis.cpython-313.pyc
│   │   └── topic_analysis.cpython-313.pyc
│   ├── analysis_summary.py
│   ├── nlp_preprocess.py
│   ├── preprocess.py
│   ├── sentiment_analysis.py
│   └── topic_analysis.py
├── static/
│   └── css/
│       └── style.css
├── templates/
│   ├── index.html
│   └── upload.html
├── app.py
├── check_analysis_summary.py
├── generate_test.py
├── requirements.txt
├── test_data.py
├── test_nlp.py
├── test_sentiment.py
├── test_topic_analysis.py
└── test_visualization.py

## 6.环境配置与运行

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
项目中的示例评论为人工模拟的中文城市文旅主题评论，主要用于展示数据上传、文本处理、情感分类和话题分析流程。
这些数据不代表真实社交媒体平台采集结果，也不代表真实用户调查。

## 8. 模型与结果说明
- 情感分析使用预训练多语言模型进行预测。
- 模型置信度反映模型对预测类别的估计把握程度，不等于实际预测准确率。
- TF-IDF 关键词权重用于衡量词语在文本集合中的相对重要程度。
- LDA 话题结果是基于当前数据集和模型参数得到的主题分布，可能随数据变化而变化。

## 9. 项目局限性
- 当前示例数据为模拟数据，尚不能代表真实平台用户观点。
- 情感分类结果可能受到文本语境、反讽表达和领域词汇的影响。
- 话题发现效果受到数据规模、文本质量和模型参数影响。
- 当前项目以 CSV 文件作为数据输入和结果存储形式。

## 10. 后续改进方向
- 接入合规的数据来源，支持真实评论数据分析。
- 使用人工标注数据评估情感分类模型的准确率、精确率、召回率和 F1 值。
- 针对文旅领域优化情感分类效果。
- 增加更多文本分析和可视化功能。
- 引入数据库，提升数据管理能力。
- 优化大规模评论数据的处理效率。

## 11. 项目声明
本项目为学习与作品集展示用途，重点展示 Flask Web 开发、自然语言处理、机器学习应用和数据可视化的综合实践能力。
