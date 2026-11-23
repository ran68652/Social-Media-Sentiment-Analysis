from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file
)
import pandas as pd
import io

from pathlib import Path

from services.preprocess import clean_comments

from services.sentiment_analysis import (
    analyze_sentiment,
    generate_sentiment_summary
)

from services.topic_analysis import (
    extract_keywords,
    discover_topics
)

from services.analysis_summary import (
    generate_analysis_summary
)


# 1. Flask 基础配置

app = Flask(__name__)

app.secret_key = "change-this-to-a-random-secret"

# 上传文件大小限制：5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# 2. 文件路径配置

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

SAMPLE_FILE = DATA_DIR / "sample_comments.csv"

PROCESSED_FILE = DATA_DIR / "processed_comments.csv"

SENTIMENT_FILE = DATA_DIR / "sentiment_comments.csv"

SUMMARY_FILE = DATA_DIR / "sentiment_summary.csv"

KEYWORDS_FILE = DATA_DIR / "keywords.csv"

TOPICS_FILE = DATA_DIR / "topics.csv"


# 3. 数据读取函数

def load_comments():
    """读取当前评论数据"""

    if PROCESSED_FILE.exists():
        data_file = PROCESSED_FILE
    else:
        data_file = SAMPLE_FILE

    if not data_file.exists():
        raise FileNotFoundError(
            "找不到评论数据文件，请先准备评论数据"
        )

    df = pd.read_csv(
        data_file,
        encoding="utf-8-sig"
    )

    if "content" not in df.columns:
        raise ValueError("评论数据缺少 content 字段")

    if "comment_id" not in df.columns:
        df["comment_id"] = range(1, len(df) + 1)

    if "platform" not in df.columns:
        df["platform"] = "未知平台"

    return df


def read_uploaded_csv(file_bytes):
    """读取上传的 CSV 文件并处理编码异常"""

    if not file_bytes:
        raise ValueError("上传的文件为空")

    for encoding in ["utf-8-sig", "utf-8", "gb18030"]:

        try:
            df = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
                on_bad_lines="error"
            )

            return df

        except UnicodeDecodeError:
            continue

        except pd.errors.EmptyDataError:
            raise ValueError("CSV 文件没有有效内容")

        except pd.errors.ParserError:
            raise ValueError(
                "CSV 文件格式错误，请检查分隔符和数据列"
            )

    raise ValueError(
        "无法识别文件编码，请使用 UTF-8 或 GB18030 编码"
    )


# 4. 关键词和话题分析结果保存函数

def save_topic_analysis(df):
    """执行关键词提取和话题探索，并保存结果"""

    # TF-IDF 关键词提取
    keyword_df = extract_keywords(
        df,
        top_n=10
    )

    keyword_df.to_csv(
        KEYWORDS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # LDA 话题探索
    topics = discover_topics(
        df,
        n_topics=3,
        n_words=5
    )

    # 将话题数据整理成 DataFrame
    topic_rows = []

    for topic in topics:

        topic_rows.append({
            "topic_id": topic["topic_id"],
            "keywords": "、".join(topic["keywords"])
        })

    topic_df = pd.DataFrame(
        topic_rows,
        columns=["topic_id", "keywords"]
    )

    topic_df.to_csv(
        TOPICS_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# 5. 关键词和话题分析结果读取函数

def load_topic_analysis():
    """读取已保存的关键词和话题分析结果"""

    keyword_df = pd.read_csv(
        KEYWORDS_FILE,
        encoding="utf-8-sig"
    )

    topic_df = pd.read_csv(
        TOPICS_FILE,
        encoding="utf-8-sig"
    )

    # 转换关键词数据
    keyword_data = (
        keyword_df
        .fillna(0)
        .to_dict(orient="records")
    )

    # 转换话题数据
    topics = []

    for _, row in topic_df.iterrows():

        keywords = str(row["keywords"])

        topics.append({
            "topic_id": int(row["topic_id"]),
            "keywords": [
                word
                for word in keywords.split("、")
                if word
            ]
        })

    return keyword_data, topics


# 6. 情感分析报告默认数据

def get_empty_analysis_summary():
    """生成默认的空情感分析报告"""

    return {
        "has_data": False,
        "total": 0,
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0,
        "positive_percentage": 0,
        "negative_percentage": 0,
        "neutral_percentage": 0,
        "dominant_sentiment": "暂无数据",
        "dominant_percentage": 0,
        "summary_text": "目前没有可供分析的评论数据。"
    }


# 7. 首页路由

@app.route("/")
def index():

    # 默认情感统计数据
    sentiment_data = [
        {
            "sentiment": "正面",
            "count": 0,
            "percentage": 0
        },
        {
            "sentiment": "负面",
            "count": 0,
            "percentage": 0
        },
        {
            "sentiment": "中性",
            "count": 0,
            "percentage": 0
        }
    ]

    sentiment_comments = []

    # 默认情感分析报告
    analysis_summary = get_empty_analysis_summary()

    # 默认关键词和话题数据
    keyword_data = []
    topics = []

    keyword_error = None
    topic_error = None

    total_count = 0
    platform_count = 0
    comments = []

    try:

        # 读取评论数据
        df = load_comments()

        total_count = len(df)

        platform_count = (
            df["platform"].nunique()
            if "platform" in df.columns
            else 0
        )

        comments = (
            df.head(10)
            .fillna("")
            .to_dict(orient="records")
        )

        # 读取情感分析详细结果
        if SENTIMENT_FILE.exists():

            sentiment_df = pd.read_csv(
                SENTIMENT_FILE,
                encoding="utf-8-sig"
            )

            sentiment_comments = (
                sentiment_df
                .fillna("")
                .to_dict(orient="records")
            )

            # 根据详细分析结果生成文字报告
            try:

                analysis_summary = generate_analysis_summary(
                    sentiment_df
                )

            except Exception:

                app.logger.exception(
                    "情感分析报告生成失败"
                )

                analysis_summary = get_empty_analysis_summary()

        # 读取情感统计结果
        if SUMMARY_FILE.exists():

            summary_df = pd.read_csv(
                SUMMARY_FILE,
                encoding="utf-8-sig"
            )

            sentiment_data = (
                summary_df
                .fillna(0)
                .to_dict(orient="records")
            )

        # 关键词提取与话题探索
        if not df.empty:

            try:

                # 如果分析结果不存在，则重新计算
                if (
                    not KEYWORDS_FILE.exists()
                    or not TOPICS_FILE.exists()
                ):

                    save_topic_analysis(df)

                # 读取保存好的分析结果
                keyword_data, topics = load_topic_analysis()

            except Exception as e:

                app.logger.exception(
                    "关键词与话题分析失败"
                )

                keyword_error = str(e)
                topic_error = str(e)

    except Exception as e:

        app.logger.exception("首页数据读取失败")

        return render_template(
            "index.html",
            total_count=total_count,
            platform_count=platform_count,
            comments=comments,
            sentiment_data=sentiment_data,
            sentiment_comments=sentiment_comments,
            analysis_summary=analysis_summary,
            keyword_data=keyword_data,
            topics=topics,
            keyword_error=keyword_error,
            topic_error=topic_error,
            error=str(e)
        )

    return render_template(
        "index.html",
        total_count=total_count,
        platform_count=platform_count,
        comments=comments,
        sentiment_data=sentiment_data,
        sentiment_comments=sentiment_comments,
        analysis_summary=analysis_summary,
        keyword_data=keyword_data,
        topics=topics,
        keyword_error=keyword_error,
        topic_error=topic_error,
        error=None
    )


# 8. CSV 上传路由

@app.route("/upload", methods=["GET", "POST"])
def upload():

    if request.method == "GET":
        return render_template("upload.html")

    uploaded_file = request.files.get("file")

    if uploaded_file is None or uploaded_file.filename == "":
        flash("请先选择一个 CSV 文件", "error")
        return redirect(url_for("upload"))

    if not uploaded_file.filename.lower().endswith(".csv"):
        flash("目前只支持 CSV 文件", "error")
        return redirect(url_for("upload"))

    try:

        # 读取上传文件
        file_bytes = uploaded_file.read()

        df = read_uploaded_csv(file_bytes)

        # 执行数据清洗
        cleaned_df, result = clean_comments(df)

        # 保存清洗后的数据
        cleaned_df.to_csv(
            PROCESSED_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        # 上传新数据后，清除所有旧分析结果
        result_files = [
            SENTIMENT_FILE,
            SUMMARY_FILE,
            KEYWORDS_FILE,
            TOPICS_FILE
        ]

        for result_file in result_files:

            if result_file.exists():
                result_file.unlink()

        flash(
            f"上传成功！原始数据 {result['original_count']} 条，"
            f"清洗后 {result['cleaned_count']} 条；"
            f"删除空评论 {result['empty_count']} 条，"
            f"删除重复评论 {result['duplicate_count']} 条。",
            "success"
        )

        return redirect(url_for("index"))

    except ValueError as e:

        flash(
            f"数据验证失败：{str(e)}",
            "error"
        )

        return redirect(url_for("upload"))

    except Exception:

        app.logger.exception("上传数据时发生异常")

        flash(
            "系统处理数据时发生错误，请检查文件格式后重试",
            "error"
        )

        return redirect(url_for("upload"))


# 9. 情感分析路由

@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        if not PROCESSED_FILE.exists():

            flash(
                "请先上传并清洗评论数据，再进行情感分析！",
                "error"
            )

            return redirect(url_for("index"))

        df = pd.read_csv(
            PROCESSED_FILE,
            encoding="utf-8-sig"
        )

        if df.empty:

            flash(
                "当前没有可供分析的评论！",
                "error"
            )

            return redirect(url_for("index"))

        if "content" not in df.columns:

            flash(
                "评论数据缺少 content 字段，无法进行分析！",
                "error"
            )

            return redirect(url_for("index"))

        flash(
            "正在进行情感分析，首次运行需要加载预训练模型，请耐心等待。",
            "success"
        )

        # 执行情感分析
        result_df = analyze_sentiment(df)

        # 生成情感统计
        summary_df = generate_sentiment_summary(result_df)

        # 保存详细分析结果
        result_df.to_csv(
            SENTIMENT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        # 保存情感统计结果
        summary_df.to_csv(
            SUMMARY_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        flash(
            f"情感分析完成！共分析 {len(result_df)} 条评论。",
            "success"
        )

    except Exception as e:

        app.logger.exception("情感分析失败")

        flash(
            f"情感分析失败：{str(e)}",
            "error"
        )

    return redirect(url_for("index"))


# 10. 文件过大异常处理

@app.errorhandler(413)
def file_too_large(error):

    flash(
        "文件不能超过 5 MB",
        "error"
    )

    return redirect(url_for("upload"))

@app.route("/export")
def export_report():
    """
    导出情感分析结果 CSV 文件
    """

    # 检查是否已经完成情感分析
    if not SENTIMENT_FILE.exists():
        flash("请先完成情感分析，再导出报告。", "warning")
        return redirect(url_for("index"))

    try:
        # 读取情感分析结果
        sentiment_df = pd.read_csv(
            SENTIMENT_FILE,
            encoding="utf-8-sig"
        )

        # 检查数据是否为空
        if sentiment_df.empty:
            flash("当前没有可导出的分析数据。", "warning")
            return redirect(url_for("index"))

        # 检查必要字段
        if "sentiment" not in sentiment_df.columns:
            flash("分析结果文件缺少情感分类字段，无法导出。", "danger")
            return redirect(url_for("index"))

        # 转换为 CSV 内容
        csv_content = sentiment_df.to_csv(index=False)

        # 使用 UTF-8 BOM，方便 Excel 正确识别中文
        output = io.BytesIO(
            csv_content.encode("utf-8-sig")
        )

        output.seek(0)

        # 返回下载文件
        return send_file(
            output,
            mimetype="text/csv",
            as_attachment=True,
            download_name="MediaInsight_情感分析结果.csv"
        )

    except pd.errors.EmptyDataError:
        flash("分析结果文件为空，请重新进行情感分析。", "warning")
        return redirect(url_for("index"))

    except Exception as e:
        app.logger.exception("导出情感分析报告时发生错误")
        flash("报告导出失败，请检查分析结果文件。", "danger")
        return redirect(url_for("index"))

# 11. 启动 Flask

if __name__ == "__main__":

    app.run(debug=True)
