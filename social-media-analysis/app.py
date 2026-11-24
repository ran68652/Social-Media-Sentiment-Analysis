import codecs
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
import json
import sqlite3
import hashlib
import os
import tempfile

from pathlib import Path

from werkzeug.utils import secure_filename

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


# =====================================================
# 1. Flask 基础配置
# =====================================================

app = Flask(__name__)

app.secret_key = "change-this-to-a-random-secret"

# 最大上传文件：200 MB
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# 分块读取数量
CHUNK_SIZE = 5000

# 情感分析每批处理数量
SENTIMENT_BATCH_SIZE = 256

# 首页话题分析最大数据量
TOPIC_SAMPLE_SIZE = 10000

# 默认情感分析数量
DEFAULT_ANALYSIS_SIZE = 10000

# 允许用户选择的分析数量
ALLOWED_ANALYSIS_SIZES = [
    1000,
    5000,
    10000,
    50000
]


# =====================================================
# 2. 文件路径配置
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"

PROCESSED_DIR = DATA_DIR / "processed"

SAMPLE_FILE = DATA_DIR / "sample_comments.csv"

RAW_FILE = RAW_DIR / "uploaded_original.csv"

PROCESSED_FILE = DATA_DIR / "processed_comments.csv"

SENTIMENT_FILE = DATA_DIR / "sentiment_comments.csv"

SUMMARY_FILE = DATA_DIR / "sentiment_summary.csv"

KEYWORDS_FILE = DATA_DIR / "keywords.csv"

TOPICS_FILE = DATA_DIR / "topics.csv"

STATS_FILE = DATA_DIR / "dataset_stats.json"


# 自动创建目录
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 3. 评论数据读取函数
# =====================================================

def load_comments(nrows=10000):
    """
    读取有限数量的评论。

    首页只读取部分数据，
    避免一次性加载百万条评论。
    """

    if PROCESSED_FILE.exists():

        data_file = PROCESSED_FILE

    else:

        data_file = SAMPLE_FILE

    if not data_file.exists():

        raise FileNotFoundError(
            "找不到评论数据文件，请先上传评论数据"
        )

    df = pd.read_csv(
        data_file,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        nrows=nrows
    )

    if "content" not in df.columns:

        raise ValueError(
            "评论数据缺少 content 字段"
        )

    if "comment_id" not in df.columns:

        df.insert(
            0,
            "comment_id",
            range(1, len(df) + 1)
        )

    if "platform" not in df.columns:

        df["platform"] = "未知平台"

    return df


# =====================================================
# 4. CSV 编码检测
# =====================================================

def detect_csv_encoding(file_path):
    """
    自动检测 CSV 文件编码。

    支持：
    UTF-8 BOM
    UTF-8
    GB18030
    GBK
    UTF-16
    UTF-16 LE
    UTF-16 BE
    """

    with open(file_path, "rb") as f:
        sample = f.read(1024 * 1024)

    if not sample:
        raise ValueError("CSV 文件为空")

    # 1. 优先识别 BOM
    if sample.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"

    if sample.startswith(codecs.BOM_UTF16_LE):
        return "utf-16"

    if sample.startswith(codecs.BOM_UTF16_BE):
        return "utf-16"

    # 2. 检测无 BOM 的 UTF-16
    if len(sample) >= 4:
        even_nulls = sample[0::2].count(0)
        odd_nulls = sample[1::2].count(0)

        if odd_nulls > len(sample) * 0.1:
            return "utf-16-le"

        if even_nulls > len(sample) * 0.1:
            return "utf-16-be"

    # 3. 依次尝试常见编码
    encodings = [
        "utf-8",
        "gb18030",
        "gbk",
        "utf-16-le",
        "utf-16-be"
    ]

    for encoding in encodings:
        try:
            decoder = codecs.getincrementaldecoder(encoding)(
                errors="strict"
            )

            # final=False，允许检测样本刚好截断多字节字符的情况
            decoder.decode(sample, final=False)

            return encoding

        except (UnicodeDecodeError, UnicodeError):
            continue

    raise ValueError(
        "无法识别 CSV 文件编码。"
        "请确认文件不是损坏文件，并尝试另存为 UTF-8 编码的 CSV。"
    )

# =====================================================
# 5. 大文件分块读取和清洗
# =====================================================

def process_large_csv(file_path):
    """
    大规模 CSV 数据处理。

    1. 分块读取 CSV
    2. 每批清洗 5000 条
    3. 删除空评论
    4. 删除重复评论
    5. 使用 SQLite 进行跨批次去重
    6. 保留原始字段
    7. 保存清洗后的 CSV
    8. 统计数据集信息

    注意：
    处理成功后才替换原有清洗文件。
    """

    encoding = detect_csv_encoding(file_path)

    # 创建临时输出文件
    temp_output = None

    # 创建临时 SQLite 文件
    db_fd, db_name = tempfile.mkstemp(
        suffix=".db",
        dir=str(DATA_DIR)
    )

    os.close(db_fd)

    sqlite_file = Path(db_name)

    original_count = 0

    cleaned_count = 0

    empty_count = 0

    duplicate_count = 0

    # 统计不同平台
    platforms = set()

    first_chunk = True
    conn = None
    try:

        # 创建临时 CSV
        output_fd, output_name = tempfile.mkstemp(
            suffix=".csv",
            dir=str(PROCESSED_DIR)
        )

        os.close(output_fd)

        temp_output = Path(output_name)

        # 建立 SQLite 数据库
        conn = sqlite3.connect(
            str(sqlite_file)
        )

        conn.execute("""
            CREATE TABLE seen_content (
                content_hash TEXT PRIMARY KEY
            )
        """)

        conn.execute("""
            CREATE TABLE seen_ids (
                comment_id TEXT PRIMARY KEY
            )
        """)

        # 分块读取 CSV
        chunks = pd.read_csv(
            file_path,
            encoding=encoding,
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
            on_bad_lines="error"
        )

        for chunk in chunks:

            original_count += len(chunk)

            # 检查必要字段
            if "content" not in chunk.columns:

                raise ValueError(
                    "CSV 文件必须包含 content 字段"
                )

            # 自动生成评论编号
            if "comment_id" not in chunk.columns:

                chunk.insert(
                    0,
                    "comment_id",
                    range(
                        original_count - len(chunk) + 1,
                        original_count + 1
                    )
                )

            # 自动补充平台
            if "platform" not in chunk.columns:

                chunk["platform"] = "Weibo"

            # 执行当前数据块清洗
            cleaned_df, result = clean_comments(
                chunk,
                start_id=original_count - len(chunk) + 1
            )

            empty_count += result["empty_count"]

            duplicate_count += result["duplicate_count"]

            # 统计平台
            if "platform" in cleaned_df.columns:

                platforms.update(
                    cleaned_df["platform"]
                    .dropna()
                    .astype(str)
                    .unique()
                )

            if cleaned_df.empty:

                continue

            # -----------------------------------------
            # 5.1 检查跨数据块重复编号
            # -----------------------------------------

            ids = (
                cleaned_df["comment_id"]
                .astype(str)
                .tolist()
            )

            # SQLite 分批检查，避免 SQL 参数过多
            for start in range(0, len(ids), 500):

                id_batch = ids[start:start + 500]

                placeholders = ",".join(
                    ["?"] * len(id_batch)
                )

                query = (
                    "SELECT comment_id FROM seen_ids "
                    f"WHERE comment_id IN ({placeholders})"
                )

                existing_ids = {
                    row[0]
                    for row in conn.execute(
                        query,
                        id_batch
                    ).fetchall()
                }

                if existing_ids:

                    raise ValueError(
                        "发现跨数据块重复 comment_id："
                        + ", ".join(list(existing_ids)[:5])
                    )

                conn.executemany(
                    """
                    INSERT INTO seen_ids (comment_id)
                    VALUES (?)
                    """,
                    [
                        (comment_id,)
                        for comment_id in id_batch
                    ]
                )

            # -----------------------------------------
            # 5.2 计算评论哈希
            # -----------------------------------------

            hashes = cleaned_df["content"].apply(
                lambda x: hashlib.sha256(
                    x.encode("utf-8")
                ).hexdigest()
            )

            keep_hashes = set()

            # 分批检查全局重复评论
            for start in range(0, len(hashes), 500):

                hash_batch = hashes.iloc[
                    start:start + 500
                ].tolist()

                placeholders = ",".join(
                    ["?"] * len(hash_batch)
                )

                query = (
                    "SELECT content_hash FROM seen_content "
                    f"WHERE content_hash IN ({placeholders})"
                )

                existing_hashes = {
                    row[0]
                    for row in conn.execute(
                        query,
                        hash_batch
                    ).fetchall()
                }

                # 当前批次中尚未出现的评论
                new_hashes = [
                    h
                    for h in hash_batch
                    if h not in existing_hashes
                ]

                if new_hashes:

                    conn.executemany(
                        """
                        INSERT INTO seen_content
                        (content_hash)
                        VALUES (?)
                        """,
                        [
                            (h,)
                            for h in new_hashes
                        ]
                    )

                    keep_hashes.update(new_hashes)

            # -----------------------------------------
            # 5.3 保留非重复评论
            # -----------------------------------------

            keep_mask = hashes.isin(keep_hashes)

            filtered_df = cleaned_df.loc[
                keep_mask
            ].copy()

            cross_duplicate_count = (
                len(cleaned_df) - len(filtered_df)
            )

            duplicate_count += cross_duplicate_count

            if filtered_df.empty:

                conn.commit()

                continue

            # -----------------------------------------
            # 5.4 分块追加保存
            # -----------------------------------------

            filtered_df.to_csv(
                temp_output,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
                encoding=(
                    "utf-8-sig"
                    if first_chunk
                    else "utf-8"
                )
            )

            first_chunk = False

            cleaned_count += len(filtered_df)

            conn.commit()

            # 控制台显示处理进度
            app.logger.info(
                "已读取 %s 条，清洗后保留 %s 条",
                original_count,
                cleaned_count
            )


        # 检查最终结果
        if cleaned_count == 0:

            raise ValueError(
                "清洗后没有可用评论数据"
            )

        # 原子替换清洗后的文件
        os.replace(
            temp_output,
            PROCESSED_FILE
        )

        result = {
            "original_count": original_count,
            "cleaned_count": cleaned_count,
            "empty_count": empty_count,
            "duplicate_count": duplicate_count,
            "removed_count": original_count - cleaned_count,
            "platform_count": len(platforms)
        }

        return result

    except pd.errors.EmptyDataError:

        raise ValueError(
            "CSV 文件没有有效内容"
        )

    except pd.errors.ParserError:

        raise ValueError(
            "CSV 文件格式错误，请检查数据分隔符和列结构"
        )

    finally:
        # 确保 SQLite 数据库连接已经关闭
        if conn is not None:
            conn.close()
            conn = None

        # 清理临时 CSV
        if temp_output is not None and temp_output.exists():
            temp_output.unlink()

        # 清理临时 SQLite 数据库
        if sqlite_file.exists():
            sqlite_file.unlink()


# =====================================================
# 6. 随机抽样函数
# =====================================================

def load_random_sample(file_path, sample_size):
    """
    分块读取整个数据集，
    使用蓄水池抽样思想获得随机样本。

    不需要把百万条评论一次性放入内存。
    """

    encoding = detect_csv_encoding(file_path)

    reservoir = pd.DataFrame()

    chunks = pd.read_csv(
        file_path,
        encoding=encoding,
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
        on_bad_lines="error"
    )

    for batch_number, chunk in enumerate(chunks):

        if reservoir.empty:

            combined = chunk

        else:

            combined = pd.concat(
                [reservoir, chunk],
                ignore_index=True
            )

        if len(combined) > sample_size:

            reservoir = combined.sample(
                n=sample_size,
                random_state=42 + batch_number
            ).reset_index(drop=True)

        else:

            reservoir = combined.reset_index(drop=True)

    return reservoir


# =====================================================
# 7. 关键词和话题分析结果保存
# =====================================================

def save_topic_analysis(df):
    """
    执行关键词提取和话题探索。
    """

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

    topic_rows = []

    for topic in topics:

        topic_rows.append({
            "topic_id": topic["topic_id"],
            "keywords": "、".join(
                topic["keywords"]
            )
        })

    topic_df = pd.DataFrame(
        topic_rows,
        columns=[
            "topic_id",
            "keywords"
        ]
    )

    topic_df.to_csv(
        TOPICS_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# =====================================================
# 8. 关键词和话题结果读取
# =====================================================

def load_topic_analysis():
    """
    读取已保存的关键词和话题结果。
    """

    keyword_df = pd.read_csv(
        KEYWORDS_FILE,
        encoding="utf-8-sig"
    )

    topic_df = pd.read_csv(
        TOPICS_FILE,
        encoding="utf-8-sig"
    )

    keyword_data = (
        keyword_df
        .fillna(0)
        .to_dict(orient="records")
    )

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


# =====================================================
# 9. 默认情感分析报告
# =====================================================

def get_empty_analysis_summary():
    """
    生成默认的空情感分析报告。
    """

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


# =====================================================
# 10. 首页路由
# =====================================================

@app.route("/")
def index():

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

    analysis_summary = get_empty_analysis_summary()

    keyword_data = []

    topics = []

    keyword_error = None

    topic_error = None

    total_count = 0

    platform_count = 0

    comments = []

    dataset_stats = {}

    try:

        # -----------------------------------------
        # 10.1 读取有限数量的评论
        # -----------------------------------------

        df = load_comments(
            nrows=TOPIC_SAMPLE_SIZE
        )

        # 默认使用当前预览数据
        total_count = len(df)

        platform_count = (
            df["platform"].nunique()
            if "platform" in df.columns
            else 0
        )

        # -----------------------------------------
        # 10.2 读取完整数据集统计信息
        # -----------------------------------------

        if (
            PROCESSED_FILE.exists()
            and STATS_FILE.exists()
        ):

            with open(
                STATS_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                dataset_stats = json.load(f)

            total_count = dataset_stats.get(
                "cleaned_count",
                total_count
            )

            platform_count = dataset_stats.get(
                "platform_count",
                platform_count
            )

        # -----------------------------------------
        # 10.3 首页评论预览
        # -----------------------------------------

        comments = (
            df.head(10)
            .fillna("")
            .to_dict(orient="records")
        )

        # -----------------------------------------
        # 10.4 情感分析详细结果
        # -----------------------------------------

        if SENTIMENT_FILE.exists():

            sentiment_df = pd.read_csv(
                SENTIMENT_FILE,
                encoding="utf-8-sig"
            )

            sentiment_comments = (
                sentiment_df.head(500)
                .fillna("")
                .to_dict(orient="records")
            )

            try:

                analysis_summary = generate_analysis_summary(
                    sentiment_df
                )

            except Exception:

                app.logger.exception(
                    "情感分析报告生成失败"
                )

                analysis_summary = get_empty_analysis_summary()

        # -----------------------------------------
        # 10.5 情感统计结果
        # -----------------------------------------

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

        # -----------------------------------------
        # 10.6 关键词和话题分析
        # -----------------------------------------

        if not df.empty:

            try:

                if (
                    not KEYWORDS_FILE.exists()
                    or not TOPICS_FILE.exists()
                ):

                    save_topic_analysis(df)

                keyword_data, topics = load_topic_analysis()

            except Exception as e:

                app.logger.exception(
                    "关键词与话题分析失败"
                )

                keyword_error = str(e)

                topic_error = str(e)

    except Exception as e:

        app.logger.exception(
            "首页数据读取失败"
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


# =====================================================
# 11. CSV 上传路由
# =====================================================

@app.route("/upload", methods=["GET", "POST"])
def upload():

    if request.method == "GET":

        return render_template(
            "upload.html"
        )

    uploaded_file = request.files.get("file")

    if (
        uploaded_file is None
        or uploaded_file.filename == ""
    ):

        flash(
            "请先选择一个 CSV 文件",
            "error"
        )

        return redirect(url_for("upload"))

    if not uploaded_file.filename.lower().endswith(".csv"):

        flash(
            "目前只支持 CSV 文件",
            "error"
        )

        return redirect(url_for("upload"))

    filename = secure_filename(
        uploaded_file.filename
    )

    if not filename:

        flash(
            "文件名无效",
            "error"
        )

        return redirect(url_for("upload"))

    temp_file = None

    try:

        # -----------------------------------------
        # 11.1 保存上传文件到临时位置
        # -----------------------------------------

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".csv",
            dir=str(RAW_DIR),
            delete=False
        ) as temp:

            temp_file = Path(temp.name)

            uploaded_file.save(temp)

        if temp_file.stat().st_size == 0:

            raise ValueError(
                "上传的文件为空"
            )

        # -----------------------------------------
        # 11.2 分块清洗数据
        # -----------------------------------------

        result = process_large_csv(
            temp_file
        )

        # -----------------------------------------
        # 11.3 保存原始文件
        # -----------------------------------------

        os.replace(
            temp_file,
            RAW_FILE
        )

        temp_file = None

        # -----------------------------------------
        # 11.4 保存数据集统计信息
        # -----------------------------------------

        stats_fd, stats_name = tempfile.mkstemp(
            suffix=".json",
            dir=str(DATA_DIR)
        )

        os.close(stats_fd)

        stats_temp = Path(stats_name)

        try:

            with open(
                stats_temp,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    result,
                    f,
                    ensure_ascii=False,
                    indent=4
                )

            os.replace(
                stats_temp,
                STATS_FILE
            )

        finally:

            if stats_temp.exists():

                stats_temp.unlink()

        # -----------------------------------------
        # 11.5 清除旧分析结果
        # -----------------------------------------

        result_files = [
            SENTIMENT_FILE,
            SUMMARY_FILE,
            KEYWORDS_FILE,
            TOPICS_FILE
        ]

        for result_file in result_files:

            if result_file.exists():

                result_file.unlink()

        # -----------------------------------------
        # 11.6 上传成功提示
        # -----------------------------------------

        flash(
            f"数据处理完成！"
            f"原始数据 {result['original_count']} 条，"
            f"清洗后 {result['cleaned_count']} 条；"
            f"删除空评论 {result['empty_count']} 条，"
            f"删除重复评论 {result['duplicate_count']} 条。",
            "success"
        )

        return redirect(
            url_for("index")
        )

    except ValueError as e:

        flash(
            f"数据验证失败：{str(e)}",
            "error"
        )

        return redirect(
            url_for("upload")
        )

    except Exception:

        app.logger.exception(
            "大文件上传和处理失败"
        )

        flash(
            "文件处理失败，请检查 CSV 格式和数据内容。",
            "error"
        )

        return redirect(
            url_for("upload")
        )

    finally:

        if (
            temp_file is not None
            and temp_file.exists()
        ):

            temp_file.unlink()


# =====================================================
# 12. 情感分析路由
# =====================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    try:

        if not PROCESSED_FILE.exists():

            flash(
                "请先上传并清洗评论数据，再进行情感分析！",
                "error"
            )

            return redirect(
                url_for("index")
            )

        # -----------------------------------------
        # 12.1 获取用户选择的分析数量
        # -----------------------------------------

        try:

            sample_size = int(
                request.form.get(
                    "sample_size",
                    DEFAULT_ANALYSIS_SIZE
                )
            )

        except (ValueError, TypeError):

            sample_size = DEFAULT_ANALYSIS_SIZE

        if sample_size not in ALLOWED_ANALYSIS_SIZES:

            sample_size = DEFAULT_ANALYSIS_SIZE

        # -----------------------------------------
        # 12.2 随机抽取指定数量
        # -----------------------------------------

        df = load_random_sample(
            PROCESSED_FILE,
            sample_size
        )

        if df.empty:

            flash(
                "当前没有可供分析的评论！",
                "error"
            )

            return redirect(
                url_for("index")
            )

        if "content" not in df.columns:

            flash(
                "评论数据缺少 content 字段！",
                "error"
            )

            return redirect(
                url_for("index")
            )

        # -----------------------------------------
        # 12.3 分批执行情感分析
        # -----------------------------------------

        result_parts = []

        for start in range(
            0,
            len(df),
            SENTIMENT_BATCH_SIZE
        ):

            batch = df.iloc[
                start:start + SENTIMENT_BATCH_SIZE
            ].copy()

            batch_result = analyze_sentiment(
                batch
            )

            result_parts.append(
                batch_result
            )

            app.logger.info(
                "情感分析进度：%s / %s",
                min(
                    start + SENTIMENT_BATCH_SIZE,
                    len(df)
                ),
                len(df)
            )

        result_df = pd.concat(
            result_parts,
            ignore_index=True
        )

        # -----------------------------------------
        # 12.4 生成情感统计
        # -----------------------------------------

        summary_df = generate_sentiment_summary(
            result_df
        )

        # -----------------------------------------
        # 12.5 保存详细分析结果
        # -----------------------------------------

        result_df.to_csv(
            SENTIMENT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        # -----------------------------------------
        # 12.6 保存情感统计结果
        # -----------------------------------------

        summary_df.to_csv(
            SUMMARY_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        flash(
            f"情感分析完成！"
            f"本次随机抽取并分析 {len(result_df)} 条评论。",
            "success"
        )

    except Exception as e:

        app.logger.exception(
            "情感分析失败"
        )

        flash(
            f"情感分析失败：{str(e)}",
            "error"
        )

    return redirect(
        url_for("index")
    )


# =====================================================
# 13. 文件过大异常处理
# =====================================================

@app.errorhandler(413)
def file_too_large(error):

    flash(
        "文件不能超过 200 MB，请压缩或拆分 CSV 文件。",
        "error"
    )

    return redirect(
        url_for("upload")
    )


# =====================================================
# 14. CSV 导出路由
# =====================================================

@app.route("/export")
def export_report():
    """
    导出情感分析结果 CSV 文件。
    """

    if not SENTIMENT_FILE.exists():

        flash(
            "请先完成情感分析，再导出报告。",
            "warning"
        )

        return redirect(
            url_for("index")
        )

    try:

        sentiment_df = pd.read_csv(
            SENTIMENT_FILE,
            encoding="utf-8-sig"
        )

        if sentiment_df.empty:

            flash(
                "当前没有可导出的分析数据。",
                "warning"
            )

            return redirect(
                url_for("index")
            )

        if "sentiment" not in sentiment_df.columns:

            flash(
                "分析结果文件缺少情感分类字段，无法导出。",
                "danger"
            )

            return redirect(
                url_for("index")
            )

        # 转换 CSV 内容
        csv_content = sentiment_df.to_csv(
            index=False
        )

        output = io.BytesIO(
            csv_content.encode("utf-8-sig")
        )

        output.seek(0)

        return send_file(
            output,
            mimetype="text/csv",
            as_attachment=True,
            download_name="MediaInsight_情感分析结果.csv"
        )

    except pd.errors.EmptyDataError:

        flash(
            "分析结果文件为空，请重新进行情感分析。",
            "warning"
        )

        return redirect(
            url_for("index")
        )

    except Exception:

        app.logger.exception(
            "导出情感分析报告时发生错误"
        )

        flash(
            "报告导出失败，请检查分析结果文件。",
            "danger"
        )

        return redirect(
            url_for("index")
        )


# =====================================================
# 15. Flask 启动
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
