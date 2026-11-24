
import pandas as pd

from functools import lru_cache
from transformers import pipeline


# ==============================
# 一、模型配置
# ==============================

MODEL_NAME = (
    "lxyuan/"
    "distilbert-base-multilingual-cased-sentiments-student"
)

# 每批处理的评论数量
DEFAULT_BATCH_SIZE = 32


# ==============================
# 二、加载情感分析模型
# ==============================

@lru_cache(maxsize=1)
def load_sentiment_model():
    """
    加载预训练情感分析模型。

    使用缓存，避免每次分析都重新加载模型。
    """

    classifier = pipeline(
        task="text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1
    )

    return classifier


# ==============================
# 三、统一情感标签
# ==============================

def convert_label(raw_label, classifier):
    """
    将模型输出标签转换为统一的英文和中文标签。
    """

    label = str(raw_label).strip().lower()

    label_mapping = {
        "positive": ("positive", "正面"),
        "pos": ("positive", "正面"),
        "negative": ("negative", "负面"),
        "neg": ("negative", "负面"),
        "neutral": ("neutral", "中性"),
        "积极": ("positive", "正面"),
        "正面": ("positive", "正面"),
        "消极": ("negative", "负面"),
        "负面": ("negative", "负面"),
        "中性": ("neutral", "中性")
    }

    # 直接识别标签
    if label in label_mapping:
        return label_mapping[label]

    # 处理 LABEL_0、LABEL_1 等标签
    if label.startswith("label_"):
        try:
            label_id = int(label.split("_")[-1])

            model_labels = classifier.model.config.id2label

            actual_label = model_labels.get(
                label_id,
                model_labels.get(str(label_id), "")
            )

            actual_label = str(actual_label).strip().lower()

            if actual_label in label_mapping:
                return label_mapping[actual_label]

        except (ValueError, TypeError, AttributeError):
            pass

    raise ValueError(
        f"无法识别模型返回的情感标签：{raw_label}"
    )


# ==============================
# 四、批量情感分析
# ==============================

def analyze_sentiment(df, batch_size=DEFAULT_BATCH_SIZE):
    """
    对评论进行分批情感分析。

    参数：
        df: 评论 DataFrame
        batch_size: 每批预测的评论数量

    返回：
        增加情感标签和置信度的 DataFrame
    """

    if df is None or df.empty:
        raise ValueError("没有可供分析的评论数据")

    if batch_size < 1:
        raise ValueError("batch_size 必须大于 0")

    # 优先使用规范化文本
    if "content_clean" in df.columns:
        text_column = "content_clean"

    elif "content" in df.columns:
        text_column = "content"

    else:
        raise ValueError(
            "数据中缺少 content 或 content_clean 字段"
        )

    # 复制原始数据，保留原有字段
    result_df = df.copy().reset_index(drop=True)

    # 处理评论内容
    texts = (
        result_df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    # 找出有效评论
    valid_indices = [
        index
        for index, text in enumerate(texts)
        if text
    ]

    if not valid_indices:
        raise ValueError(
            "评论内容全部为空，无法进行情感分析"
        )

    valid_texts = [
        texts[index]
        for index in valid_indices
    ]

    # 加载模型
    classifier = load_sentiment_model()

    # 初始化结果列
    result_df["sentiment"] = "未分析"
    result_df["sentiment_en"] = "unavailable"
    result_df["confidence"] = 0.0
    result_df["model_name"] = MODEL_NAME

    # 分批预测
    total_count = len(valid_texts)

    for start in range(0, total_count, batch_size):

        end = min(start + batch_size, total_count)

        batch_texts = valid_texts[start:end]

        predictions = classifier(
            batch_texts,
            batch_size=batch_size,
            truncation=True
        )

        # 保存当前批次结果
        for prediction_index, prediction in enumerate(predictions):

            original_index = valid_indices[start + prediction_index]

            english_label, chinese_label = convert_label(
                prediction["label"],
                classifier
            )

            result_df.at[
                original_index, "sentiment"
            ] = chinese_label

            result_df.at[
                original_index, "sentiment_en"
            ] = english_label

            result_df.at[
                original_index, "confidence"
            ] = round(
                float(prediction["score"]),
                4
            )

    return result_df


# ==============================
# 五、情感分析统计
# ==============================

def generate_sentiment_summary(result_df):
    """
    统计正面、负面和中性评论数量及占比。
    """

    categories = ["正面", "负面", "中性"]

    if result_df is None or result_df.empty:
        raise ValueError("没有可统计的评论")

    if "sentiment" not in result_df.columns:
        raise ValueError("数据中缺少 sentiment 字段")

    # 只统计成功完成情感分类的评论
    analyzed_df = result_df[
        result_df["sentiment"].isin(categories)
    ]

    total_count = len(analyzed_df)

    if total_count == 0:
        raise ValueError("没有已完成情感分析的评论")

    summary = []

    for category in categories:

        count = int(
            (analyzed_df["sentiment"] == category).sum()
        )

        percentage = round(
            count / total_count * 100,
            2
        )

        summary.append({
            "sentiment": category,
            "count": count,
            "percentage": percentage
        })

    return pd.DataFrame(summary)

