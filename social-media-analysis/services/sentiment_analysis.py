import pandas as pd

from functools import lru_cache
from transformers import pipeline


# 预训练模型名称
MODEL_NAME = (
    "lxyuan/"
    "distilbert-base-multilingual-cased-sentiments-student"
)


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


def convert_label(raw_label, classifier):
    """
    将模型输出标签转换为统一的英文和中文标签。
    """

    label = str(raw_label).strip().lower()

    # 兼容不同模型的标签名称
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

    # 先尝试直接识别
    if label in label_mapping:
        return label_mapping[label]

    # 如果模型返回 LABEL_0 这种形式，
    # 则根据模型配置文件中的 id2label 获取真实标签
    if label.startswith("label_"):
        try:
            label_id = int(label.split("_")[-1])

            model_labels = classifier.model.config.id2label

            actual_label = model_labels.get(
                label_id,
                model_labels.get(str(label_id), "")
            )

            actual_label = str(actual_label).lower()

            if actual_label in label_mapping:
                return label_mapping[actual_label]

        except (ValueError, TypeError):
            pass

    raise ValueError(
        f"无法识别模型返回的情感标签：{raw_label}"
    )


def analyze_sentiment(df):
    """
    对评论进行情感分析。

    输入：评论 DataFrame
    输出：增加情感标签和置信度的 DataFrame
    """

    if df is None or df.empty:
        raise ValueError("没有可供分析的评论数据")

    # 优先使用规范化文本
    if "content_clean" in df.columns:
        text_column = "content_clean"

    elif "content" in df.columns:
        text_column = "content"

    else:
        raise ValueError(
            "数据中缺少 content 或 content_clean 字段"
        )

    result_df = df.copy()

    # 检查评论内容
    texts = (
        result_df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    if not any(texts):
        raise ValueError("评论内容全部为空，无法进行情感分析")

    # 加载模型
    classifier = load_sentiment_model()

    # 只分析非空评论，保留原始行号
    valid_indices = [
        index
        for index, text in enumerate(texts)
        if text
    ]

    valid_texts = [
        texts[index]
        for index in valid_indices
    ]

    # 批量预测，减少重复调用
    predictions = classifier(
        valid_texts,
        batch_size=8,
        truncation=True
    )

    # 初始化结果列
    result_df["sentiment"] = "未分析"
    result_df["sentiment_en"] = "unavailable"
    result_df["confidence"] = 0.0
    result_df["model_name"] = MODEL_NAME

    # 写入预测结果
    for index, prediction in zip(valid_indices, predictions):

        english_label, chinese_label = convert_label(
            prediction["label"],
            classifier
        )

        result_df.at[index, "sentiment"] = chinese_label
        result_df.at[index, "sentiment_en"] = english_label
        result_df.at[index, "confidence"] = round(
            float(prediction["score"]),
            4
        )

    return result_df


def generate_sentiment_summary(result_df):
    """
    统计正面、负面和中性评论数量及占比。
    """

    categories = ["正面", "负面", "中性"]

    total_count = len(result_df)

    if total_count == 0:
        raise ValueError("没有可统计的评论")

    summary = []

    for category in categories:

        count = int(
            (result_df["sentiment"] == category).sum()
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
