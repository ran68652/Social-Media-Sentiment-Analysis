
import pandas as pd


def generate_analysis_summary(df):
    """
    根据情感分析结果生成文字摘要。

    参数：
        df: 包含 sentiment 字段的 DataFrame

    返回：
        包含统计数据和文字摘要的字典
    """

    if df is None or df.empty:
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

    if "sentiment" not in df.columns:
        raise ValueError("数据缺少 sentiment 字段，无法生成分析摘要。")

    # 统一中英文情感标签
    sentiment_mapping = {
        "正面": "positive",
        "负面": "negative",
        "中性": "neutral",
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral"
    }

    sentiment_values = (
        df["sentiment"]
        .astype(str)
        .str.strip()
        .map(sentiment_mapping)
    )

    # 只统计能够识别的情感标签
    sentiment_values = sentiment_values.dropna()

    total = len(sentiment_values)

    if total == 0:
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
            "summary_text": "没有识别到有效的情感分析结果。"
        }

    # 统计三种情感的数量
    counts = sentiment_values.value_counts()

    positive_count = int(counts.get("positive", 0))
    negative_count = int(counts.get("negative", 0))
    neutral_count = int(counts.get("neutral", 0))

    # 计算比例
    positive_percentage = round(
        positive_count / total * 100, 2
    )

    negative_percentage = round(
        negative_count / total * 100, 2
    )

    neutral_percentage = round(
        neutral_count / total * 100, 2
    )

    # 判断占比最高的情感类别
    sentiment_names = {
        "positive": "正面",
        "negative": "负面",
        "neutral": "中性"
    }

    all_counts = {
        "positive": positive_count,
        "negative": negative_count,
        "neutral": neutral_count
    }

    dominant_key = max(
        all_counts,
        key=all_counts.get
    )

    dominant_sentiment = sentiment_names[dominant_key]

    dominant_percentage = round(
        all_counts[dominant_key] / total * 100, 2
    )

    # 生成文字摘要
    summary_text = (
        f"本次共分析 {total} 条评论。"
        f"模型识别出正面评论 {positive_count} 条，"
        f"占 {positive_percentage}%；"
        f"负面评论 {negative_count} 条，"
        f"占 {negative_percentage}%；"
        f"中性评论 {neutral_count} 条，"
        f"占 {neutral_percentage}%。"
    )

    if positive_count > negative_count:
        summary_text += (
            f"正面评论数量最多，占比为 {positive_percentage}%，"
            "模型识别结果整体偏向正面。"
        )

    elif negative_count > positive_count:
        summary_text += (
            f"负面评论数量最多，占比为 {negative_percentage}%，"
            "模型识别结果整体偏向负面。"
        )

    else:
        summary_text += (
            "正面与负面评论数量相同，"
            "需要结合具体评论进一步分析。"
        )

    return {
        "has_data": True,
        "total": total,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "positive_percentage": positive_percentage,
        "negative_percentage": negative_percentage,
        "neutral_percentage": neutral_percentage,
        "dominant_sentiment": dominant_sentiment,
        "dominant_percentage": dominant_percentage,
        "summary_text": summary_text
    }
