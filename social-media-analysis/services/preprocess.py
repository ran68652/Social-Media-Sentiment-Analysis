import pandas as pd
import re


def clean_comments(df):
    """
    评论数据验证与清洗

    1. 检查数据是否为空
    2. 检查必要字段
    3. 检查评论内容长度
    4. 删除空评论
    5. 删除重复评论
    6. 检查评论编号
    7. 规范化评论文本
    """

    # 1. 检查数据是否为空
    if df is None or df.empty:
        raise ValueError("CSV 文件中没有可用的数据")

    original_count = len(df)

    # 2. 检查必要字段
    if "content" not in df.columns:
        raise ValueError("CSV 文件必须包含 content 字段")

    df = df.copy()

    # 3. 检查评论内容
    df["content"] = df["content"].fillna("").astype(str)
    df["content"] = df["content"].str.strip()

    # 删除空评论
    empty_count = (df["content"] == "").sum()
    df = df[df["content"] != ""].copy()

    if df.empty:
        raise ValueError("所有评论内容均为空，请检查数据")

    # 4. 检查评论长度
    too_long = df["content"].str.len() > 2000

    if too_long.any():
        raise ValueError(
            f"发现 {too_long.sum()} 条评论超过 2000 字符，请检查数据"
        )

    # 5. 检查重复评论编号
    if "comment_id" not in df.columns:
        df.insert(
            0,
            "comment_id",
            range(1, len(df) + 1)
        )
    else:
        df["comment_id"] = df["comment_id"].fillna("").astype(str).str.strip()

        # 空编号自动生成
        for index in df.index:
            if df.loc[index, "comment_id"] == "":
                df.loc[index, "comment_id"] = f"AUTO-{index + 1}"

        if df["comment_id"].duplicated().any():
            raise ValueError("发现重复的 comment_id，请检查评论编号")

    # 6. 检查平台字段
    if "platform" not in df.columns:
        df["platform"] = "未注明"
    else:
        df["platform"] = (
            df["platform"]
            .fillna("未注明")
            .astype(str)
            .str.strip()
        )
        df.loc[df["platform"] == "", "platform"] = "未注明"

    # 7. 删除完全重复的评论
    before_dedup = len(df)

    df = df.drop_duplicates(
        subset=["content"],
        keep="first"
    ).copy()

    duplicate_count = before_dedup - len(df)

    # 8. 生成适合 NLP 的规范化文本
    # 保留原始 content，不覆盖用户原始评论
    df["content_clean"] = (
        df["content"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # 9. 整理字段
    df = df[
        [
            "comment_id",
            "platform",
            "content",
            "content_clean"
        ]
    ]

    df = df.reset_index(drop=True)

    # 10. 统计清洗结果
    cleaned_count = len(df)

    result = {
        "original_count": original_count,
        "cleaned_count": cleaned_count,
        "empty_count": int(empty_count),
        "duplicate_count": int(duplicate_count),
        "removed_count": original_count - cleaned_count
    }

    return df, result
