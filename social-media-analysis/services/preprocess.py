
import pandas as pd


def clean_comments(df, start_id=1):
    """
    评论数据验证与清洗。

    参数：
        df: 当前批次的评论数据
        start_id: 自动生成 comment_id 时的起始编号

    返回：
        cleaned_df: 清洗后的 DataFrame
        result: 清洗统计信息

    保留所有原始字段，并增加：
        comment_id
        platform
        content_clean
    """

    # 1. 检查数据
    if df is None or df.empty:
        raise ValueError("CSV 文件中没有可用的数据")

    original_count = len(df)

    # 2. 检查必要字段
    if "content" not in df.columns:
        raise ValueError("CSV 文件必须包含 content 字段")

    df = df.copy()

    # 3. 清理评论文本
    df["content"] = (
        df["content"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # 4. 删除空评论
    empty_count = int((df["content"] == "").sum())

    df = df[df["content"] != ""].copy()

    if df.empty:
        result = {
            "original_count": original_count,
            "cleaned_count": 0,
            "empty_count": empty_count,
            "duplicate_count": 0,
            "removed_count": original_count
        }

        df["content_clean"] = pd.Series(dtype=str)

        return df, result

    # 5. 检查评论长度
    too_long = df["content"].str.len() > 2000

    if too_long.any():
        raise ValueError(
            f"发现 {int(too_long.sum())} 条评论超过 2000 字符"
        )

    # 6. 生成或规范化评论编号
    if "comment_id" not in df.columns:
        df.insert(
            0,
            "comment_id",
            range(start_id, start_id + len(df))
        )
    else:
        df["comment_id"] = (
            df["comment_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        if (df["comment_id"] == "").any():
            raise ValueError(
                "发现空 comment_id，请检查原始数据"
            )

        if df["comment_id"].duplicated().any():
            raise ValueError(
                "当前数据块中发现重复 comment_id"
            )

    # 7. 规范化平台字段
    if "platform" not in df.columns:
        df["platform"] = "未注明"
    else:
        df["platform"] = (
            df["platform"]
            .fillna("未注明")
            .astype(str)
            .str.strip()
        )

        df.loc[
            df["platform"] == "",
            "platform"
        ] = "未注明"

    # 8. 删除当前数据块中的重复评论
    before_dedup = len(df)

    df = df.drop_duplicates(
        subset=["content"],
        keep="first"
    ).copy()

    duplicate_count = before_dedup - len(df)

    # 9. 生成规范化文本
    df["content_clean"] = (
        df["content"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # 10. 保留所有字段
    df = df.reset_index(drop=True)

    cleaned_count = len(df)

    result = {
        "original_count": original_count,
        "cleaned_count": cleaned_count,
        "empty_count": empty_count,
        "duplicate_count": int(duplicate_count),
        "removed_count": original_count - cleaned_count
    }

    return df, result

