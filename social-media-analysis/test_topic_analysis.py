
from pathlib import Path

import pandas as pd

from services.topic_analysis import (
    extract_keywords,
    discover_topics
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "processed_comments.csv"


def test_keyword_extraction(df):
    """测试关键词提取功能"""

    print("\n[测试 1] 关键词提取")

    keyword_df = extract_keywords(
        df,
        top_n=10
    )

    assert not keyword_df.empty, "关键词结果不能为空"

    assert "keyword" in keyword_df.columns
    assert "score" in keyword_df.columns

    assert len(keyword_df) <= 10

    assert keyword_df["score"].is_monotonic_decreasing, (
        "关键词没有按照权重降序排列"
    )

    assert keyword_df["keyword"].notna().all()
    assert keyword_df["score"].notna().all()

    print("通过：成功提取关键词")
    print(f"关键词数量：{len(keyword_df)}")

    print(keyword_df.to_string(index=False))

    output_file = DATA_DIR / "keywords.csv"

    keyword_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"结果已保存：{output_file}")

    return keyword_df


def test_topic_discovery(df):
    """测试 LDA 话题探索功能"""

    print("\n[测试 2] LDA 话题探索")

    topics = discover_topics(
        df,
        n_topics=3,
        n_words=5
    )

    assert len(topics) > 0, "没有生成任何话题"

    assert len(topics) <= 3

    for topic in topics:

        assert "topic_id" in topic
        assert "keywords" in topic

        assert len(topic["keywords"]) == 5

        print(
            f"\n话题 {topic['topic_id']}："
            + "、".join(topic["keywords"])
        )

    # 保存话题结果
    topic_rows = []

    for topic in topics:
        topic_rows.append({
            "topic_id": topic["topic_id"],
            "keywords": "、".join(topic["keywords"])
        })

    topic_df = pd.DataFrame(topic_rows)

    output_file = DATA_DIR / "topics.csv"

    topic_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n通过：LDA 话题探索成功")
    print(f"结果已保存：{output_file}")

    return topics


def test_invalid_input():
    """测试异常输入处理"""

    print("\n[测试 3] 异常输入处理")

    # 空数据测试
    empty_df = pd.DataFrame()

    try:
        extract_keywords(empty_df)
        raise AssertionError("空数据没有触发异常")

    except ValueError:
        print("通过：关键词模块正确处理空数据")

    try:
        discover_topics(empty_df)
        raise AssertionError("空数据没有触发异常")

    except ValueError:
        print("通过：话题模块正确处理空数据")

    # 缺少 content 字段测试
    invalid_df = pd.DataFrame({
        "comment_id": [1, 2],
        "platform": ["平台A", "平台B"]
    })

    try:
        extract_keywords(invalid_df)
        raise AssertionError("缺少字段时没有触发异常")

    except ValueError:
        print("通过：关键词模块正确检查字段")

    try:
        discover_topics(invalid_df)
        raise AssertionError("缺少字段时没有触发异常")

    except ValueError:
        print("通过：话题模块正确检查字段")


def main():

    print("=" * 55)
    print("社交媒体受众分析系统 - Day 7 自动化测试")
    print("=" * 55)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "找不到 processed_comments.csv，请先上传评论数据"
        )

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig"
    )

    print(f"\n读取评论数量：{len(df)}")

    if df.empty:
        raise ValueError("当前评论数据为空，无法测试")

    test_keyword_extraction(df)

    test_topic_discovery(df)

    test_invalid_input()

    print("\n" + "=" * 55)
    print("Day 7 全部测试通过！")
    print("=" * 55)


if __name__ == "__main__":
    main()
