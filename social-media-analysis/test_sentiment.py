from pathlib import Path
import pandas as pd

from services.sentiment_analysis import (
    analyze_sentiment,
    generate_sentiment_summary
)


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

# 输入文件
INPUT_FILE = DATA_DIR / "processed_comments.csv"

# 输出文件
RESULT_FILE = DATA_DIR / "sentiment_comments.csv"

SUMMARY_FILE = DATA_DIR / "sentiment_summary.csv"


def main():

    print("=" * 50)
    print("中文评论情感分析系统")
    print("=" * 50)

    # 1. 检查文件
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "找不到 processed_comments.csv，请先上传评论数据"
        )

    # 2. 读取评论数据
    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig"
    )

    print(f"\n读取评论数量：{len(df)}")

    # 3. 执行情感分析
    print("\n正在加载情感分析模型，请稍候...")

    result_df = analyze_sentiment(df)

    # 4. 生成统计结果
    summary_df = generate_sentiment_summary(result_df)

    # 5. 保存详细结果
    result_df.to_csv(
        RESULT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # 6. 保存统计结果
    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # 7. 展示预测结果
    print("\n评论情感预测结果:")

    for _, row in result_df.iterrows():

        print(f"\n评论:{row['content']}")
        print(f"情感：{row['sentiment']}")
        print(f"置信度：{row['confidence']}")

    # 8. 展示统计结果
    print("\n情感分析统计:")

    print(
        summary_df.to_string(index=False)
    )

    print("\n分析完成!")
    print(f"详细结果：{RESULT_FILE}")
    print(f"统计结果：{SUMMARY_FILE}")


if __name__ == "__main__":
    main()
