
from pathlib import Path
import pandas as pd

from services.nlp_preprocess import (
    load_stopwords,
    analyze_comments
)


# 获取项目根目录
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

# 输入文件
INPUT_FILE = DATA_DIR / "processed_comments.csv"

# 停用词文件
STOPWORDS_FILE = DATA_DIR / "stopwords.txt"

# 输出文件
NLP_OUTPUT_FILE = DATA_DIR / "nlp_comments.csv"
FREQUENCY_OUTPUT_FILE = DATA_DIR / "word_frequency.csv"
KEYWORD_OUTPUT_FILE = DATA_DIR / "tfidf_keywords.csv"


def main():

    print("=" * 45)
    print("中文评论 NLP 分析测试")
    print("=" * 45)

    # 1. 检查输入文件
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "找不到 processed_comments.csv，请先上传并清洗评论数据"
        )

    # 2. 读取评论数据
    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig"
    )

    print(f"读取评论数量：{len(df)}")

    # 3. 读取停用词
    stopwords = load_stopwords(STOPWORDS_FILE)

    print(f"停用词数量：{len(stopwords)}")

    # 4. 执行 NLP 分析
    result_df, frequency_df, keyword_df = analyze_comments(
        df,
        stopwords
    )

    # 5. 保存分词结果
    result_df.to_csv(
        NLP_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # 6. 保存词频统计
    frequency_df.to_csv(
        FREQUENCY_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # 7. 保存 TF-IDF 关键词
    keyword_df.to_csv(
        KEYWORD_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # 8. 输出分析结果
    print("\n前 5 条评论的分词结果：")

    for index, row in result_df.head(5).iterrows():
        print(f"\n评论 {index + 1}：{row['content']}")
        print(f"分词结果：{row['token_text']}")

    print("\n高频词 Top 10：")
    print(frequency_df.head(10).to_string(index=False))

    print("\nTF-IDF 关键词 Top 10：")
    print(keyword_df.head(10).to_string(index=False))

    print("\n分析完成！")
    print("已生成以下文件：")
    print("1. nlp_comments.csv")
    print("2. word_frequency.csv")
    print("3. tfidf_keywords.csv")


if __name__ == "__main__":
    main()
