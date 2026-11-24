
import re
import jieba
import pandas as pd

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


def load_stopwords(file_path):
    """读取停用词表"""

    with open(file_path, "r", encoding="utf-8") as file:
        stopwords = {
            line.strip()
            for line in file
            if line.strip()
        }

    return stopwords


def tokenize_text(text, stopwords):
    """
    中文分词与停用词过滤
    """

    if pd.isna(text):
        return []

    text = str(text).strip()

    # 使用 Jieba 分词
    words = jieba.cut(text)

    result = []

    for word in words:
        word = word.strip()

        # 过滤空字符
        if not word:
            continue

        # 只保留长度至少为 2 的中文词或英文单词
        if not re.fullmatch(
            r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{2,}",
            word
        ):
            continue

        # 去除停用词
        if word in stopwords:
            continue

        result.append(word)

    return result


def analyze_comments(df, stopwords):
    """
    对评论数据进行 NLP 预处理和 TF-IDF 分析
    """

    if df is None or df.empty:
        raise ValueError("没有可供分析的评论数据")

    if "content_clean" in df.columns:
        text_column = "content_clean"
    elif "content" in df.columns:
        text_column = "content"
    else:
        raise ValueError("数据中缺少 content 或 content_clean 字段")

    result_df = df.copy()

    # 1. 中文分词
    result_df["tokens"] = result_df[text_column].apply(
        lambda text: tokenize_text(text, stopwords)
    )

    # 2. 保存分词结果
    result_df["token_text"] = result_df["tokens"].apply(
        lambda words: " ".join(words)
    )

    # 3. 统计词频
    all_words = []

    for words in result_df["tokens"]:
        all_words.extend(words)

    word_frequency = Counter(all_words)

    frequency_df = pd.DataFrame(
        word_frequency.most_common(50),
        columns=["keyword", "frequency"]
    )

    # 4. TF-IDF 分析
    documents = result_df["token_text"].tolist()

    if not any(documents):
        raise ValueError(
            "分词后没有可分析的词语，请检查评论内容或停用词表"
        )

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        max_features=100
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    feature_names = vectorizer.get_feature_names_out()

    # 计算每个词在全部评论中的平均 TF-IDF 权重
    average_scores = tfidf_matrix.mean(axis=0).A1

    keyword_df = pd.DataFrame({
        "keyword": feature_names,
        "tfidf_score": average_scores
    })

    keyword_df = keyword_df.sort_values(
        by="tfidf_score",
        ascending=False
    ).reset_index(drop=True)

    # 5. 返回分析结果
    return result_df, frequency_df, keyword_df
