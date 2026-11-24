
import re

import jieba
import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)
from sklearn.decomposition import LatentDirichletAllocation


# 中文停用词
STOP_WORDS = {
    "的", "了", "是", "我", "你", "他", "她", "它",
    "我们", "你们", "他们", "这", "那", "一个",
    "也", "都", "就", "而", "和", "与", "在",
    "有", "没有", "很", "非常", "真的", "还是",
    "但是", "因为", "所以", "如果", "什么", "怎么",
    "可以", "觉得", "感觉", "比较", "这个", "那个",
    "就是", "不是", "还是", "自己", "已经"
}


def clean_text(text):
    """
    清理文本中的特殊字符。
    """

    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize_text(text):
    """
    使用 Jieba 对中文文本进行分词，
    并过滤停用词和过短词语。
    """

    text = clean_text(text)

    words = jieba.lcut(text)

    filtered_words = [
        word.strip()
        for word in words
        if word.strip()
        and word.strip() not in STOP_WORDS
        and len(word.strip()) > 1
    ]

    return filtered_words


def extract_keywords(df, top_n=10):
    """
    使用 TF-IDF 提取评论集合中的关键词。

    参数：
        df: 评论 DataFrame
        top_n: 返回关键词数量

    返回：
        包含关键词和 TF-IDF 权重的 DataFrame
    """

    if df is None or df.empty:
        raise ValueError("没有可供分析的评论数据")

    if "content" not in df.columns:
        raise ValueError("数据中缺少 content 字段")

    texts = df["content"].fillna("").astype(str).tolist()

    tokenized_texts = [
        " ".join(tokenize_text(text))
        for text in texts
    ]

    valid_texts = [
        text for text in tokenized_texts if text.strip()
    ]

    if not valid_texts:
        raise ValueError("没有有效的文本内容，无法提取关键词")

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        max_features=1000
    )

    tfidf_matrix = vectorizer.fit_transform(valid_texts)

    words = vectorizer.get_feature_names_out()

    scores = tfidf_matrix.mean(axis=0).A1

    keyword_df = pd.DataFrame({
        "keyword": words,
        "score": scores
    })

    keyword_df = keyword_df.sort_values(
        by="score",
        ascending=False
    )

    return keyword_df.head(top_n).reset_index(drop=True)


def discover_topics(df, n_topics=3, n_words=5):
    """
    使用 LDA 模型进行初步话题探索。

    参数：
        df: 评论 DataFrame
        n_topics: 需要探索的话题数量
        n_words: 每个话题展示的关键词数量

    返回：
        每个话题及其代表性关键词
    """

    if df is None or df.empty:
        raise ValueError("没有可供分析的评论数据")

    if "content" not in df.columns:
        raise ValueError("数据中缺少 content 字段")

    if n_topics < 1 or n_words < 1:
        raise ValueError("话题数量和关键词数量必须大于 0")

    texts = df["content"].fillna("").astype(str).tolist()

    tokenized_texts = [
        " ".join(tokenize_text(text))
        for text in texts
    ]

    valid_texts = [
        text for text in tokenized_texts if text.strip()
    ]

    if len(valid_texts) < 2:
        raise ValueError("有效评论数量太少，至少需要 2 条评论")

    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        max_features=1000
    )

    document_matrix = vectorizer.fit_transform(valid_texts)

    if document_matrix.shape[1] < 2:
        raise ValueError("有效词语数量太少，无法进行话题探索")

    actual_topics = min(n_topics, len(valid_texts))

    lda_model = LatentDirichletAllocation(
        n_components=actual_topics,
        random_state=42,
        max_iter=20,
        learning_method="batch"
    )

    lda_model.fit(document_matrix)

    feature_names = vectorizer.get_feature_names_out()

    topics = []

    for topic_index, topic_weights in enumerate(
        lda_model.components_
    ):
        top_indices = topic_weights.argsort()[:-n_words - 1:-1]

        keywords = [
            feature_names[index]
            for index in top_indices
        ]

        topics.append({
            "topic_id": topic_index + 1,
            "keywords": keywords
        })

    return topics
