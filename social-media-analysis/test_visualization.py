
import unittest
import pandas as pd

from app import (
    app,
    load_comments,
    save_topic_analysis,
    KEYWORDS_FILE,
    TOPICS_FILE
)


class TestVisualization(unittest.TestCase):
    """可视化功能自动化测试"""

    @classmethod
    def setUpClass(cls):
        """测试前准备分析数据"""

        app.config["TESTING"] = True

        # 读取当前评论数据
        cls.df = load_comments()

        # 生成并保存关键词和话题分析结果
        save_topic_analysis(cls.df)

        # 创建 Flask 测试客户端
        cls.client = app.test_client()

    def test_keyword_file_exists(self):
        """测试关键词文件是否生成"""

        self.assertTrue(
            KEYWORDS_FILE.exists(),
            "keywords.csv 文件不存在"
        )

    def test_keyword_file_structure(self):
        """测试关键词文件的数据结构"""

        keyword_df = pd.read_csv(
            KEYWORDS_FILE,
            encoding="utf-8-sig"
        )

        self.assertIn("keyword", keyword_df.columns)
        self.assertIn("score", keyword_df.columns)

        self.assertGreater(
            len(keyword_df),
            0,
            "关键词数据不能为空"
        )

        # 检查关键词权重是否按照降序排列
        scores = keyword_df["score"].tolist()

        self.assertEqual(
            scores,
            sorted(scores, reverse=True),
            "关键词权重没有按照降序排列"
        )

    def test_topic_file_exists(self):
        """测试话题文件是否生成"""

        self.assertTrue(
            TOPICS_FILE.exists(),
            "topics.csv 文件不存在"
        )

    def test_topic_file_structure(self):
        """测试话题文件的数据结构"""

        topic_df = pd.read_csv(
            TOPICS_FILE,
            encoding="utf-8-sig"
        )

        self.assertIn("topic_id", topic_df.columns)
        self.assertIn("keywords", topic_df.columns)

        self.assertEqual(
            len(topic_df),
            3,
            "预期生成3个话题"
        )

        # 检查每个话题是否包含5个关键词
        for keywords in topic_df["keywords"]:

            word_list = str(keywords).split("、")

            self.assertEqual(
                len(word_list),
                5,
                "每个话题应包含5个关键词"
            )

    def test_homepage_status(self):
        """测试首页是否正常访问"""

        response = self.client.get("/")

        self.assertEqual(
            response.status_code,
            200
        )

    def test_keyword_chart_exists(self):
        """测试关键词图表是否正常渲染"""

        response = self.client.get("/")

        self.assertIn(
            b"keywordChart",
            response.data
        )

    def test_topic_chart_exists(self):
        """测试话题图表是否正常渲染"""

        response = self.client.get("/")

        self.assertIn(
            b"topicChart",
            response.data
        )

    def test_visualization_sections(self):
        """测试可视化相关页面内容"""

        response = self.client.get("/")

        self.assertIn(
            "关键词分析".encode("utf-8"),
            response.data
        )

        self.assertIn(
            "话题探索".encode("utf-8"),
            response.data
        )

        self.assertIn(
            "评论情感分布".encode("utf-8"),
            response.data
        )


if __name__ == "__main__":

    unittest.main(verbosity=2)
