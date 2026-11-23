
import pandas as pd

from services.analysis_summary import generate_analysis_summary


# 读取已经完成情感分析的数据
df = pd.read_csv(
    "data/sentiment_comments.csv",
    encoding="utf-8-sig"
)

# 生成文字摘要
result = generate_analysis_summary(df)

print("\n========== 情感分析摘要 ==========")

for key, value in result.items():
    print(f"{key}: {value}")

print("\n========== 测试结果 ==========")

assert result["has_data"] is True
assert result["total"] == 45

assert result["positive_count"] == 30
assert result["negative_count"] == 12
assert result["neutral_count"] == 3

assert result["dominant_sentiment"] == "正面"

print("所有测试通过！")
