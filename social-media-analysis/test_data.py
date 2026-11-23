import pandas as pd

df = pd.read_csv("data/sample_comments.csv", encoding="utf-8")

print("前5条评论：")
print(df.head())

print("\n数据基本信息：")
print(df.info())

print("\n评论总数：", len(df))

print("\n空值统计：")
print(df.isnull().sum())