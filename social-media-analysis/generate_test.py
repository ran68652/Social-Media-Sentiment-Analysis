import pandas as pd

df = pd.DataFrame({
    "comment_id": ["1"],
    "platform": ["短视频平台"],
    "content": ["很棒" * 1001]
})

df.to_csv(
    "data/test_long_comment.csv",
    index=False,
    encoding="utf-8-sig"
)

print("测试文件生成成功！")