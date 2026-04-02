from pathlib import Path

import pandas as pd
from datasets import load_dataset


import logging

logger = logging.getLogger(__name__)


def fetch_wiki(output_file: Path, target_count: int, min_char_length=None):
	logger.info("Loading wikimedia/wikipedia (20231101.en)...")

	# 使用 streaming=True，避免下载几十GB的数据
	# "20231101.en" 是目前常用的英文版本，如果你测中文请改为 "20231101.zh"
	ds = load_dataset(
		"wikimedia/wikipedia", "20231101.en", split="train", streaming=True
	)

	collected_data = []

	for i, article in enumerate(ds):
		text = article.get("text", "")

		# 核心逻辑：跳过短文章，只留长文章
		if min_char_length and len(text) < min_char_length:
			continue

		# 收集数据
		collected_data.append(
			{
				"id": article["id"],
				"title": article["title"],
				"text": text,
				"url": article["url"],
			}
		)

		# 进度打印
		if len(collected_data) % 10 == 0:
			logger.info(f"已收集: {len(collected_data)} / {target_count}")

		if len(collected_data) >= target_count:
			break

	# 保存为 CSV
	df = pd.DataFrame(collected_data)
	df.to_csv(output_file, index=False)
	logger.info(f"\n✅ 完成！已保存 {len(df)} 篇长文章到 {output_file}")
	logger.info(f"平均长度: {df['text'].str.len().mean():.0f} 字符")


if __name__ == "__main__":
	import sys

	if len(sys.argv) < 3:
		print(f"Usage: {sys.argv[0]} <output_file> <target_count>")
		sys.exit(1)
	fetch_wiki(Path(sys.argv[1]), int(sys.argv[2]))
