"""
简单的 Token 统计工具（近似）
"""
from typing import List

def count_chars(text: str) -> int:
	"""以字符数近似统计"""
	return len(text or "")

def count_list_chars(texts: List[str]) -> int:
	return sum(len(t or "") for t in texts)


