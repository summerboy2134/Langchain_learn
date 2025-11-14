"""
LCEL 外部编排示例：先抽取结构化字段，再自行调用函数
"""
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import sys
import os

# 允许脚本直接运行时找到 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
	get_modelverse_api_key,
	get_modelverse_api_base,
	get_modelverse_model,
)


# 1) 定义结构化 schema
class RectInput(BaseModel):
	length: float = Field(..., description="长/边1（cm）")
	width: float = Field(..., description="宽/边2（cm）")


# 2) 抽取链（LCEL）
parser = JsonOutputParser(pydantic_object=RectInput)
format_instructions = parser.get_format_instructions()
prompt = ChatPromptTemplate.from_template(
	"从文本中抽取 length 与 width（单位cm；若只给两个数按先长后宽）。\n"
	"{format_instructions}\n文本：{text}"
)

llm = ChatOpenAI(
	model=get_modelverse_model(),
	base_url=get_modelverse_api_base(),
	openai_api_key=get_modelverse_api_key(),
	temperature=0.0,
)
extract_chain = prompt | llm | parser


# 3) 先抽取
def extract_rect(text: str) -> RectInput:
	result = extract_chain.invoke({"text": text, "format_instructions": format_instructions})
	# 有些版本的解析器返回 dict，这里统一转为 Pydantic 模型，方便后续 data.dict()
	if isinstance(result, BaseModel):
		return result
	if isinstance(result, dict):
		return RectInput(**result)
	raise TypeError(f"Unexpected parse result type: {type(result)}")


# 4) 再调用你的函数/工具（顺序由你控制）
def rectangle_area(length: float, width: float) -> float:
	return length * width


if __name__ == "__main__":
	raw = "一边11，另一边2"
	data = extract_rect(raw)
	print(f"输入文本: {raw}")
	print(f"抽取结果: {data}")
	print(f"面积: {rectangle_area(**data.model_dump())}")


