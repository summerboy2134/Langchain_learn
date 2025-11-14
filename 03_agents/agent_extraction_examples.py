"""
信息抽取作为“工具”的 Agent 示例（内部编排）+ 结构化参数版

思路：
1) 字符串协议版：把“抽取 length/width 的链”包装成一个工具，Agent 先调用该工具拿到标准字符串，
   再调用“面积计算”工具得到最终答案（ZERO_SHOT）。
2) 结构化参数版：仅提供面积计算 StructuredTool，模型需按 schema 直接填参（STRUCTURED_AGENT）。
"""
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    get_modelverse_api_key,
    get_modelverse_api_base,
    get_modelverse_model,
)


class RectInput(BaseModel):
    length: float = Field(..., description="长/边1（cm）")
    width: float = Field(..., description="宽/边2（cm）")


def build_extraction_chain():
    parser = JsonOutputParser(pydantic_object=RectInput)
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_template(
        "从文本中抽取 length 与 width（单位统一为 cm）。"
        "若未注明单位视为 cm；若只给两个数，按先长后宽映射；可接受别名如“长/宽/边1/边2”。\n"
        "{format_instructions}\n文本：{text}"
    )

    llm = ChatOpenAI(
        model=get_modelverse_model(),
        openai_api_key=get_modelverse_api_key(),
        base_url=get_modelverse_api_base(),
        temperature=0,
    )
    chain = prompt | llm | parser
    return chain, format_instructions


def build_agent():
    # 1) 抽取器链
    extract_chain, fmt = build_extraction_chain()

    # 包装为工具：输入=自然语言，输出=标准字符串 "length=.., width=.."
    def extract_rect(text: str) -> str:
        data = extract_chain.invoke({"text": text, "format_instructions": fmt})
        return f"length={data['length']}, width={data['width']}"

    # 2) 面积计算工具：输入= "length=.., width=.."
    def area_from_string(s: str) -> str:
        m = re.findall(r"(length|width)\s*=\s*([0-9]+\.?[0-9]*)", s)
        values = {k: float(v) for k, v in m}
        if "length" in values and "width" in values:
            return f"面积={values['length'] * values['width']} (cm^2)"
        return "解析失败，请提供 'length=.., width=..' 格式"

    tools = [
        Tool(
            name="ExtractRectangle",
            func=extract_rect,
            description=(
                "当输入是口语化/非结构化时，先用我抽取 length 与 width（单位 cm）。"
                "输入=自然语言文本；输出为 'length=.., width=..'"
            ),
        ),
        Tool(
            name="RectangleAreaFromString",
            func=area_from_string,
            description="根据 'length=.., width=..' 计算面积；输入为该字符串",
        ),
    ]

    llm = ChatOpenAI(
        model=get_modelverse_model(),
        openai_api_key=get_modelverse_api_key(),
        base_url=get_modelverse_api_base(),
        temperature=0,
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent


def run_example():
    print("=" * 60)
    print("示例：Agent 内部编排（抽取工具 -> 面积工具）")
    print("=" * 60)
    agent = build_agent()
    query = "一边11，另一边是2，求面积"
    result = agent.invoke({"input": query})
    print("\n最终答案：", result.get("output", result))


def rectangle_area(length: float, width: float) -> str:
    return f"面积={length * width} (cm^2)"


def run_structured_example():
    print("\n" + "=" * 60)
    print("示例：结构化参数版（StructuredTool + STRUCTURED_AGENT）")
    print("=" * 60)

    llm = ChatOpenAI(
        model=get_modelverse_model(),
        openai_api_key=get_modelverse_api_key(),
        base_url=get_modelverse_api_base(),
        temperature=0,
    )

    # 仅提供“面积计算”一个结构化工具；模型需按 schema 直接填参
    area_tool = StructuredTool.from_function(
        func=rectangle_area,
        name="RectangleArea",
        description="根据 length 与 width 计算矩形面积（单位 cm）。",
        args_schema=RectInput,  # 显式 schema：length/width 为必填的浮点数
    )

    agent = initialize_agent(
        tools=[area_tool],
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    query = "一边11，另一边2，求面积"
    result = agent.invoke({"input": query})
    print("\n最终答案：", result.get("output", result))


if __name__ == "__main__":
    run_example()
    run_structured_example()


