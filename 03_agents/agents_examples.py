"""
Agents 简洁示例（ModelVerse / DeepSeek-R1）

包含三个最小可运行案例：
1) 计算器工具 + ZERO_SHOT_REACT_DESCRIPTION
2) 结构化工具 + STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
3) 自定义小工具集合
4) 流式可观测（Streaming + Callbacks）
"""
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.tools import StructuredTool
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    get_modelverse_api_key,
    get_modelverse_api_base,
    get_modelverse_model,
)


def example_1_calculator():
    print("=" * 60)
    print("示例 1：计算器工具 + ZERO_SHOT_REACT_DESCRIPTION")
    print("=" * 60)

    llm = ChatOpenAI(
        model=get_modelverse_model(),
        openai_api_key=get_modelverse_api_key(),
        base_url=get_modelverse_api_base(),
        temperature=0,
    )

    def calc(expr: str) -> str:
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return f"{expr} = {result}"
        except Exception as e:
            return f"错误：{e}"

    tools = [
        Tool(
            name="Calculator",
            func=calc,
            description="计算数学表达式，如 '12 + 34 / 2'. 输入应是表达式字符串。",
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    q = "如果我有 120 元，买了 3 个 19 元的本子，还剩多少钱？"
    res = agent.invoke({"input": q})
    print("\n答案：", res.get("output", res))


def example_2_structured_tool():
    print("\n" + "=" * 60)
    print("示例 2：结构化工具 + STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION")
    print("=" * 60)

    llm = ChatOpenAI(
        model=get_modelverse_model(),
        openai_api_key=get_modelverse_api_key(),
        base_url=get_modelverse_api_base(),
        temperature=0,
    )

    def rectangle_area(length: float, width: float) -> str:
        return f"面积={length * width}"

    area_tool = StructuredTool.from_function(
        func=rectangle_area,
        name="RectangleArea",
        description="计算矩形面积，需要 length(浮点) 与 width(浮点)。",
    )

    agent = initialize_agent(
        tools=[area_tool],
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    q = "计算一个长 5.5 宽 3 的矩形面积"
    res = agent.invoke({"input": q})
    print("\n答案：", res.get("output", res))


def example_3_small_toolset():
    print("\n" + "=" * 60)
    print("示例 3：自定义小工具集合")
    print("=" * 60)

    llm = ChatOpenAI(
        model=get_modelverse_model(),
        openai_api_key=get_modelverse_api_key(),
        base_url=get_modelverse_api_base(),
        temperature=0,
    )

    def word_len(s: str) -> str:
        return f"长度={len(s)}"

    def reverse_text(s: str) -> str:
        return s[::-1]

    tools = [
        Tool(name="WordLength", func=word_len, description="返回文本长度。输入=字符串"),
        Tool(name="Reverse", func=reverse_text, description="反转文本。输入=字符串"),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    q = "把 'LangChain' 反过来，并告诉我它有多少个字符"
    res = agent.invoke({"input": q})
    print("\n答案：", res.get("output", res))


class AgentRunPrinter(BaseCallbackHandler):
    """简单的回调：打印工具调用与进度"""
    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "Tool")
        print(f"\n[Tool Start] {name} <- {input_str}")
    def on_tool_end(self, output, **kwargs):
        display = output if isinstance(output, str) else str(output)
        print(f"[Tool End] Output: {display[:120]}{'...' if len(display) > 120 else ''}")
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("\n[LLM Start]")
    def on_llm_end(self, response, **kwargs):
        print("\n[LLM End]")


def example_4_streaming_agent():
    print("\n" + "=" * 60)
    print("示例 4：流式可观测（Streaming + Callbacks）")
    print("=" * 60)

    # 启用流式输出 + 自定义回调
    llm = ChatOpenAI(
        model=get_modelverse_model(),
        openai_api_key=get_modelverse_api_key(),
        base_url=get_modelverse_api_base(),
        temperature=0,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler(), AgentRunPrinter()],
    )

    def calc(expr: str) -> str:
        try:
            return str(eval(expr, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"错误：{e}"

    tools = [
        Tool(
            name="Calculator",
            func=calc,
            description="计算数学表达式。输入应是表达式字符串，如 '120 - 3*19'",
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,  # 显示 ReAct 思考/动作/观察
    )

    query = "我有 120 元，买 3 个 19 元的笔记本，还剩多少钱？请先思考再给出答案。"
    print("\n用户问题：", query)
    result = agent.invoke({"input": query})
    print("\n最终答案：", result.get("output", result))


if __name__ == "__main__":
    print("\n🤖 Agents 简洁示例（ModelVerse / DeepSeek-V3）\n")
    example_1_calculator()
    example_2_structured_tool()
    example_3_small_toolset()
    example_4_streaming_agent()
    print("\n✅ 示例完成")


