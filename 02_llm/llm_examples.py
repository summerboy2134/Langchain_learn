"""
LLM 简洁示例（LangChain 1.0 写法）

统一使用 ModelVerse（deepseek-ai/DeepSeek-R1）
包含：基础调用、LCEL 链、批量处理、流式输出
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import sys
import os

# 允许示例以脚本方式运行时找到 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    get_modelverse_api_key,
    get_modelverse_api_base,
    get_modelverse_model,
)


def example_1_basic_call():
    """示例 1：最小可用的直接调用"""
    print("=" * 60)
    print("示例 1：基础 LLM 调用（.invoke）")
    print("=" * 60)

    try:
        llm = ChatOpenAI(
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
            temperature=0.7,
        )

        result = llm.invoke("用一句话解释什么是人工智能。")
        print(f"\n回答: {result.content}")
    except Exception as e:
        print(f"错误: {e}")
        print("提示：请配置 MODELVERSE_API_KEY")


def example_2_lcel_chain():
    """示例 2：LCEL（prompt | llm | parser）"""
    print("\n" + "=" * 60)
    print("示例 2：LCEL 链（prompt | llm | StrOutputParser）")
    print("=" * 60)

    try:
        llm = ChatOpenAI(
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
            temperature=0.3,
        )

        prompt = ChatPromptTemplate.from_template(
            "你是资深{role}。请用通俗语言解释：{topic}"
        )
        chain = prompt | llm | StrOutputParser()

        text = chain.invoke({"role": "Python 工程师", "topic": "生成器"})
        print("\n输出：")
        print(text)
    except Exception as e:
        print(f"错误: {e}")


def example_3_batch_processing():
    """示例 3：批量调用（.batch）"""
    print("\n" + "=" * 60)
    print("示例 3：批量处理（.batch）")
    print("=" * 60)

    try:
        llm = ChatOpenAI(
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
            temperature=0.2,
        )

        questions = [
            "什么是机器学习？",
            "什么是深度学习？",
            "什么是神经网络？",
        ]
        results = llm.batch(questions)

        for i, r in enumerate(results, 1):
            content = r.content if hasattr(r, "content") else str(r)
            print(f"\n{i}. {content[:100]}...")
    except Exception as e:
        print(f"错误: {e}")


def example_4_streaming():
    """示例 4：流式输出（streaming=True + 回调）"""
    print("\n" + "=" * 60)
    print("示例 4：流式输出（StreamingStdOutCallbackHandler）")
    print("=" * 60)

    try:
        llm = ChatOpenAI(
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
            temperature=0.7,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )

        print("\n输出（流式）：", end="", flush=True)
        _ = llm.invoke("请用 2 句话评价 LangChain 的核心优势。")
        print("\n")  # 换行
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    print("\n🧪 LLM 简洁示例（ModelVerse / DeepSeek-R1）\n")
    example_1_basic_call()
    example_2_lcel_chain()
    example_3_batch_processing()
    example_4_streaming()
    print("✅ 示例完成")


