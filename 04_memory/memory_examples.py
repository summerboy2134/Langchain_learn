"""
Memory 简洁示例（ModelVerse / DeepSeek-R1）

包含：BufferMemory、WindowMemory、SummaryMemory（最小可运行）
"""
from langchain_openai import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
)
from langchain.chains import ConversationChain
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    get_modelverse_api_key,
    get_modelverse_api_base,
    get_modelverse_model,
)


def example_1_buffer():
    print("=" * 60)
    print("示例 1：BufferMemory（保存所有对话）")
    print("=" * 60)

    llm = ChatOpenAI(
        model=get_modelverse_model(),
        openai_api_key=get_modelverse_api_key(),
        base_url=get_modelverse_api_base(),
        temperature=0.7,
    )

    memory = ConversationBufferMemory()
    conv = ConversationChain(llm=llm, memory=memory, verbose=False)

    conv.predict(input="你好，我叫小明")
    conv.predict(input="我喜欢 Python")
    res = conv.predict(input="我叫什么名字？")
    print("回答：", res)


def example_2_window():
    print("\n" + "=" * 60)
    print("示例 2：WindowMemory（只保留最近 2 轮）")
    print("=" * 60)

    llm = ChatOpenAI(
        model=get_modelverse_model(),
        openai_api_key=get_modelverse_api_key(),
        base_url=get_modelverse_api_base(),
        temperature=0.7,
    )

    memory = ConversationBufferWindowMemory(k=2)
    conv = ConversationChain(llm=llm, memory=memory, verbose=False)

    conv.predict(input="我叫王五")
    conv.predict(input="我在北京")
    conv.predict(input="我喜欢编程")
    res = conv.predict(input="我叫什么名字？")  # 早期信息可能已被遗忘
    print("回答：", res)


def example_3_summary():
    print("\n" + "=" * 60)
    print("示例 3：SummaryMemory（自动摘要历史）")
    print("=" * 60)

    llm = ChatOpenAI(
        model=get_modelverse_model(),
        openai_api_key=get_modelverse_api_key(),
        base_url=get_modelverse_api_base(),
        temperature=0.7,
    )

    memory = ConversationSummaryMemory(llm=llm)
    conv = ConversationChain(llm=llm, memory=memory, verbose=False)

    conv.predict(input="我是一名数据科学家")
    conv.predict(input="我在金融行业工作")
    conv.predict(input="我擅长 Python 和机器学习")
    print("\n当前摘要：")
    print(memory.load_memory_variables({})["history"])

    res = conv.predict(input="我从事哪个行业？")
    print("\n问答：我从事哪个行业？")
    print("回答：", res)


if __name__ == "__main__":
    print("\n🧠 Memory 简洁示例（ModelVerse / DeepSeek-R1）\n")
    example_1_buffer()
    example_2_window()
    example_3_summary()
    print("\n✅ 示例完成")


