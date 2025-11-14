"""
ModelVerse API 使用示例
使用 DeepSeek-R1 等模型进行 LLM 调用
"""
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import get_modelverse_api_key, get_modelverse_api_base, get_modelverse_model


def basic_modelverse_call():
    """基础 ModelVerse LLM 调用"""
    print("=" * 50)
    print("示例 1: 基础 ModelVerse API 调用")
    print("=" * 50)
    
    try:
        # 初始化 LLM，使用 ModelVerse API
        chat = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base()
        )
        
        # 简单调用
        prompt = "用一句话解释什么是人工智能。"
        response = chat.invoke(prompt)
        
        print(f"\n提示: {prompt}")
        print(f"回答: {response.content}")
        
    except Exception as e:
        print(f"错误: {e}")


def chat_with_system_message():
    """带系统消息的聊天"""
    print("\n" + "=" * 50)
    print("示例 2: 带系统消息的聊天")
    print("=" * 50)
    
    try:
        chat = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base()
        )
        
        messages = [
            SystemMessage(content="你是一个专业的 Python 编程专家。请用简洁专业的方式回答问题。"),
            HumanMessage(content="如何在 Python 中使用列表推导式？")
        ]
        
        response = chat.invoke(messages)
        
        print("\n消息:")
        for msg in messages:
            print(f"  {msg.type}: {msg.content}")
        
        print(f"\nAI回答:\n{response.content}")
        
    except Exception as e:
        print(f"错误: {e}")


def multi_turn_conversation():
    """多轮对话示例"""
    print("\n" + "=" * 50)
    print("示例 3: 多轮对话")
    print("=" * 50)
    
    try:
        chat = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base()
        )
        
        # 对话历史
        messages = [
            SystemMessage(content="你是一个友好的AI助手。"),
        ]
        
        # 第一轮
        messages.append(HumanMessage(content="你好！请介绍一下 LangChain。"))
        response1 = chat.invoke(messages)
        print("\n【第1轮对话】")
        print(f"用户: {messages[-1].content}")
        print(f"AI: {response1.content}\n")
        messages.append(response1)
        
        # 第二轮
        messages.append(HumanMessage(content="它有哪些主要功能？"))
        response2 = chat.invoke(messages)
        print("【第2轮对话】")
        print(f"用户: {messages[-1].content}")
        print(f"AI: {response2.content}\n")
        messages.append(response2)
        
        # 第三轮
        messages.append(HumanMessage(content="给我一个简单的使用例子"))
        response3 = chat.invoke(messages)
        print("【第3轮对话】")
        print(f"用户: {messages[-1].content}")
        print(f"AI: {response3.content}")
        
    except Exception as e:
        print(f"错误: {e}")


def different_temperatures():
    """对比不同温度参数的影响"""
    print("\n" + "=" * 50)
    print("示例 4: 温度参数对比")
    print("=" * 50)
    
    try:
        prompt = "写一句关于春天的诗句"
        temperatures = [0.1, 0.7, 1.0]
        
        for temp in temperatures:
            chat = ChatOpenAI(
                temperature=temp,
                model=get_modelverse_model(),
                openai_api_key=get_modelverse_api_key(),
                base_url=get_modelverse_api_base()
            )
            
            response = chat.invoke(prompt)
            
            print(f"\n【温度: {temp}】")
            print(f"提示: {prompt}")
            print(f"回答: {response.content}")
        
        print("\n说明:")
        print("• 温度 = 0.1: 输出更确定、一致、保守")
        print("• 温度 = 0.7: 平衡创造力和一致性")
        print("• 温度 = 1.0: 输出更有创造性、多样性")
        
    except Exception as e:
        print(f"错误: {e}")


def code_generation_example():
    """代码生成示例"""
    print("\n" + "=" * 50)
    print("示例 5: 代码生成")
    print("=" * 50)
    
    try:
        chat = ChatOpenAI(
            temperature=0.3,  # 代码生成使用较低温度
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base()
        )
        
        messages = [
            SystemMessage(content="你是一个专业的 Python 程序员。请提供简洁、可运行的代码。"),
            HumanMessage(content="写一个 Python 函数，计算斐波那契数列的第 n 项。")
        ]
        
        response = chat.invoke(messages)
        
        print("\n【任务】计算斐波那契数列")
        print(f"\n生成的代码:\n{response.content}")
        
    except Exception as e:
        print(f"错误: {e}")


def batch_questions():
    """批量问答"""
    print("\n" + "=" * 50)
    print("示例 6: 批量问答")
    print("=" * 50)
    
    try:
        chat = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base()
        )
        
        questions = [
            "什么是机器学习？",
            "什么是深度学习？",
            "什么是神经网络？",
        ]
        
        print("\n批量问答结果:")
        for i, question in enumerate(questions, 1):
            response = chat.invoke(question)
            print(f"\n{i}. 问题: {question}")
            print(f"   回答: {response.content[:100]}...")
        
    except Exception as e:
        print(f"错误: {e}")


def practical_use_case():
    """实用案例：文本翻译"""
    print("\n" + "=" * 50)
    print("示例 7: 实用案例 - 文本翻译")
    print("=" * 50)
    
    try:
        chat = ChatOpenAI(
            temperature=0.3,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base()
        )
        
        messages = [
            SystemMessage(content="你是一个专业的翻译助手。请将用户输入的中文翻译成英文，保持原意和语气。"),
            HumanMessage(content="今天天气真好，我们去公园散步吧。")
        ]
        
        response = chat.invoke(messages)
        
        print("\n【翻译任务】")
        print(f"原文: {messages[-1].content}")
        print(f"译文: {response.content}")
        
    except Exception as e:
        print(f"错误: {e}")


def check_api_info():
    """检查 API 配置信息"""
    print("\n" + "=" * 50)
    print("API 配置信息")
    print("=" * 50)
    
    try:
        api_key = get_modelverse_api_key()
        api_base = get_modelverse_api_base()
        model_name = get_modelverse_model()
        
        print(f"\n✓ API Key: {api_key[:20]}...")
        print(f"✓ API Base URL: {api_base}")
        print(f"✓ Model: {model_name}")
        
    except Exception as e:
        print(f"✗ 配置错误: {e}")


if __name__ == "__main__":
    print("\n🚀 ModelVerse API 使用示例\n")
    print("使用 DeepSeek-R1 模型进行各种任务")
    
    # 首先检查 API 配置
    check_api_info()
    
    try:
        # 运行各个示例
        basic_modelverse_call()
        chat_with_system_message()
        multi_turn_conversation()
        different_temperatures()
        code_generation_example()
        batch_questions()
        practical_use_case()
        
        print("\n" + "=" * 50)
        print("✅ 所有示例运行完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        print("\n请检查:")
        print("1. MODELVERSE_API_KEY 环境变量是否设置")
        print("2. 网络连接是否正常")
        print("3. API 配额是否充足")


