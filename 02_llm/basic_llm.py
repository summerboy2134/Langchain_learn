"""
基础 LLM 调用示例

演示如何使用 LangChain 与大型语言模型交互（ModelVerse / DeepSeek-R1）
"""
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    get_modelverse_api_key,
    get_modelverse_api_base,
    get_modelverse_model,
)


def basic_llm_call():
    """基础 LLM 调用"""
    print("=" * 50)
    print("示例 1: 基础 LLM 调用")
    print("=" * 50)
    
    try:
        # 使用 ModelVerse（DeepSeek-R1）
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 简单调用
        prompt = "用一句话解释什么是人工智能。"
        response = llm.invoke(prompt)
        
        print(f"\n提示: {prompt}")
        print(f"回答: {response.content if hasattr(response, 'content') else response}")
        
    except Exception as e:
        print(f"错误: {e}")


def chat_model_call():
    """聊天模型调用"""
    print("\n" + "=" * 50)
    print("示例 2: 聊天模型调用")
    print("=" * 50)
    
    try:
        # 初始化聊天模型
        chat = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 使用消息格式
        messages = [
            SystemMessage(content="你是一个专业的 Python 教师。"),
            HumanMessage(content="请解释什么是装饰器？")
        ]
        
        response = chat.invoke(messages)
        
        print("\n消息:")
        for msg in messages:
            print(f"{msg.type}: {msg.content}")
        
        print(f"\n助手: {response.content if hasattr(response, 'content') else response}")
        
    except Exception as e:
        print(f"错误: {e}")


def temperature_comparison():
    """温度参数对比"""
    print("\n" + "=" * 50)
    print("示例 3: 温度参数的影响")
    print("=" * 50)
    
    try:
        prompt = "给我讲一个关于机器人的短故事。"
        
        temperatures = [0.0, 0.5, 1.0]
        
        for temp in temperatures:
            llm = ChatOpenAI(
                temperature=temp,
                model=get_modelverse_model(),
                openai_api_key=get_modelverse_api_key(),
                base_url=get_modelverse_api_base(),
            )
            
            response = llm.invoke(prompt)
            
            print(f"\n--- 温度: {temp} ---")
            content = response.content if hasattr(response, "content") else str(response)
            print(f"回答: {content[:150]}...")  # 只显示前150个字符
            
        print("\n说明:")
        print("- 温度 = 0.0: 输出更确定、一致")
        print("- 温度 = 0.5: 平衡创造力和一致性")
        print("- 温度 = 1.0: 输出更有创造性、多样性")
        
    except Exception as e:
        print(f"错误: {e}")


def max_tokens_example():
    """限制输出长度"""
    print("\n" + "=" * 50)
    print("示例 4: 控制输出长度")
    print("=" * 50)
    
    try:
        prompt = "详细解释机器学习的概念。"
        
        max_tokens_list = [50, 100, 200]
        
        for max_tokens in max_tokens_list:
            llm = ChatOpenAI(
                temperature=0.7,
                model=get_modelverse_model(),
                max_tokens=max_tokens,
                openai_api_key=get_modelverse_api_key(),
                base_url=get_modelverse_api_base(),
            )
            
            response = llm.invoke(prompt)
            
            print(f"\n--- 最大令牌数: {max_tokens} ---")
            content = response.content if hasattr(response, "content") else str(response)
            print(f"回答: {content}")
            print(f"实际长度: {len(content)} 字符")
        
    except Exception as e:
        print(f"错误: {e}")


def multi_turn_conversation():
    """多轮对话"""
    print("\n" + "=" * 50)
    print("示例 5: 多轮对话")
    print("=" * 50)
    
    try:
        chat = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 构建对话历史
        messages = [
            SystemMessage(content="你是一个友好的编程助手。"),
        ]
        
        # 第一轮
        messages.append(HumanMessage(content="什么是列表？"))
        response1 = chat.invoke(messages)
        messages.append(AIMessage(content=response1.content))
        
        print("轮次 1:")
        print(f"用户: 什么是列表？")
        print(f"助手: {response1.content[:100]}...\n")
        
        # 第二轮
        messages.append(HumanMessage(content="如何向列表添加元素？"))
        response2 = chat.invoke(messages)
        messages.append(AIMessage(content=response2.content))
        
        print("轮次 2:")
        print(f"用户: 如何向列表添加元素？")
        print(f"助手: {response2.content[:100]}...\n")
        
        # 第三轮
        messages.append(HumanMessage(content="给我一个例子"))
        response3 = chat.invoke(messages)
        
        print("轮次 3:")
        print(f"用户: 给我一个例子")
        print(f"助手: {response3.content if hasattr(response3, 'content') else response3}")
        
    except Exception as e:
        print(f"错误: {e}")


def batch_processing():
    """批量处理"""
    print("\n" + "=" * 50)
    print("示例 6: 批量处理多个提示")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 多个提示
        prompts = [
            "Python 中的列表是什么？",
            "Python 中的字典是什么？",
            "Python 中的元组是什么？",
        ]
        
        print("\n批量处理结果:")
        for i, prompt in enumerate(prompts, 1):
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            print(f"\n{i}. 问题: {prompt}")
            print(f"   回答: {content[:80]}...")
        
    except Exception as e:
        print(f"错误: {e}")


def different_models():
    """对比不同模型"""
    print("\n" + "=" * 50)
    print("示例 7: 使用不同的模型")
    print("=" * 50)
    
    try:
        prompt = "解释量子计算的基本概念。"
        
        models = [get_modelverse_model()]
        
        for model_name in models:
            try:
                llm = ChatOpenAI(
                    temperature=0.7,
                    model=model_name,
                    openai_api_key=get_modelverse_api_key(),
                    base_url=get_modelverse_api_base(),
                )
                
                response = llm.invoke(prompt)
                content = response.content if hasattr(response, "content") else str(response)
                
                print(f"\n--- 模型: {model_name} ---")
                print(f"回答: {content[:150]}...")
                
            except Exception as e:
                print(f"\n--- 模型: {model_name} ---")
                print(f"错误: {e}")
                print("提示: 请检查 MODELVERSE 配置")
        
    except Exception as e:
        print(f"错误: {e}")


def error_handling():
    """错误处理"""
    print("\n" + "=" * 50)
    print("示例 8: 错误处理")
    print("=" * 50)
    
    try:
        # 尝试使用无效的 API 密钥
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key="invalid_key"
        )
        
        response = llm.invoke("测试")
        
    except Exception as e:
        print(f"\n捕获到错误: {type(e).__name__}")
        print(f"错误信息: {str(e)[:100]}...")
        print("\n正确的做法:")
        print("1. 检查 API 密钥是否正确")
        print("2. 确保网络连接正常")
        print("3. 检查 API 配额是否用尽")


if __name__ == "__main__":
    print("\n🤖 LangChain 基础 LLM 示例\n")
    
    try:
        # 首先测试 API 密钥
        api_key = get_modelverse_api_key()
        print(f"✓ API 密钥已加载: {api_key[:10]}...")
        
        # 运行各个示例
        basic_llm_call()
        chat_model_call()
        temperature_comparison()
        max_tokens_example()
        multi_turn_conversation()
        batch_processing()
        # different_models()  # 取消注释以测试不同模型
        
    except ValueError as e:
        print(f"\n❌ 错误: {e}")
        print("\n请按以下步骤设置:")
        print("1. 在环境中设置 MODELVERSE_API_KEY")
        print("2. 或创建 .env 并添加 MODELVERSE_API_KEY")
        
    # 错误处理示例
    error_handling()
    
    print("\n✅ 所有示例运行完成!")


