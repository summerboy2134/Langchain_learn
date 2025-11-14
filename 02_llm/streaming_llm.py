"""
ModelVerse API 流式输出示例
演示如何实时流式接收 AI 的响应
"""
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import get_modelverse_api_key, get_modelverse_api_base, get_modelverse_model


def basic_streaming():
    """基础流式输出"""
    print("=" * 50)
    print("示例 1: 基础流式输出")
    print("=" * 50)
    
    try:
        # 使用 streaming=True 和回调处理器
        chat = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        print("\n提示: 请介绍一下人工智能的发展历史")
        print("\nAI 回答（流式输出）:")
        print("-" * 50)
        
        _ = chat.invoke("请介绍一下人工智能的发展历史")
        
        print("\n" + "-" * 50)
        print("✓ 流式输出完成")
        
    except Exception as e:
        print(f"错误: {e}")


def streaming_with_system_message():
    """带系统消息的流式输出"""
    print("\n" + "=" * 50)
    print("示例 2: 带系统消息的流式输出")
    print("=" * 50)
    
    try:
        chat = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        messages = [
            SystemMessage(content="你是一个专业的 Python 教师，请用通俗易懂的方式解释概念。"),
            HumanMessage(content="什么是 Python 装饰器？请详细解释并给出例子。")
        ]
        
        print("\n任务: Python 装饰器讲解")
        print("\nAI 回答（流式输出）:")
        print("-" * 50)
        
        _ = chat.invoke(messages)
        
        print("\n" + "-" * 50)
        print("✓ 流式输出完成")
        
    except Exception as e:
        print(f"错误: {e}")


def streaming_story_generation():
    """流式生成故事"""
    print("\n" + "=" * 50)
    print("示例 3: 流式生成故事")
    print("=" * 50)
    
    try:
        chat = ChatOpenAI(
            temperature=0.9,  # 高温度以获得更有创意的输出
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        print("\n任务: 创作一个关于未来AI的短篇科幻故事")
        print("\nAI 创作（流式输出）:")
        print("-" * 50)
        
        _ = chat.invoke("写一个200字左右的短篇科幻故事，主题是：2050年，AI与人类和谐共存的世界")
        
        print("\n" + "-" * 50)
        print("✓ 故事创作完成")
        
    except Exception as e:
        print(f"错误: {e}")


def streaming_code_generation():
    """流式生成代码"""
    print("\n" + "=" * 50)
    print("示例 4: 流式生成代码")
    print("=" * 50)
    
    try:
        chat = ChatOpenAI(
            temperature=0.2,  # 代码生成使用较低温度
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        messages = [
            SystemMessage(content="你是一个 Python 专家。请生成简洁、高效、有注释的代码。"),
            HumanMessage(content="写一个 Python 类，实现一个简单的银行账户，包括存款、取款和查询余额功能。")
        ]
        
        print("\n任务: 生成银行账户类")
        print("\nAI 生成代码（流式输出）:")
        print("-" * 50)
        
        _ = chat.invoke(messages)
        
        print("\n" + "-" * 50)
        print("✓ 代码生成完成")
        
    except Exception as e:
        print(f"错误: {e}")


def streaming_translation():
    """流式翻译"""
    print("\n" + "=" * 50)
    print("示例 5: 流式翻译")
    print("=" * 50)
    
    try:
        chat = ChatOpenAI(
            temperature=0.3,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        text = """
        人工智能是计算机科学的一个分支，它企图了解智能的实质，
        并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
        """
        
        messages = [
            SystemMessage(content="你是专业翻译，请将中文翻译成流畅自然的英文。"),
            HumanMessage(content=f"请将以下中文翻译成英文：{text}")
        ]
        
        print("\n任务: 中译英")
        print(f"原文: {text.strip()}")
        print("\n英文翻译（流式输出）:")
        print("-" * 50)
        
        _ = chat.invoke(messages)
        
        print("\n" + "-" * 50)
        print("✓ 翻译完成")
        
    except Exception as e:
        print(f"错误: {e}")


def compare_streaming_vs_normal():
    """对比流式输出和普通输出"""
    print("\n" + "=" * 50)
    print("示例 6: 流式 vs 普通输出对比")
    print("=" * 50)
    
    import time
    
    prompt = "请详细解释什么是区块链技术，包括它的工作原理和应用场景。"
    
    try:
        # 普通输出
        print("\n【普通输出模式】")
        print("等待响应...")
        start_time = time.time()
        
        chat_normal = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
            streaming=False
        )
        
        response_obj = chat_normal.invoke(prompt)
        normal_time = time.time() - start_time
        response = response_obj.content if hasattr(response_obj, "content") else str(response_obj)
        
        print(f"\n{response[:200]}...")
        print(f"\n耗时: {normal_time:.2f} 秒")
        print("说明: 需要等待完整响应后才能看到结果")
        
        # 流式输出
        print("\n" + "-" * 50)
        print("\n【流式输出模式】")
        print("实时显示响应:")
        start_time = time.time()
        
        chat_streaming = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        _ = chat_streaming.invoke(prompt)
        streaming_time = time.time() - start_time
        
        print(f"\n\n耗时: {streaming_time:.2f} 秒")
        print("说明: 内容逐步显示，用户体验更好")
        
        print("\n" + "=" * 50)
        print("流式输出的优势:")
        print("✓ 用户立即看到响应开始")
        print("✓ 更好的交互体验")
        print("✓ 适合长文本生成")
        print("✓ 可以提前中断不满意的响应")
        
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    print("\n🌊 ModelVerse API 流式输出示例\n")
    print("演示实时流式接收 DeepSeek-R1 模型的响应")
    
    try:
        # 运行各个示例
        basic_streaming()
        streaming_with_system_message()
        streaming_story_generation()
        streaming_code_generation()
        streaming_translation()
        compare_streaming_vs_normal()
        
        print("\n" + "=" * 50)
        print("✅ 所有流式示例运行完成！")
        print("=" * 50)
        
        print("\n💡 流式输出的应用场景:")
        print("• 聊天机器人（实时对话）")
        print("• 内容生成（文章、故事、代码）")
        print("• 实时翻译")
        print("• 代码助手（实时代码生成）")
        print("• 长文本分析和总结")
        
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")


