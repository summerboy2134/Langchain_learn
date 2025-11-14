"""
基础记忆示例

演示 LangChain 中的记忆管理基础
记忆用于在对话中保持上下文
"""
from langchain_openai import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
)
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, AIMessage
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    get_modelverse_api_key,
    get_modelverse_api_base,
    get_modelverse_model,
)


def basic_buffer_memory():
    """基础缓冲记忆"""
    print("=" * 50)
    print("示例 1: 基础缓冲记忆（保存所有对话）")
    print("=" * 50)
    
    try:
        # 创建记忆对象
        memory = ConversationBufferMemory()
        
        # 手动添加对话历史
        memory.save_context(
            {"input": "你好！"},
            {"output": "你好！很高兴见到你。有什么我可以帮助你的吗？"}
        )
        
        memory.save_context(
            {"input": "我叫张三"},
            {"output": "很高兴认识你，张三！"}
        )
        
        memory.save_context(
            {"input": "我在学习 Python"},
            {"output": "Python 是一个很好的选择！需要帮助吗？"}
        )
        
        # 查看记忆内容
        print("\n记忆中的对话历史:")
        print(memory.load_memory_variables({})["history"])
        
        # 查看原始消息
        print("\n原始消息列表:")
        for msg in memory.chat_memory.messages:
            print(f"{msg.type}: {msg.content}")
        
    except Exception as e:
        print(f"错误: {e}")


def conversation_with_memory():
    """带记忆的对话链"""
    print("\n" + "=" * 50)
    print("示例 2: 带记忆的对话链")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 创建带记忆的对话链
        memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True  # 显示详细信息
        )
        
        # 进行多轮对话
        print("\n开始对话...\n")
        
        response1 = conversation.predict(input="你好，我叫李明")
        print(f"用户: 你好，我叫李明")
        print(f"助手: {response1}\n")
        
        response2 = conversation.predict(input="我在学习 LangChain")
        print(f"用户: 我在学习 LangChain")
        print(f"助手: {response2}\n")
        
        response3 = conversation.predict(input="我叫什么名字？")
        print(f"用户: 我叫什么名字？")
        print(f"助手: {response3}\n")
        
        # 查看完整的对话历史
        print("\n完整对话历史:")
        print(memory.load_memory_variables({})["history"])
        
    except Exception as e:
        print(f"错误: {e}")


def window_memory():
    """窗口记忆（只保留最近的 K 轮对话）"""
    print("\n" + "=" * 50)
    print("示例 3: 窗口记忆（只保留最近 K 轮）")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 创建窗口记忆，只保留最近 2 轮对话
        memory = ConversationBufferWindowMemory(k=2)
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )
        
        # 进行多轮对话
        conversations = [
            "我叫王五",
            "我今年25岁",
            "我住在北京",
            "我喜欢编程",
            "我的名字是什么？"  # 这时第一轮对话应该已经被遗忘了
        ]
        
        for i, user_input in enumerate(conversations, 1):
            print(f"\n对话 {i}:")
            print(f"用户: {user_input}")
            response = conversation.predict(input=user_input)
            print(f"助手: {response}")
            
            # 显示当前窗口中的记忆
            print(f"\n当前记忆窗口（最近 {memory.k} 轮）:")
            history = memory.load_memory_variables({})["history"]
            print(history if history else "[空]")
        
        print("\n说明：由于只保留最近2轮对话，所以助手可能不记得早期的信息")
        
    except Exception as e:
        print(f"错误: {e}")


def memory_key_example():
    """记忆键值操作"""
    print("\n" + "=" * 50)
    print("示例 4: 记忆的键值操作")
    print("=" * 50)
    
    try:
        # 创建记忆
        memory = ConversationBufferMemory()
        
        # 添加多轮对话
        conversations = [
            ("Python 是什么？", "Python 是一种编程语言"),
            ("它容易学吗？", "是的，Python 以易学著称"),
            ("推荐学习资源", "我推荐官方文档和实践项目"),
        ]
        
        for user_msg, ai_msg in conversations:
            memory.save_context({"input": user_msg}, {"output": ai_msg})
        
        # 获取记忆变量
        memory_vars = memory.load_memory_variables({})
        print("\n记忆变量:")
        for key, value in memory_vars.items():
            print(f"\n键: {key}")
            print(f"值:\n{value}")
        
        # 获取消息列表
        print("\n\n消息列表:")
        for i, msg in enumerate(memory.chat_memory.messages, 1):
            print(f"{i}. {msg.type.upper()}: {msg.content}")
        
        # 清除记忆
        print("\n\n清除记忆...")
        memory.clear()
        print("记忆已清除")
        print(f"剩余消息数: {len(memory.chat_memory.messages)}")
        
    except Exception as e:
        print(f"错误: {e}")


def custom_memory_key():
    """自定义记忆键名"""
    print("\n" + "=" * 50)
    print("示例 5: 自定义记忆键名")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 使用自定义键名
        memory = ConversationBufferMemory(
            memory_key="chat_history",  # 自定义键名
            input_key="question",       # 输入键名
            output_key="answer"         # 输出键名
        )
        
        # 手动添加对话
        memory.save_context(
            {"question": "什么是机器学习？"},
            {"answer": "机器学习是 AI 的一个分支，让计算机从数据中学习。"}
        )
        
        # 查看记忆
        vars = memory.load_memory_variables({})
        print("\n记忆内容:")
        print(f"键名: chat_history")
        print(f"内容:\n{vars['chat_history']}")
        
    except Exception as e:
        print(f"错误: {e}")


def memory_with_return_messages():
    """返回消息对象而不是字符串"""
    print("\n" + "=" * 50)
    print("示例 6: 返回消息对象")
    print("=" * 50)
    
    try:
        # 创建返回消息对象的记忆
        memory = ConversationBufferMemory(return_messages=True)
        
        # 添加对话
        memory.save_context(
            {"input": "介绍一下深度学习"},
            {"output": "深度学习是机器学习的一个子集，使用神经网络。"}
        )
        
        memory.save_context(
            {"input": "它有什么应用？"},
            {"output": "深度学习应用于图像识别、自然语言处理等。"}
        )
        
        # 获取消息对象
        messages = memory.load_memory_variables({})["history"]
        
        print("\n消息对象列表:")
        for i, msg in enumerate(messages, 1):
            print(f"\n消息 {i}:")
            print(f"  类型: {msg.type}")
            print(f"  内容: {msg.content}")
            print(f"  对象类型: {type(msg).__name__}")
        
    except Exception as e:
        print(f"错误: {e}")


def compare_memory_types():
    """比较不同的记忆类型"""
    print("\n" + "=" * 50)
    print("示例 7: 比较不同记忆类型")
    print("=" * 50)
    
    # 准备测试数据
    test_conversations = [
        ("我叫小明", "很高兴认识你，小明！"),
        ("我在北京工作", "北京是个好地方！"),
        ("我是工程师", "工程师是很棒的职业！"),
        ("我喜欢编程", "编程是很有趣的！"),
    ]
    
    print("\n--- ConversationBufferMemory（保留所有对话）---")
    buffer_memory = ConversationBufferMemory()
    for user, ai in test_conversations:
        buffer_memory.save_context({"input": user}, {"output": ai})
    
    buffer_history = buffer_memory.load_memory_variables({})["history"]
    print(f"记忆内容长度: {len(buffer_history)} 字符")
    print(f"包含的对话轮数: {len(test_conversations)}")
    print(f"记忆内容:\n{buffer_history}\n")
    
    print("\n--- ConversationBufferWindowMemory（只保留最近2轮）---")
    window_memory = ConversationBufferWindowMemory(k=2)
    for user, ai in test_conversations:
        window_memory.save_context({"input": user}, {"output": ai})
    
    window_history = window_memory.load_memory_variables({})["history"]
    print(f"记忆内容长度: {len(window_history)} 字符")
    print(f"窗口大小: {window_memory.k} 轮")
    print(f"记忆内容:\n{window_history}\n")
    
    print("\n说明：")
    print("- BufferMemory: 保留完整历史，内容完整但占用内存较大")
    print("- WindowMemory: 只保留最近K轮，节省内存但会丢失早期信息")


if __name__ == "__main__":
    print("\n🧠 LangChain 基础记忆示例\n")
    
    print("记忆（Memory）是什么？")
    print("记忆使 LLM 能够在对话中保持上下文，记住之前的交互。\n")
    
    try:
        # 运行不需要 API 的示例
        basic_buffer_memory()
        memory_key_example()
        memory_with_return_messages()
        compare_memory_types()
        custom_memory_key()
        
        # 测试 API 密钥
        print("\n" + "=" * 50)
        print("以下示例需要 API 密钥")
        print("=" * 50)
        
        api_key = get_modelverse_api_key()
        print(f"\n✓ API 密钥已加载")
        
        # 运行需要 API 的示例
        conversation_with_memory()
        window_memory()
        
        print("\n💡 记忆类型总结:")
        print("1. ConversationBufferMemory - 保存完整对话历史")
        print("2. ConversationBufferWindowMemory - 只保留最近 K 轮对话")
        print("3. ConversationSummaryMemory - 总结对话历史（见下一个示例）")
        print("4. ConversationSummaryBufferMemory - 结合摘要和缓冲")
        
        print("\n💡 使用场景:")
        print("- 短对话：使用 BufferMemory")
        print("- 长对话：使用 WindowMemory 或 SummaryMemory")
        print("- 需要完整上下文：BufferMemory")
        print("- 节省内存/Token：WindowMemory 或 SummaryMemory")
        
    except ValueError as e:
        print(f"\n❌ 错误: {e}")
        print("\n请确保在环境中设置 MODELVERSE_API_KEY")
    
    print("\n✅ 所有示例运行完成!")


