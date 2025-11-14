"""
对话记忆示例

演示各种对话记忆类型和高级用法
"""
from langchain_openai import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    ConversationTokenBufferMemory,
)
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    get_modelverse_api_key,
    get_modelverse_api_base,
    get_modelverse_model,
)


def summary_memory():
    """摘要记忆 - 自动总结对话历史"""
    print("=" * 50)
    print("示例 1: 摘要记忆")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 创建摘要记忆
        memory = ConversationSummaryMemory(llm=llm)
        
        # 手动添加对话
        memory.save_context(
            {"input": "你好，我叫张三，是一名软件工程师"},
            {"output": "你好张三！很高兴认识你。作为软件工程师，你一定有丰富的编程经验。"}
        )
        
        memory.save_context(
            {"input": "我在一家互联网公司工作，主要做后端开发，使用 Python 和 Go"},
            {"output": "很棒！Python 和 Go 都是优秀的后端语言。Python 简洁易读，Go 性能出色。"}
        )
        
        memory.save_context(
            {"input": "最近在学习 LangChain，想用它开发一些 AI 应用"},
            {"output": "LangChain 是个很好的选择！它可以帮助你快速构建 AI 应用。有什么具体想做的项目吗？"}
        )
        
        # 查看摘要
        print("\n对话摘要:")
        summary = memory.load_memory_variables({})["history"]
        print(summary)
        
        print("\n说明：摘要记忆会自动总结对话内容，节省 Token 消耗")
        
    except Exception as e:
        print(f"错误: {e}")


def summary_memory_in_conversation():
    """在对话中使用摘要记忆"""
    print("\n" + "=" * 50)
    print("示例 2: 在对话中使用摘要记忆")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 创建带摘要记忆的对话
        memory = ConversationSummaryMemory(llm=llm)
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True
        )
        
        # 进行多轮对话
        responses = []
        
        print("\n开始对话...\n")
        
        inputs = [
            "我是一名数据科学家，在金融行业工作",
            "我每天的工作包括数据分析、建模和可视化",
            "我最喜欢用 Python 的 pandas 和 scikit-learn",
            "我还在学习深度学习，特别是 PyTorch"
        ]
        
        for i, user_input in enumerate(inputs, 1):
            print(f"轮次 {i}:")
            print(f"用户: {user_input}")
            response = conversation.predict(input=user_input)
            print(f"助手: {response[:100]}...")
            responses.append(response)
            
            # 显示当前摘要
            if i < len(inputs):
                print(f"\n当前摘要:")
                print(memory.load_memory_variables({})["history"][:150] + "...")
            print()
        
        # 测试记忆
        print("\n测试记忆:")
        print("用户: 我是做什么工作的？")
        test_response = conversation.predict(input="我是做什么工作的？")
        print(f"助手: {test_response}")
        
    except Exception as e:
        print(f"错误: {e}")


def summary_buffer_memory():
    """摘要缓冲记忆 - 结合摘要和缓冲"""
    print("\n" + "=" * 50)
    print("示例 3: 摘要缓冲记忆")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 创建摘要缓冲记忆
        # max_token_limit: 当对话超过此限制时，旧对话会被总结
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=100  # Token 限制
        )
        
        # 添加对话
        conversations = [
            ("介绍一下你自己", "我是一个 AI 助手，可以帮助你解答问题。"),
            ("你能做什么？", "我可以回答问题、提供建议、帮助编程等。"),
            ("Python 和 Java 有什么区别？", "Python 语法简洁，适合快速开发；Java 更严谨，适合大型项目。"),
            ("推荐学习哪个？", "如果你是初学者，我推荐从 Python 开始。"),
        ]
        
        for user, ai in conversations:
            memory.save_context({"input": user}, {"output": ai})
        
        # 查看记忆内容
        print("\n记忆内容（摘要 + 最近的对话）:")
        history = memory.load_memory_variables({})["history"]
        print(history)
        
        print("\n说明：这种记忆类型保留最近的完整对话，并总结较早的对话")
        
    except Exception as e:
        print(f"错误: {e}")


def token_buffer_memory():
    """基于 Token 的缓冲记忆"""
    print("\n" + "=" * 50)
    print("示例 4: Token 缓冲记忆")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 创建 Token 缓冲记忆
        memory = ConversationTokenBufferMemory(
            llm=llm,
            max_token_limit=60  # 最大 Token 数
        )
        
        # 添加对话
        memory.save_context(
            {"input": "什么是机器学习？"},
            {"output": "机器学习是人工智能的一个分支，让计算机能够从数据中学习和改进。"}
        )
        
        memory.save_context(
            {"input": "它有哪些类型？"},
            {"output": "主要有三种：监督学习、无监督学习和强化学习。"}
        )
        
        memory.save_context(
            {"input": "监督学习是什么？"},
            {"output": "监督学习使用标注数据来训练模型，如分类和回归任务。"}
        )
        
        # 查看记忆
        print("\n当前记忆（受 Token 限制）:")
        history = memory.load_memory_variables({})["history"]
        print(history)
        
        print(f"\n说明：此记忆保留的对话内容不超过 {memory.max_token_limit} 个 Token")
        
    except Exception as e:
        print(f"错误: {e}")


def custom_prompt_with_memory():
    """自定义提示与记忆结合"""
    print("\n" + "=" * 50)
    print("示例 5: 自定义提示与记忆")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 自定义提示模板
        template = """你是一个专业的编程导师，擅长用简单的语言解释复杂的概念。
        
        对话历史:
        {history}
        
        学生问题: {input}
        
        导师回答:"""
        
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )
        
        # 创建记忆
        memory = ConversationBufferMemory()
        
        # 创建对话链
        conversation = ConversationChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            verbose=False
        )
        
        # 对话
        questions = [
            "什么是递归？",
            "能给我一个例子吗？",
            "递归和循环有什么区别？"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n问题 {i}: {question}")
            response = conversation.predict(input=question)
            print(f"回答: {response[:150]}...")
        
    except Exception as e:
        print(f"错误: {e}")


def multiple_memory_chains():
    """使用多个独立的记忆链"""
    print("\n" + "=" * 50)
    print("示例 6: 多个独立的记忆链")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 创建两个独立的对话
        memory1 = ConversationBufferMemory()
        conversation1 = ConversationChain(
            llm=llm,
            memory=memory1,
            verbose=False
        )
        
        memory2 = ConversationBufferMemory()
        conversation2 = ConversationChain(
            llm=llm,
            memory=memory2,
            verbose=False
        )
        
        # 对话 1：关于 Python
        print("\n--- 对话 1：讨论 Python ---")
        response1 = conversation1.predict(input="我想学习 Python")
        print(f"用户: 我想学习 Python")
        print(f"助手: {response1[:100]}...")
        
        response1_2 = conversation1.predict(input="Python 适合哪些应用？")
        print(f"\n用户: Python 适合哪些应用？")
        print(f"助手: {response1_2[:100]}...")
        
        # 对话 2：关于 Java
        print("\n\n--- 对话 2：讨论 Java ---")
        response2 = conversation2.predict(input="我想学习 Java")
        print(f"用户: 我想学习 Java")
        print(f"助手: {response2[:100]}...")
        
        response2_2 = conversation2.predict(input="Java 的优势是什么？")
        print(f"\n用户: Java 的优势是什么？")
        print(f"助手: {response2_2[:100]}...")
        
        # 验证记忆独立性
        print("\n\n--- 验证记忆独立性 ---")
        test1 = conversation1.predict(input="我们在讨论什么语言？")
        print(f"对话1 - 用户: 我们在讨论什么语言？")
        print(f"对话1 - 助手: {test1}")
        
        test2 = conversation2.predict(input="我们在讨论什么语言？")
        print(f"\n对话2 - 用户: 我们在讨论什么语言？")
        print(f"对话2 - 助手: {test2}")
        
    except Exception as e:
        print(f"错误: {e}")


def memory_persistence():
    """记忆持久化示例"""
    print("\n" + "=" * 50)
    print("示例 7: 记忆管理和清除")
    print("=" * 50)
    
    try:
        # 创建记忆
        memory = ConversationBufferMemory()
        
        # 添加对话
        print("\n添加3轮对话...")
        memory.save_context({"input": "你好"}, {"output": "你好！"})
        memory.save_context({"input": "我叫小红"}, {"output": "很高兴认识你，小红！"})
        memory.save_context({"input": "我在学习 AI"}, {"output": "AI 很有趣！"})
        
        print(f"对话数量: {len(memory.chat_memory.messages)}")
        print(f"记忆内容:\n{memory.load_memory_variables({})['history']}")
        
        # 清除记忆
        print("\n\n清除记忆...")
        memory.clear()
        print(f"对话数量: {len(memory.chat_memory.messages)}")
        
        # 重新添加
        print("\n添加新对话...")
        memory.save_context({"input": "新的对话"}, {"output": "这是新的开始"})
        print(f"对话数量: {len(memory.chat_memory.messages)}")
        print(f"记忆内容:\n{memory.load_memory_variables({})['history']}")
        
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    print("\n💬 LangChain 对话记忆示例\n")
    
    try:
        # 测试 API 密钥
        api_key = get_modelverse_api_key()
        print(f"✓ API 密钥已加载\n")
        
        # 运行各个示例
        summary_memory()
        summary_memory_in_conversation()
        summary_buffer_memory()
        token_buffer_memory()
        custom_prompt_with_memory()
        multiple_memory_chains()
        memory_persistence()
        
        print("\n💡 记忆类型详解:")
        print("1. ConversationBufferMemory")
        print("   - 保存完整对话历史")
        print("   - 适合：短对话、需要完整上下文")
        
        print("\n2. ConversationBufferWindowMemory")
        print("   - 只保留最近 K 轮对话")
        print("   - 适合：长对话、固定内存限制")
        
        print("\n3. ConversationSummaryMemory")
        print("   - 总结所有对话历史")
        print("   - 适合：长对话、节省 Token")
        
        print("\n4. ConversationSummaryBufferMemory")
        print("   - 总结旧对话，保留最近完整对话")
        print("   - 适合：平衡详细度和效率")
        
        print("\n5. ConversationTokenBufferMemory")
        print("   - 基于 Token 数量限制")
        print("   - 适合：严格控制 Token 消耗")
        
        print("\n💡 选择建议:")
        print("- 演示/测试 → BufferMemory")
        print("- 生产环境（短对话）→ BufferWindowMemory")
        print("- 生产环境（长对话）→ SummaryBufferMemory")
        print("- 严格 Token 限制 → TokenBufferMemory")
        
    except ValueError as e:
        print(f"\n❌ 错误: {e}")
        print("\n请确保在环境中设置 MODELVERSE_API_KEY")
    
    print("\n✅ 所有示例运行完成!")


