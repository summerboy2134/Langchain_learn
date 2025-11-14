"""
摘要记忆高级示例

演示如何使用摘要记忆处理长对话和复杂场景
"""
from langchain_openai import ChatOpenAI
from langchain.memory import (
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
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


def long_conversation_with_summary():
    """长对话使用摘要记忆"""
    print("=" * 50)
    print("示例 1: 长对话使用摘要记忆")
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
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )
        
        # 模拟一次长对话
        print("\n进行长对话...\n")
        
        inputs = [
            "你好，我是一名软件工程师，在一家互联网公司工作",
            "我主要使用 Python 和 JavaScript 进行开发",
            "我们公司最近在做一个电商项目",
            "这个项目使用微服务架构，前后端分离",
            "后端用 Django，前端用 React",
            "我们团队有10个人，包括5个后端、3个前端、2个测试",
            "项目预计6个月完成，现在已经进行了2个月",
            "目前遇到的主要挑战是性能优化和数据一致性",
            "我们在考虑引入缓存和消息队列",
            "老板对项目进度比较满意"
        ]
        
        for i, user_input in enumerate(inputs, 1):
            if i % 3 == 1:  # 每3轮显示一次
                print(f"--- 轮次 {i} ---")
                print(f"用户: {user_input}")
            response = conversation.predict(input=user_input)
            if i % 3 == 1:
                print(f"助手: {response[:80]}...\n")
        
        # 查看摘要
        print("\n对话摘要:")
        summary = memory.load_memory_variables({})["history"]
        print(summary)
        
        # 测试记忆
        print("\n\n测试记忆理解:")
        test_questions = [
            "我是做什么工作的？",
            "我们团队有多少人？",
            "项目进展如何？"
        ]
        
        for question in test_questions:
            print(f"\n用户: {question}")
            answer = conversation.predict(input=question)
            print(f"助手: {answer}")
        
    except Exception as e:
        print(f"错误: {e}")


def compare_summary_vs_buffer():
    """对比摘要记忆和缓冲记忆"""
    print("\n" + "=" * 50)
    print("示例 2: 摘要记忆 vs 缓冲记忆对比")
    print("=" * 50)
    
    try:
        from langchain.memory import ConversationBufferMemory
        
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 准备测试对话
        conversations = [
            ("我叫李华，是一名数据科学家", "很高兴认识你，李华！"),
            ("我在金融行业工作了5年", "金融行业经验很宝贵！"),
            ("我擅长机器学习和数据分析", "这些是非常重要的技能！"),
            ("我最近在研究深度学习", "深度学习是个激动人心的领域！"),
            ("特别关注计算机视觉应用", "计算机视觉有很多实际应用！")
        ]
        
        # 1. 缓冲记忆
        print("\n--- 使用缓冲记忆 ---")
        buffer_memory = ConversationBufferMemory()
        for user, ai in conversations:
            buffer_memory.save_context({"input": user}, {"output": ai})
        
        buffer_history = buffer_memory.load_memory_variables({})["history"]
        print(f"记忆长度: {len(buffer_history)} 字符")
        print(f"\n完整内容:\n{buffer_history}")
        
        # 2. 摘要记忆
        print("\n\n--- 使用摘要记忆 ---")
        summary_memory = ConversationSummaryMemory(llm=llm)
        for user, ai in conversations:
            summary_memory.save_context({"input": user}, {"output": ai})
        
        summary_history = summary_memory.load_memory_variables({})["history"]
        print(f"摘要长度: {len(summary_history)} 字符")
        print(f"\n摘要内容:\n{summary_history}")
        
        # 比较
        print("\n\n--- 对比分析 ---")
        print(f"缓冲记忆长度: {len(buffer_history)} 字符")
        print(f"摘要记忆长度: {len(summary_history)} 字符")
        print(f"压缩比例: {len(summary_history) / len(buffer_history) * 100:.1f}%")
        
        print("\n优缺点:")
        print("缓冲记忆:")
        print("  ✓ 保留完整细节")
        print("  ✗ 占用空间大")
        print("  ✗ Token 消耗多")
        
        print("\n摘要记忆:")
        print("  ✓ 节省空间")
        print("  ✓ 降低 Token 消耗")
        print("  ✗ 可能丢失细节")
        
    except Exception as e:
        print(f"错误: {e}")


def summary_buffer_memory_advanced():
    """摘要缓冲记忆高级用法"""
    print("\n" + "=" * 50)
    print("示例 3: 摘要缓冲记忆高级用法")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 创建摘要缓冲记忆
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=150,
            return_messages=False
        )
        
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False
        )
        
        # 进行对话
        print("\n进行对话...\n")
        
        dialogue = [
            "我正在学习 Python 编程",
            "我已经学了基础语法和数据结构",
            "现在想学习面向对象编程",
            "听说类和对象很重要",
            "你能给我一些学习建议吗？"
        ]
        
        for i, user_input in enumerate(dialogue, 1):
            print(f"轮次 {i}: {user_input}")
            response = conversation.predict(input=user_input)
            print(f"回答: {response[:100]}...\n")
            
            # 显示当前记忆状态
            if i == len(dialogue):
                print("最终记忆内容:")
                final_history = memory.load_memory_variables({})["history"]
                print(final_history)
        
        # 继续对话，测试记忆
        print("\n\n测试记忆:")
        test_input = "我在学习什么？请总结一下我们的讨论。"
        print(f"用户: {test_input}")
        test_response = conversation.predict(input=test_input)
        print(f"助手: {test_response}")
        
    except Exception as e:
        print(f"错误: {e}")


def custom_summary_prompt():
    """自定义摘要提示"""
    print("\n" + "=" * 50)
    print("示例 4: 自定义摘要提示")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 自定义摘要提示
        summary_prompt = PromptTemplate(
            input_variables=["summary", "new_lines"],
            template="""以项目管理的角度，简洁地总结以下对话内容，重点关注：
            1. 关键信息和数据
            2. 重要决策
            3. 待办事项

            当前摘要:
            {summary}

            新的对话:
            {new_lines}

            新的摘要:"""
        )
        
        # 创建带自定义提示的摘要记忆
        memory = ConversationSummaryMemory(
            llm=llm,
            prompt=summary_prompt
        )
        
        # 添加项目讨论对话
        project_discussions = [
            ("项目名称是智能客服系统", "好的，记录下来了"),
            ("预算是50万元人民币", "了解，预算已记录"),
            ("需要在3个月内完成", "明白，时间比较紧"),
            ("需要支持多语言", "这是一个重要需求"),
            ("要集成现有的CRM系统", "需要做好对接工作"),
        ]
        
        print("\n添加项目讨论对话...\n")
        for user, ai in project_discussions:
            print(f"用户: {user}")
            print(f"助手: {ai}\n")
            memory.save_context({"input": user}, {"output": ai})
        
        # 查看定制化的摘要
        print("项目摘要（以项目管理视角）:")
        summary = memory.load_memory_variables({})["history"]
        print(summary)
        
    except Exception as e:
        print(f"错误: {e}")


def dynamic_summary_updates():
    """动态摘要更新"""
    print("\n" + "=" * 50)
    print("示例 5: 动态摘要更新过程")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        memory = ConversationSummaryMemory(llm=llm)
        
        # 逐步添加对话，观察摘要变化
        conversations = [
            ("我叫Tom，是产品经理", "很高兴认识你，Tom！"),
            ("我们在做一个社交APP", "听起来很有趣！"),
            ("主要功能是短视频分享", "短视频很受欢迎"),
            ("目标用户是18-30岁年轻人", "这个年龄段活跃度高"),
            ("计划先在北京试运营", "从一个城市开始是好策略"),
        ]
        
        print("\n观察摘要的动态变化:\n")
        
        for i, (user, ai) in enumerate(conversations, 1):
            print(f"--- 添加第 {i} 轮对话 ---")
            print(f"用户: {user}")
            print(f"助手: {ai}")
            
            memory.save_context({"input": user}, {"output": ai})
            
            # 显示当前摘要
            current_summary = memory.load_memory_variables({})["history"]
            print(f"\n当前摘要:\n{current_summary}\n")
        
        print("\n说明：每添加新对话，摘要都会自动更新以包含新信息")
        
    except Exception as e:
        print(f"错误: {e}")


def memory_for_different_scenarios():
    """不同场景下的记忆选择"""
    print("\n" + "=" * 50)
    print("示例 6: 不同场景的记忆选择建议")
    print("=" * 50)
    
    scenarios = [
        {
            "name": "客服聊天机器人",
            "特点": "对话较长，但只需记住关键信息",
            "推荐": "ConversationSummaryBufferMemory",
            "原因": "保留最近完整对话，总结历史关键信息"
        },
        {
            "name": "代码助手",
            "特点": "需要记住用户的代码上下文",
            "推荐": "ConversationBufferWindowMemory",
            "原因": "保留最近几轮完整对话，包含代码细节"
        },
        {
            "name": "教学助手",
            "特点": "长时间对话，需要记住学习进度",
            "推荐": "ConversationSummaryMemory",
            "原因": "总结学习历程，节省 Token"
        },
        {
            "name": "快速问答",
            "特点": "短对话，每个问题独立",
            "推荐": "ConversationBufferMemory 或不使用记忆",
            "原因": "对话简短，不需要复杂的记忆管理"
        },
        {
            "name": "项目讨论助手",
            "特点": "需要记住项目细节和决策",
            "推荐": "ConversationSummaryBufferMemory + 自定义提示",
            "原因": "保留重要决策细节，总结一般讨论"
        }
    ]
    
    print("\n场景分析与推荐:\n")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print(f"   特点: {scenario['特点']}")
        print(f"   推荐: {scenario['推荐']}")
        print(f"   原因: {scenario['原因']}\n")
    
    print("\n💡 通用建议:")
    print("- 对话轮数 < 5 → BufferMemory")
    print("- 对话轮数 5-15 → BufferWindowMemory(k=5)")
    print("- 对话轮数 > 15 → SummaryBufferMemory")
    print("- 需要严格控制成本 → SummaryMemory")


def summary_memory_cost_analysis():
    """摘要记忆的成本分析"""
    print("\n" + "=" * 50)
    print("示例 7: 记忆类型的成本分析")
    print("=" * 50)
    
    print("\n假设场景：20轮对话，每轮平均50个token\n")
    
    print("--- BufferMemory ---")
    print("保存内容: 全部20轮对话")
    print("Token 消耗: 20 × 50 = 1000 tokens")
    print("优点: 完整上下文")
    print("缺点: Token 消耗线性增长")
    
    print("\n--- BufferWindowMemory (k=5) ---")
    print("保存内容: 最近5轮对话")
    print("Token 消耗: 5 × 50 = 250 tokens")
    print("优点: Token 消耗固定")
    print("缺点: 丢失早期对话")
    
    print("\n--- SummaryMemory ---")
    print("保存内容: 所有对话的摘要")
    print("Token 消耗: ~200 tokens (摘要)")
    print("       + ~50 tokens (每次总结成本)")
    print("优点: 保留关键信息，节省 Token")
    print("缺点: 需要额外调用 LLM 进行总结")
    
    print("\n--- SummaryBufferMemory (limit=300) ---")
    print("保存内容: 最近几轮完整 + 早期摘要")
    print("Token 消耗: ~300 tokens")
    print("优点: 平衡细节和效率")
    print("缺点: 配置稍复杂")
    
    print("\n\n💰 成本优化建议:")
    print("1. 开发测试阶段：使用 BufferMemory（简单直接）")
    print("2. 生产环境：根据对话长度选择合适的记忆类型")
    print("3. 高频应用：优先考虑 SummaryBufferMemory")
    print("4. 监控 Token 使用：定期分析和优化")


if __name__ == "__main__":
    print("\n📝 LangChain 摘要记忆高级示例\n")
    
    try:
        # 测试 API 密钥
        api_key = get_modelverse_api_key()
        print(f"✓ API 密钥已加载\n")
        
        # 运行各个示例
        long_conversation_with_summary()
        compare_summary_vs_buffer()
        summary_buffer_memory_advanced()
        custom_summary_prompt()
        dynamic_summary_updates()
        memory_for_different_scenarios()
        summary_memory_cost_analysis()
        
        print("\n\n" + "=" * 50)
        print("总结：摘要记忆的使用场景")
        print("=" * 50)
        
        print("\n✅ 适合使用摘要记忆的情况:")
        print("1. 长时间对话（>15轮）")
        print("2. 需要节省 Token 成本")
        print("3. 只需要记住关键信息")
        print("4. 对话总结性质强")
        
        print("\n❌ 不适合使用摘要记忆的情况:")
        print("1. 需要精确的历史细节")
        print("2. 短对话（<5轮）")
        print("3. 代码或技术讨论（细节重要）")
        print("4. 每个回合都需要完整上下文")
        
    except ValueError as e:
        print(f"\n❌ 错误: {e}")
        print("\n请确保在环境中设置 MODELVERSE_API_KEY")
    
    print("\n✅ 所有示例运行完成!")


