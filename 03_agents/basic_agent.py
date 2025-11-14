"""
基础智能体示例

演示 LangChain 智能体的基本概念和使用方法
智能体可以根据任务自动选择和使用工具
"""
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools, create_react_agent
from langchain.tools import Tool
from langchain import hub
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    get_modelverse_api_key,
    get_modelverse_api_base,
    get_modelverse_model,
)


def simple_calculator_tool():
    """简单的计算器工具"""
    print("=" * 50)
    print("示例 1: 带计算器工具的智能体")
    print("=" * 50)
    
    try:
        # 初始化 LLM
        llm = ChatOpenAI(
            temperature=0,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 加载内置工具
        tools = load_tools(
            ["llm-math"],  # 数学计算工具
            llm=llm
        )
        
        # 创建智能体
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,  # 显示思考过程
            handle_parsing_errors=True
        )
        
        # 运行智能体
        questions = [
            "25 的平方根是多少？",
            "如果我有 150 元，买了 3 个 35 元的商品，还剩多少钱？"
        ]
        
        for question in questions:
            print(f"\n问题: {question}")
            # 统一使用 invoke（也可用 run）
            result = agent.invoke({"input": question})
            result_text = result.get("output", result)
            print(f"答案: {result_text}\n")
        
    except Exception as e:
        print(f"错误: {e}")


def custom_tools_agent():
    """使用自定义工具的智能体"""
    print("\n" + "=" * 50)
    print("示例 2: 自定义工具智能体")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 定义自定义工具函数
        def get_word_length(word: str) -> int:
            """返回单词的长度"""
            return len(word)
        
        def multiply_by_two(number: str) -> int:
            """将数字乘以 2"""
            try:
                return int(number) * 2
            except ValueError:
                return "请提供有效的数字"
        
        def reverse_string(text: str) -> str:
            """反转字符串"""
            return text[::-1]
        
        # 创建工具列表
        tools = [
            Tool(
                name="WordLength",
                func=get_word_length,
                description="用于获取一个单词或文本的长度。输入应该是一个字符串。"
            ),
            Tool(
                name="MultiplyByTwo",
                func=multiply_by_two,
                description="将一个数字乘以 2。输入应该是一个数字。"
            ),
            Tool(
                name="ReverseString",
                func=reverse_string,
                description="反转一个字符串。输入应该是要反转的文本。"
            ),
        ]
        
        # 创建智能体
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        # 测试问题
        questions = [
            "单词 'artificial' 有多少个字母？",
            "如果 15 乘以 2 是多少？",
            "将 'LangChain' 这个词反转。"
        ]
        
        for question in questions:
            print(f"\n问题: {question}")
            result = agent.invoke({"input": question})
            print(f"答案: {result.get('output', result)}\n")
        
    except Exception as e:
        print(f"错误: {e}")


def multi_tool_agent():
    """使用多个工具协同工作的智能体"""
    print("\n" + "=" * 50)
    print("示例 3: 多工具协同智能体")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 定义多个工具
        def celsius_to_fahrenheit(celsius: str) -> str:
            """将摄氏度转换为华氏度"""
            try:
                c = float(celsius)
                f = (c * 9/5) + 32
                return f"{c}°C = {f}°F"
            except:
                return "请提供有效的温度数值"
        
        def fahrenheit_to_celsius(fahrenheit: str) -> str:
            """将华氏度转换为摄氏度"""
            try:
                f = float(fahrenheit)
                c = (f - 32) * 5/9
                return f"{f}°F = {c:.2f}°C"
            except:
                return "请提供有效的温度数值"
        
        def is_freezing(celsius: str) -> str:
            """判断温度是否低于冰点"""
            try:
                c = float(celsius)
                if c <= 0:
                    return "是的，这个温度低于冰点（0°C）"
                else:
                    return "不，这个温度高于冰点（0°C）"
            except:
                return "请提供有效的温度数值"
        
        # 创建工具
        tools = [
            Tool(
                name="CelsiusToFahrenheit",
                func=celsius_to_fahrenheit,
                description="将摄氏度转换为华氏度。输入应该是摄氏度的数值。"
            ),
            Tool(
                name="FahrenheitToCelsius",
                func=fahrenheit_to_celsius,
                description="将华氏度转换为摄氏度。输入应该是华氏度的数值。"
            ),
            Tool(
                name="IsFreezing",
                func=is_freezing,
                description="判断给定的摄氏度温度是否低于冰点。输入应该是摄氏度的数值。"
            ),
        ]
        
        # 创建智能体
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        # 复杂问题
        question = "如果外面的温度是 32 华氏度，这个温度是多少摄氏度？水会结冰吗？"
        
        print(f"\n问题: {question}")
        result = agent.invoke({"input": question})
        print(f"\n最终答案: {result.get('output', result)}\n")
        
    except Exception as e:
        print(f"错误: {e}")


def agent_with_different_types():
    """不同类型的智能体"""
    print("\n" + "=" * 50)
    print("示例 4: 不同类型的智能体")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 简单的工具
        def get_current_date() -> str:
            """获取当前日期"""
            from datetime import datetime
            return datetime.now().strftime("%Y年%m月%d日")
        
        tools = [
            Tool(
                name="GetDate",
                func=get_current_date,
                description="获取今天的日期"
            ),
        ]
        
        # ZERO_SHOT_REACT_DESCRIPTION 类型
        print("\n--- 使用 ZERO_SHOT_REACT_DESCRIPTION ---")
        agent_react = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        question = "今天是几号？"
        result = agent_react.invoke({"input": question})
        print(f"答案: {result.get('output', result)}")
        
    except Exception as e:
        print(f"错误: {e}")


def agent_error_handling():
    """智能体的错误处理"""
    print("\n" + "=" * 50)
    print("示例 5: 智能体错误处理")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        def divide_numbers(input_str: str) -> str:
            """除法运算，输入格式：'数字1,数字2'"""
            try:
                parts = input_str.split(',')
                if len(parts) != 2:
                    return "错误：请提供两个数字，用逗号分隔"
                
                a = float(parts[0].strip())
                b = float(parts[1].strip())
                
                if b == 0:
                    return "错误：除数不能为零"
                
                result = a / b
                return f"{a} ÷ {b} = {result}"
            except Exception as e:
                return f"错误：{str(e)}"
        
        tools = [
            Tool(
                name="Divide",
                func=divide_numbers,
                description="执行除法运算。输入应该是两个数字，用逗号分隔，例如：'10,2'"
            ),
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3  # 限制最大迭代次数
        )
        
        # 测试各种情况
        test_cases = [
            "计算 100 除以 4",
            "计算 50 除以 0",  # 会触发错误
        ]
        
        for question in test_cases:
            print(f"\n问题: {question}")
            try:
                result = agent.invoke({"input": question})
                print(f"答案: {result.get('output', result)}")
            except Exception as e:
                print(f"捕获到错误: {e}")
        
    except Exception as e:
        print(f"错误: {e}")


def agent_with_max_iterations():
    """限制智能体的迭代次数"""
    print("\n" + "=" * 50)
    print("示例 6: 限制智能体迭代次数")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 简单工具
        def get_info(topic: str) -> str:
            """获取主题信息"""
            info_db = {
                "python": "Python 是一种高级编程语言",
                "java": "Java 是一种面向对象的编程语言",
                "javascript": "JavaScript 是一种网页脚本语言"
            }
            return info_db.get(topic.lower(), "抱歉，没有这个主题的信息")
        
        tools = [
            Tool(
                name="GetInfo",
                func=get_info,
                description="获取编程语言的信息。输入应该是语言名称。"
            ),
        ]
        
        # 设置最大迭代次数
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=2,  # 最多 2 次迭代
            early_stopping_method="generate"
        )
        
        question = "告诉我关于 Python 的信息"
        result = agent.invoke({"input": question})
        print(f"\n答案: {result.get('output', result)}")
        
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    print("\n🤖 LangChain 智能体基础示例\n")
    
    print("智能体（Agent）是什么？")
    print("智能体是能够使用工具（Tools）来完成任务的系统。")
    print("它遵循 '思考-行动-观察' 循环：")
    print("  1. 思考（Thought）：分析问题，决定下一步")
    print("  2. 行动（Action）：选择并使用一个工具")
    print("  3. 观察（Observation）：查看工具的输出")
    print("  4. 重复直到得到最终答案\n")
    
    try:
        # 测试 API 密钥
        api_key = get_modelverse_api_key()
        print(f"✓ API 密钥已加载\n")
        
        # 运行各个示例
        simple_calculator_tool()
        custom_tools_agent()
        multi_tool_agent()
        agent_with_different_types()
        agent_error_handling()
        agent_with_max_iterations()
        
        print("\n💡 智能体的关键概念:")
        print("1. 工具（Tools）：智能体可以使用的功能")
        print("2. 思考-行动循环：智能体的决策过程")
        print("3. 类型：不同的智能体类型适用于不同场景")
        print("4. 错误处理：智能体需要能够处理异常情况")
        
    except ValueError as e:
        print(f"\n❌ 错误: {e}")
        print("\n请确保已设置 MODELVERSE_API_KEY")
    
    print("\n✅ 所有示例运行完成!")


