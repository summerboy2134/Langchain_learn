"""
工具集成智能体示例

演示如何创建和使用各种工具，构建更强大的智能体
"""
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.tools import StructuredTool
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    get_modelverse_api_key,
    get_modelverse_api_base,
    get_modelverse_model,
)


def search_tool_example():
    """搜索工具示例"""
    print("=" * 50)
    print("示例 1: 模拟搜索工具")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 模拟搜索数据库
        knowledge_base = {
            "langchain": "LangChain 是一个用于开发由语言模型驱动的应用程序的框架。",
            "python": "Python 是一种高级、解释型、通用的编程语言。",
            "openai": "OpenAI 是一个 AI 研究实验室，创建了 GPT 系列模型。",
            "机器学习": "机器学习是人工智能的一个分支，让计算机能够从数据中学习。",
        }
        
        def search_knowledge(query: str) -> str:
            """在知识库中搜索信息"""
            query = query.lower().strip()
            for key, value in knowledge_base.items():
                if key in query or query in key:
                    return f"找到相关信息：{value}"
            return "抱歉，没有找到相关信息。"
        
        tools = [
            Tool(
                name="KnowledgeSearch",
                func=search_knowledge,
                description="在知识库中搜索信息。输入应该是搜索关键词。"
            ),
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        questions = [
            "LangChain 是什么？",
            "告诉我关于 Python 的信息"
        ]
        
        for question in questions:
            print(f"\n问题: {question}")
            result = agent.invoke({"input": question})
            print(f"答案: {result.get('output', result)}\n")
        
    except Exception as e:
        print(f"错误: {e}")


def structured_tool_example():
    """结构化工具示例"""
    print("\n" + "=" * 50)
    print("示例 2: 结构化工具（多参数）")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 定义多参数函数
        def calculate_rectangle_area(length: float, width: float) -> str:
            """计算矩形面积"""
            area = length * width
            return f"长度 {length} × 宽度 {width} = 面积 {area}"
        
        def calculate_circle_area(radius: float) -> str:
            """计算圆形面积"""
            import math
            area = math.pi * radius ** 2
            return f"半径 {radius} 的圆形面积 = {area:.2f}"
        
        # 使用 StructuredTool 创建工具
        tools = [
            StructuredTool.from_function(
                func=calculate_rectangle_area,
                name="RectangleArea",
                description="计算矩形的面积。需要提供长度和宽度两个参数。"
            ),
            StructuredTool.from_function(
                func=calculate_circle_area,
                name="CircleArea",
                description="计算圆形的面积。需要提供半径参数。"
            ),
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        questions = [
            "计算一个长 5 宽 3 的矩形的面积",
            "计算半径为 4 的圆的面积"
        ]
        
        for question in questions:
            print(f"\n问题: {question}")
            result = agent.invoke({"input": question})
            print(f"答案: {result.get('output', result)}\n")
        
    except Exception as e:
        print(f"错误: {e}")


def file_operations_tools():
    """文件操作工具示例"""
    print("\n" + "=" * 50)
    print("示例 3: 文件操作工具")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 模拟文件系统
        file_system = {
            "notes.txt": "这是我的笔记文件",
            "todo.txt": "1. 学习 LangChain\n2. 练习 Python\n3. 完成项目",
            "data.txt": "一些数据内容"
        }
        
        def list_files() -> str:
            """列出所有文件"""
            files = list(file_system.keys())
            return f"可用文件: {', '.join(files)}"
        
        def read_file(filename: str) -> str:
            """读取文件内容"""
            if filename in file_system:
                return f"文件 '{filename}' 的内容：\n{file_system[filename]}"
            return f"错误：文件 '{filename}' 不存在"
        
        def file_exists(filename: str) -> str:
            """检查文件是否存在"""
            exists = filename in file_system
            return f"文件 '{filename}' {'存在' if exists else '不存在'}"
        
        tools = [
            Tool(
                name="ListFiles",
                func=list_files,
                description="列出所有可用的文件。不需要输入参数。"
            ),
            Tool(
                name="ReadFile",
                func=read_file,
                description="读取指定文件的内容。输入应该是文件名。"
            ),
            Tool(
                name="FileExists",
                func=file_exists,
                description="检查文件是否存在。输入应该是文件名。"
            ),
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        questions = [
            "有哪些文件？",
            "读取 todo.txt 文件的内容",
            "检查是否存在 report.txt 文件"
        ]
        
        for question in questions:
            print(f"\n问题: {question}")
            result = agent.invoke({"input": question})
            print(f"答案: {result.get('output', result)}\n")
        
    except Exception as e:
        print(f"错误: {e}")


def data_processing_tools():
    """数据处理工具示例"""
    print("\n" + "=" * 50)
    print("示例 4: 数据处理工具")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        def calculate_average(numbers: str) -> str:
            """计算平均值"""
            try:
                nums = [float(n.strip()) for n in numbers.split(',')]
                avg = sum(nums) / len(nums)
                return f"数字 {numbers} 的平均值是 {avg:.2f}"
            except:
                return "错误：请提供用逗号分隔的数字"
        
        def find_maximum(numbers: str) -> str:
            """找出最大值"""
            try:
                nums = [float(n.strip()) for n in numbers.split(',')]
                maximum = max(nums)
                return f"数字 {numbers} 中的最大值是 {maximum}"
            except:
                return "错误：请提供用逗号分隔的数字"
        
        def count_words(text: str) -> str:
            """统计单词数量"""
            word_count = len(text.split())
            return f"文本有 {word_count} 个单词"
        
        tools = [
            Tool(
                name="CalculateAverage",
                func=calculate_average,
                description="计算一组数字的平均值。输入应该是用逗号分隔的数字，例如：'1,2,3,4,5'"
            ),
            Tool(
                name="FindMaximum",
                func=find_maximum,
                description="找出一组数字中的最大值。输入应该是用逗号分隔的数字。"
            ),
            Tool(
                name="CountWords",
                func=count_words,
                description="统计文本中的单词数量。输入应该是一段文本。"
            ),
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        questions = [
            "计算 10, 20, 30, 40, 50 的平均值",
            "找出 5, 15, 8, 23, 12 中的最大值",
            "这句话有多少个单词：LangChain is a powerful framework"
        ]
        
        for question in questions:
            print(f"\n问题: {question}")
            result = agent.invoke({"input": question})
            print(f"答案: {result.get('output', result)}\n")
        
    except Exception as e:
        print(f"错误: {e}")


def weather_tools():
    """天气工具示例（模拟）"""
    print("\n" + "=" * 50)
    print("示例 5: 天气查询工具")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 模拟天气数据
        weather_data = {
            "北京": {"温度": "15°C", "天气": "晴天", "湿度": "45%"},
            "上海": {"温度": "18°C", "天气": "多云", "湿度": "60%"},
            "广州": {"温度": "25°C", "天气": "阴天", "湿度": "75%"},
            "深圳": {"温度": "26°C", "天气": "晴天", "湿度": "70%"},
        }
        
        def get_weather(city: str) -> str:
            """获取城市天气"""
            city = city.strip()
            if city in weather_data:
                data = weather_data[city]
                return f"{city}的天气：{data['天气']}，温度 {data['温度']}，湿度 {data['湿度']}"
            return f"抱歉，没有 {city} 的天气信息"
        
        def get_temperature(city: str) -> str:
            """获取城市温度"""
            city = city.strip()
            if city in weather_data:
                return f"{city}的温度是 {weather_data[city]['温度']}"
            return f"抱歉，没有 {city} 的温度信息"
        
        def compare_temperature(input_str: str) -> str:
            """比较两个城市的温度"""
            try:
                cities = [c.strip() for c in input_str.split(',')]
                if len(cities) != 2:
                    return "请提供两个城市，用逗号分隔"
                
                city1, city2 = cities
                if city1 in weather_data and city2 in weather_data:
                    temp1 = int(weather_data[city1]['温度'].replace('°C', ''))
                    temp2 = int(weather_data[city2]['温度'].replace('°C', ''))
                    
                    if temp1 > temp2:
                        return f"{city1} ({weather_data[city1]['温度']}) 比 {city2} ({weather_data[city2]['温度']}) 更热"
                    elif temp1 < temp2:
                        return f"{city2} ({weather_data[city2]['温度']}) 比 {city1} ({weather_data[city1]['温度']}) 更热"
                    else:
                        return f"{city1} 和 {city2} 温度相同，都是 {weather_data[city1]['温度']}"
                
                return "无法获取城市温度信息"
            except:
                return "错误：请提供有效的城市名称"
        
        tools = [
            Tool(
                name="GetWeather",
                func=get_weather,
                description="获取指定城市的完整天气信息。输入应该是城市名称。"
            ),
            Tool(
                name="GetTemperature",
                func=get_temperature,
                description="获取指定城市的温度。输入应该是城市名称。"
            ),
            Tool(
                name="CompareTemperature",
                func=compare_temperature,
                description="比较两个城市的温度。输入应该是两个城市，用逗号分隔，例如：'北京,上海'"
            ),
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        questions = [
            "北京今天天气怎么样？",
            "上海的温度是多少？",
            "北京和广州哪个城市更热？"
        ]
        
        for question in questions:
            print(f"\n问题: {question}")
            result = agent.invoke({"input": question})
            print(f"答案: {result.get('output', result)}\n")
        
    except Exception as e:
        print(f"错误: {e}")


def complex_tool_chain():
    """复杂工具链示例"""
    print("\n" + "=" * 50)
    print("示例 6: 复杂工具链（多步骤）")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 模拟数据库
        users = {
            "user1": {"name": "张三", "age": 25, "city": "北京"},
            "user2": {"name": "李四", "age": 30, "city": "上海"},
            "user3": {"name": "王五", "age": 28, "city": "北京"},
        }
        
        def get_user_info(user_id: str) -> str:
            """获取用户信息"""
            if user_id in users:
                user = users[user_id]
                return f"用户 {user_id}：姓名 {user['name']}，年龄 {user['age']}，城市 {user['city']}"
            return f"用户 {user_id} 不存在"
        
        def get_user_city(user_id: str) -> str:
            """获取用户所在城市"""
            if user_id in users:
                return users[user_id]['city']
            return "未知"
        
        def count_users_in_city(city: str) -> str:
            """统计城市中的用户数量"""
            count = sum(1 for u in users.values() if u['city'] == city)
            return f"{city} 有 {count} 个用户"
        
        tools = [
            Tool(
                name="GetUserInfo",
                func=get_user_info,
                description="获取用户的详细信息。输入应该是用户ID，例如：'user1'"
            ),
            Tool(
                name="GetUserCity",
                func=get_user_city,
                description="获取用户所在的城市。输入应该是用户ID。"
            ),
            Tool(
                name="CountUsersInCity",
                func=count_users_in_city,
                description="统计指定城市的用户数量。输入应该是城市名称。"
            ),
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=5
        )
        
        # 需要多步骤推理的问题
        question = "user1 在哪个城市？那个城市总共有多少用户？"
        
        print(f"\n复杂问题: {question}")
        print("（这个问题需要智能体先查询用户城市，然后统计该城市的用户数）\n")
        
        result = agent.invoke({"input": question})
        print(f"\n最终答案: {result if isinstance(result, str) else result.get('output', result)}")
        
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    print("\n🛠️  LangChain 工具集成智能体示例\n")
    
    try:
        # 测试 API 密钥
        api_key = get_modelverse_api_key()
        print(f"✓ API 密钥已加载\n")
        
        # 运行各个示例
        search_tool_example()
        structured_tool_example()
        file_operations_tools()
        data_processing_tools()
        weather_tools()
        complex_tool_chain()
        
        print("\n💡 工具设计的最佳实践:")
        print("1. 明确的功能描述 - 让智能体知道何时使用工具")
        print("2. 清晰的输入格式 - 说明工具期望的输入")
        print("3. 有用的返回信息 - 提供足够的信息供后续决策")
        print("4. 错误处理 - 优雅地处理无效输入")
        print("5. 单一职责 - 每个工具专注于一个功能")
        
    except ValueError as e:
        print(f"\n❌ 错误: {e}")
        print("\n请确保已设置 MODELVERSE_API_KEY")
    
    print("\n✅ 所有示例运行完成!")


