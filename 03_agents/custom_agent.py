"""
自定义智能体示例

演示如何创建自定义工具和更高级的智能体应用
"""
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, Tool, AgentExecutor
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    get_modelverse_api_key,
    get_modelverse_api_base,
    get_modelverse_model,
)


# 定义输入模式
class CalculatorInput(BaseModel):
    """计算器输入模式"""
    expression: str = Field(description="要计算的数学表达式，例如：'2 + 2'")


class CustomCalculator(BaseTool):
    """自定义计算器工具"""
    name = "Calculator"
    description = "用于执行数学计算。输入应该是一个数学表达式。"
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        """执行计算"""
        try:
            # 安全地执行数学表达式
            result = eval(expression, {"__builtins__": {}}, {})
            return f"{expression} = {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        """异步执行（这里我们只是调用同步版本）"""
        return self._run(expression)


def custom_tool_class():
    """使用自定义工具类"""
    print("=" * 50)
    print("示例 1: 自定义工具类")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 使用自定义工具
        tools = [CustomCalculator()]
        
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        questions = [
            "计算 25 + 17",
            "计算 100 / 4",
            "计算 2 的 8 次方"
        ]
        
        for question in questions:
            print(f"\n问题: {question}")
            result = agent.invoke({"input": question})
            print(f"答案: {result.get('output', result)}\n")
        
    except Exception as e:
        print(f"错误: {e}")


class TextAnalysisInput(BaseModel):
    """文本分析输入"""
    text: str = Field(description="要分析的文本")


class TextAnalysisTool(BaseTool):
    """文本分析工具"""
    name = "TextAnalysis"
    description = "分析文本的各种属性，包括字符数、单词数、句子数等。"
    args_schema: Type[BaseModel] = TextAnalysisInput
    
    def _run(self, text: str) -> str:
        """分析文本"""
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        
        return f"""文本分析结果：
- 字符数：{char_count}
- 单词数：{word_count}
- 句子数：{sentence_count}
- 平均单词长度：{char_count / max(word_count, 1):.2f}"""
    
    async def _arun(self, text: str) -> str:
        return self._run(text)


def advanced_custom_tools():
    """高级自定义工具"""
    print("\n" + "=" * 50)
    print("示例 2: 高级自定义工具")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        tools = [TextAnalysisTool()]
        
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        question = "分析这段文本：LangChain is a framework. It helps developers. It's powerful!"
        
        print(f"\n问题: {question}")
        result = agent.invoke({"input": question})
        print(f"\n答案: {result.get('output', result)}")
        
    except Exception as e:
        print(f"错误: {e}")


def task_planning_agent():
    """任务规划智能体"""
    print("\n" + "=" * 50)
    print("示例 3: 任务规划智能体")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        def create_task_list(goal: str) -> str:
            """创建任务列表"""
            # 这里可以集成更复杂的规划逻辑
            return f"为实现 '{goal}' 创建的任务列表：\n1. 分解目标\n2. 收集资源\n3. 执行计划\n4. 评估结果"
        
        def estimate_time(task: str) -> str:
            """估算任务时间"""
            # 简化的时间估算
            word_count = len(task.split())
            hours = max(1, word_count // 2)
            return f"任务 '{task}' 预计需要 {hours} 小时完成"
        
        def prioritize_tasks(tasks: str) -> str:
            """任务优先级排序"""
            return f"已对任务进行优先级排序：\n1. 高优先级\n2. 中优先级\n3. 低优先级"
        
        tools = [
            Tool(
                name="CreateTaskList",
                func=create_task_list,
                description="为给定目标创建任务列表。输入应该是目标描述。"
            ),
            Tool(
                name="EstimateTime",
                func=estimate_time,
                description="估算任务完成时间。输入应该是任务描述。"
            ),
            Tool(
                name="PrioritizeTasks",
                func=prioritize_tasks,
                description="对任务进行优先级排序。输入应该是任务列表。"
            ),
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        question = "帮我规划一个学习 Python 的任务"
        
        print(f"\n问题: {question}")
        result = agent.invoke({"input": question})
        print(f"\n答案: {result.get('output', result)}")
        
    except Exception as e:
        print(f"错误: {e}")


def decision_making_agent():
    """决策辅助智能体"""
    print("\n" + "=" * 50)
    print("示例 4: 决策辅助智能体")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0.5,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        def analyze_pros(option: str) -> str:
            """分析优点"""
            pros_db = {
                "python": "易学、库丰富、社区活跃、应用广泛",
                "java": "企业级、性能好、稳定可靠、工作机会多",
                "javascript": "前端必备、全栈可用、生态系统大"
            }
            return pros_db.get(option.lower(), "优点：灵活、现代化")
        
        def analyze_cons(option: str) -> str:
            """分析缺点"""
            cons_db = {
                "python": "运行速度较慢、移动开发支持弱",
                "java": "语法繁琐、学习曲线陡",
                "javascript": "版本更新快、类型系统弱"
            }
            return cons_db.get(option.lower(), "缺点：需要持续学习")
        
        def compare_options(options: str) -> str:
            """比较多个选项"""
            opts = [o.strip() for o in options.split(',')]
            return f"比较结果：{opts[0]} 和 {opts[1] if len(opts) > 1 else '其他选项'} 各有优势，需要根据具体需求选择"
        
        tools = [
            Tool(
                name="AnalyzePros",
                func=analyze_pros,
                description="分析选项的优点。输入应该是选项名称。"
            ),
            Tool(
                name="AnalyzeCons",
                func=analyze_cons,
                description="分析选项的缺点。输入应该是选项名称。"
            ),
            Tool(
                name="CompareOptions",
                func=compare_options,
                description="比较多个选项。输入应该是用逗号分隔的选项，例如：'python,java'"
            ),
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        question = "我应该学习 Python 还是 Java？请帮我分析一下。"
        
        print(f"\n问题: {question}")
        result = agent.invoke({"input": question})
        print(f"\n答案: {result.get('output', result)}")
        
    except Exception as e:
        print(f"错误: {e}")


def code_assistant_agent():
    """代码助手智能体"""
    print("\n" + "=" * 50)
    print("示例 5: 代码助手智能体")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        def explain_concept(concept: str) -> str:
            """解释编程概念"""
            concepts = {
                "变量": "变量是用于存储数据值的容器",
                "函数": "函数是执行特定任务的可重用代码块",
                "类": "类是创建对象的蓝图或模板",
                "循环": "循环用于重复执行代码块"
            }
            return concepts.get(concept, f"{concept} 是编程中的重要概念")
        
        def check_syntax(code: str) -> str:
            """检查语法（简化版）"""
            if "print(" in code and ")" in code:
                return "✓ 语法看起来正确"
            return "⚠ 可能存在语法问题"
        
        def suggest_improvement(code: str) -> str:
            """建议改进"""
            suggestions = []
            if "for i in range" in code:
                suggestions.append("可以考虑使用列表推导式")
            if len(code.split('\n')) > 10:
                suggestions.append("考虑将代码分解为多个函数")
            
            if suggestions:
                return "改进建议：\n" + "\n".join(f"- {s}" for s in suggestions)
            return "代码看起来不错！"
        
        tools = [
            Tool(
                name="ExplainConcept",
                func=explain_concept,
                description="解释编程概念。输入应该是概念名称。"
            ),
            Tool(
                name="CheckSyntax",
                func=check_syntax,
                description="检查代码语法。输入应该是代码片段。"
            ),
            Tool(
                name="SuggestImprovement",
                func=suggest_improvement,
                description="建议代码改进。输入应该是代码片段。"
            ),
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        question = "解释一下什么是函数，并检查这段代码：print('Hello World')"
        
        print(f"\n问题: {question}")
        result = agent.invoke({"input": question})
        print(f"\n答案: {result.get('output', result)}")
        
    except Exception as e:
        print(f"错误: {e}")


def multi_agent_collaboration():
    """多智能体协作示例"""
    print("\n" + "=" * 50)
    print("示例 6: 多智能体协作")
    print("=" * 50)
    
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
        )
        
        # 研究助手工具
        def research_topic(topic: str) -> str:
            """研究主题"""
            return f"研究结果：{topic} 是一个重要的技术领域，有广泛的应用前景。"
        
        # 写作助手工具
        def write_summary(content: str) -> str:
            """撰写摘要"""
            return f"基于提供的内容，摘要如下：{content[:50]}... [已总结]"
        
        # 审核助手工具
        def review_content(text: str) -> str:
            """审核内容"""
            word_count = len(text.split())
            return f"审核通过 ✓ (共 {word_count} 个单词，结构清晰，表达准确)"
        
        tools = [
            Tool(
                name="ResearchTopic",
                func=research_topic,
                description="研究指定主题并收集信息。输入应该是主题名称。"
            ),
            Tool(
                name="WriteSummary",
                func=write_summary,
                description="根据内容撰写摘要。输入应该是要总结的内容。"
            ),
            Tool(
                name="ReviewContent",
                func=review_content,
                description="审核和检查内容质量。输入应该是要审核的文本。"
            ),
        ]
        
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=5
        )
        
        question = "请研究'人工智能'这个主题，然后写一个摘要，最后审核内容"
        
        print(f"\n复杂任务: {question}")
        print("（这需要多个工具协作完成）\n")
        
        result = agent.invoke({"input": question})
        print(f"\n最终结果: {result if isinstance(result, str) else result.get('output', result)}")
        
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    print("\n🎨 LangChain 自定义智能体示例\n")
    
    try:
        # 测试 API 密钥
        api_key = get_modelverse_api_key()
        print(f"✓ API 密钥已加载\n")
        
        # 运行各个示例
        custom_tool_class()
        advanced_custom_tools()
        task_planning_agent()
        decision_making_agent()
        code_assistant_agent()
        multi_agent_collaboration()
        
        print("\n💡 自定义智能体的关键要素:")
        print("1. 继承 BaseTool - 创建标准化的工具")
        print("2. 定义输入模式 - 使用 Pydantic 模型")
        print("3. 实现 _run 方法 - 核心功能逻辑")
        print("4. 清晰的描述 - 帮助智能体理解工具用途")
        print("5. 工具组合 - 多个工具协作完成复杂任务")
        
        print("\n💡 实际应用场景:")
        print("- 代码助手：帮助开发者理解和改进代码")
        print("- 决策支持：分析选项并提供建议")
        print("- 任务规划：分解和组织复杂任务")
        print("- 数据分析：处理和解释数据")
        print("- 内容创作：研究、写作、审核的工作流")
        
    except ValueError as e:
        print(f"\n❌ 错误: {e}") 
        print("\n请确保已设置 MODELVERSE_API_KEY")
    
    print("\n✅ 所有示例运行完成!")


