"""
基础提示模板示例（仅使用 ModelVerse / DeepSeek-R1）

演示：
- 基础 PromptTemplate 使用（不需要 API）
- 问答 PromptTemplate 使用（不需要 API）
- 批量格式化多个输入（不需要 API）
- 与 LLM 结合（使用 ModelVerse DeepSeek-R1）
"""
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import sys
import os

# 添加父目录到路径，以便导入 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    get_modelverse_api_key,
    get_modelverse_api_base,
    get_modelverse_model
)


def basic_prompt_example():
    """示例 1: 基础字符串提示模板"""
    print("=" * 50)
    print("示例 1: 基础字符串提示模板")
    print("=" * 50)
    
    # 创建一个简单的提示模板
    template = """
    你是一个专业的{profession}。
    请用通俗易懂的方式解释以下概念：{concept}
    """
    
    prompt = PromptTemplate(
        input_variables=["profession", "concept"],
        template=template
    )
    
    # 格式化提示
    formatted_prompt = prompt.format(
        profession="数据科学家",
        concept="机器学习"
    )
    
    print("\n生成的提示:")
    print(formatted_prompt)


def question_answer_template():
    """示例 2: 问答提示模板"""
    print("\n" + "=" * 50)
    print("示例 2: 问答提示模板")
    print("=" * 50)
    
    # 创建问答模板
    qa_template = """
    请根据以下上下文回答问题。如果你不知道答案，就说\"我不知道\"，不要试图编造答案。
    
    上下文: {context}
    
    问题: {question}
    
    答案:
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )
    
    # 示例数据
    context = "LangChain 是一个用于开发由语言模型驱动的应用程序的框架。它提供了模块化的组件来构建复杂的 AI 应用。"
    question = "LangChain 是什么?"
    
    formatted_prompt = prompt.format(context=context, question=question)
    print("\n生成的提示:")
    print(formatted_prompt)


def multiple_examples_template():
    """示例 3: 批量处理多个输入"""
    print("\n" + "=" * 50)
    print("示例 3: 批量处理多个输入")
    print("=" * 50)
    
    # 创建模板
    template = "将以下{language}翻译成英文: {text}"
    prompt = PromptTemplate(
        input_variables=["language", "text"],
        template=template
    )
    
    # 多个示例
    examples = [
        {"language": "中文", "text": "你好"},
        {"language": "中文", "text": "谢谢"},
        {"language": "中文", "text": "再见"},
    ]
    
    print("\n生成的提示:")
    for example in examples:
        formatted = prompt.format(**example)
        print(f"\n{formatted}")


def template_with_llm():
    """示例 4: 与 LLM 结合使用（DeepSeek-R1）"""
    print("\n" + "=" * 50)
    print("示例 4: 与 LLM 结合使用（DeepSeek-R1）")
    print("=" * 50)
    
    # 使用 ModelVerse API（DeepSeek-R1）
    try:
        print("正在连接 ModelVerse API (DeepSeek-R1)...")
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base()
        )
        print("✓ 已连接到 ModelVerse API (DeepSeek-R1)")
    except ValueError as e:
        print(f"\n错误: {e}")
        print("提示: 请确保在 .env 中设置 MODELVERSE_API_KEY，或在环境变量中导出：")
        print("export MODELVERSE_API_KEY=<YOUR_API_KEY>")
        return
    except Exception as e:
        print(f"\n连接失败: {e}")
        print("请检查: 1) API Key 2) 网络 3) 配额")
        return
    
    try:
        # 创建提示模板
        template = "用一句话描述 {topic} 的主要特点。"
        prompt = PromptTemplate(
            input_variables=["topic"],
            template=template
        )
        
        # 新语法：prompt | llm，然后使用 invoke
        chain = prompt | llm
        
        # 运行
        topic = "人工智能"
        result = chain.invoke({"topic": topic})
        
        print(f"\n主题: {topic}")
        print(f"回答: {result.content if hasattr(result, 'content') else result}")
        
    except Exception as e:
        print(f"\n发生错误: {e}")


if __name__ == "__main__":
    print("\n🚀 LangChain 基础提示模板示例\n")
    
    # 不需要 API 的示例
    basic_prompt_example()
    question_answer_template()
    multiple_examples_template()
    
    # 需要 API 的示例
    print("\n" + "=" * 50)
    print("注意: 下一个示例需要有效的 ModelVerse API 密钥（DeepSeek-R1）")
    print("=" * 50)
    template_with_llm()
    
    print("\n✅ 所有示例运行完成!")


