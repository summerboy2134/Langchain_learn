"""
三种提示模板的简洁可运行示例（仅使用 ModelVerse / DeepSeek-R1）
"""
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    get_modelverse_api_key,
    get_modelverse_api_base,
    get_modelverse_model
)


def example_1_basic_prompt():
    """示例 1: 基础 PromptTemplate"""
    print("=" * 60)
    print("示例 1: 基础提示模板 (PromptTemplate)")
    print("=" * 60)
    
    template = "用一句话解释{concept}"
    prompt = PromptTemplate(input_variables=["concept"], template=template)
    result = prompt.format(concept="人工智能")
    
    print("\n模板定义:")
    print('  "用一句话解释{concept}"')
    print("\n格式化结果:")
    print(f"  {result}")
    print("\n说明: 简单的字符串模板，支持变量替换")


def example_2_chat_prompt():
    """示例 2: 聊天 ChatPromptTemplate"""
    print("\n" + "=" * 60)
    print("示例 2: 聊天提示模板 (ChatPromptTemplate)")
    print("=" * 60)
    
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}"),
        ("human", "{question}")
    ])
    
    messages = chat_template.format_messages(
        role="Python专家",
        question="什么是列表?"
    )
    
    print("\n模板定义:")
    print('  System: "你是一个{role}"')
    print('  Human: "{question}"')
    print("\n格式化结果:")
    for msg in messages:
        print(f"  {msg.type}: {msg.content}")
    print("\n说明: 结构化消息格式，支持 system/human/ai 角色")


def example_3_few_shot_prompt():
    """示例 3: 少样本 FewShotPromptTemplate"""
    print("\n" + "=" * 60)
    print("示例 3: 少样本提示模板 (FewShotPromptTemplate)")
    print("=" * 60)
    
    examples = [
        {"输入": "开心", "输出": "😊"},
        {"输入": "悲伤", "输出": "😢"}
    ]
    
    example_prompt = PromptTemplate(
        input_variables=["输入", "输出"],
        template="输入: {输入}\n输出: {输出}"
    )
    
    few_shot = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="将情绪词转为表情:",
        suffix="输入: {input}\n输出:",
        input_variables=["input"]
    )
    
    result = few_shot.format(input="兴奋")
    
    print("\n示例数据:")
    print("  开心 -> 😊")
    print("  悲伤 -> 😢")
    print("\n格式化结果:")
    print(result)
    print("\n说明: 包含示例的模板，让模型学习特定模式")


def example_4_basic_with_llm():
    """示例 4: 基础 PromptTemplate + LLM (DeepSeek-R1)"""
    print("\n" + "=" * 60)
    print("示例 4: 基础提示模板 + LLM (DeepSeek-R1)")
    print("=" * 60)
    
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        template = "用一句话解释{concept}"
        prompt = PromptTemplate(input_variables=["concept"], template=template)
        
        chain = prompt | llm
        
        print("\n模板:")
        print('  "用一句话解释{concept}"')
        print("\n输入: concept='人工智能'")
        print("\n输出: ", end="", flush=True)
        
        _ = chain.invoke({"concept": "人工智能"})
        print("\n")
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("提示: 需要配置 MODELVERSE_API_KEY")


def example_5_chat_with_llm():
    """示例 5: 聊天 ChatPromptTemplate + LLM (DeepSeek-R1)"""
    print("\n" + "=" * 60)
    print("示例 5: 聊天提示模板 + LLM (DeepSeek-R1)")
    print("=" * 60)
    
    try:
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        chat_template = ChatPromptTemplate.from_messages([
            ("system", "你是一个{role}"),
            ("human", "{question}")
        ])
        
        messages = chat_template.format_messages(
            role="Python专家",
            question="什么是列表?"
        )
        
        print("\n模板:")
        print('  System: "你是一个{role}"')
        print('  Human: "{question}"')
        print("\n输入:")
        print(f"  role='Python专家'")
        print(f"  question='什么是列表?'")
        print("\n输出: ", end="", flush=True)
        
        _ = llm.invoke(messages)
        print("\n")
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("提示: 需要配置 MODELVERSE_API_KEY")


def example_6_few_shot_with_llm():
    """示例 6: 少样本 FewShotPromptTemplate + LLM (DeepSeek-R1)"""
    print("\n" + "=" * 60)
    print("示例 6: 少样本提示模板 + LLM (DeepSeek-R1)")
    print("=" * 60)
    
    try:
        llm = ChatOpenAI(
            temperature=0.3,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        examples = [
            {"输入": "开心", "输出": "😊"},
            {"输入": "悲伤", "输出": "😢"}
        ]
        
        example_prompt = PromptTemplate(
            input_variables=["输入", "输出"],
            template="输入: {输入}\n输出: {输出}"
        )
        
        few_shot = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="将情绪词转为表情:",
            suffix="输入: {input}\n输出:",
            input_variables=["input"]
        )
        
        chain = few_shot | llm
        
        print("\n示例:")
        print("  开心 -> 😊")
        print("  悲伤 -> 😢")
        print("\n输入: '兴奋'")
        print("\n输出: ", end="", flush=True)
        
        _ = chain.invoke({"input": "兴奋"})
        print("\n")
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("提示: 需要配置 MODELVERSE_API_KEY")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LangChain 三种提示模板示例")
    print("=" * 60)
    
    # 基础示例（不需要 API）
    example_1_basic_prompt()
    example_2_chat_prompt()
    example_3_few_shot_prompt()
    
    # 需要 API 的示例
    print("\n" + "=" * 60)
    print("以下示例需要 ModelVerse API (DeepSeek-R1)")
    print("=" * 60)
    
    example_4_basic_with_llm()
    example_5_chat_with_llm()
    example_6_few_shot_with_llm()
    
    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成!")
    print("=" * 60)

"""
LLM 简洁示例（LangChain 1.0 写法）
"""
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    get_modelverse_api_key,
    get_modelverse_api_base,
    get_modelverse_model
)


def example_1_basic_prompt():
    """示例 1: 基础提示模板"""
    print("=" * 60)
    print("示例 1: 基础提示模板 (PromptTemplate)")
    print("=" * 60)
    
    # 创建模板
    template = "用一句话解释{concept}"
    prompt = PromptTemplate(input_variables=["concept"], template=template)
    
    # 格式化提示
    result = prompt.format(concept="人工智能")
    
    print("\n模板定义:")
    print(f'  template = "用一句话解释{{concept}}"')
    print("\n格式化结果:")
    print(f"  {result}")
    print("\n说明: 简单的字符串模板，支持变量替换")


def example_2_chat_prompt():
    """示例 2: 聊天提示模板"""
    print("\n" + "=" * 60)
    print("示例 2: 聊天提示模板 (ChatPromptTemplate)")
    print("=" * 60)
    
    # 创建聊天模板
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}"),
        ("human", "{question}")
    ])
    
    # 格式化消息
    messages = chat_template.format_messages(
        role="Python专家",
        question="什么是列表?"
    )
    
    print("\n模板定义:")
    print('  ChatPromptTemplate.from_messages([')
    print('      ("system", "你是一个{role}"),')
    print('      ("human", "{question}")')
    print('  ])')
    print("\n格式化结果:")
    for msg in messages:
        print(f"  {msg.type}: {msg.content}")
    print("\n说明: 结构化消息格式，支持 system/human/ai 角色")


def example_3_few_shot_prompt():
    """示例 3: 少样本提示模板"""
    print("\n" + "=" * 60)
    print("示例 3: 少样本提示模板 (FewShotPromptTemplate)")
    print("=" * 60)
    
    # 定义示例
    examples = [
        {"输入": "开心", "输出": "😊"},
        {"输入": "悲伤", "输出": "😢"}
    ]
    
    # 创建示例模板
    example_prompt = PromptTemplate(
        input_variables=["输入", "输出"],
        template="输入: {输入}\n输出: {输出}"
    )
    
    # 创建少样本模板
    few_shot = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="将情绪词转为表情:",
        suffix="输入: {input}\n输出:",
        input_variables=["input"]
    )
    
    # 格式化
    result = few_shot.format(input="兴奋")
    
    print("\n示例数据:")
    print("  examples = [")
    print('    {"输入": "开心", "输出": "😊"},')
    print('    {"输入": "悲伤", "输出": "😢"}')
    print("  ]")
    print("\n格式化结果:")
    print(result)
    print("\n说明: 包含示例的模板，让模型学习特定模式")


def example_4_basic_with_llm():
    """示例 4: 基础提示模板 + LLM（流式输出）"""
    print("\n" + "=" * 60)
    print("示例 4: 基础提示模板 + LLM (DeepSeek-R1) - 流式输出")
    print("=" * 60)

    try:
        # 初始化 LLM（启用流式输出）
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )

        # 创建提示模板
        template = "用一句话解释{concept}"
        prompt = PromptTemplate(input_variables=["concept"], template=template)

        # 创建链并调用（流式输出）
        chain = prompt | llm

        print("\n模板:")
        print('  "用一句话解释{concept}"')
        print("\n输入: concept='人工智能'")
        print("\n输出: ", end="", flush=True)

        result = chain.invoke({"concept": "人工智能"})
        print("\n")  # 换行

    except Exception as e:
        print(f"\n错误: {e}")
        print("提示: 需要配置 MODELVERSE_API_KEY")


def example_5_chat_with_llm():
    """示例 5: 聊天提示模板 + LLM（流式输出）"""
    print("\n" + "=" * 60)
    print("示例 5: 聊天提示模板 + LLM (DeepSeek-R1) - 流式输出")
    print("=" * 60)

    try:
        # 初始化 LLM（启用流式输出）
        llm = ChatOpenAI(
            temperature=0.7,
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )

        # 创建聊天模板
        chat_template = ChatPromptTemplate.from_messages([
            ("system", "你是一个{role}"),
            ("human", "{question}")
        ])

        # 格式化消息
        messages = chat_template.format_messages(
            role="Python专家",
            question="什么是列表?"
        )

        print("\n模板:")
        print('  System: "你是一个{role}"')
        print('  Human: "{question}"')
        print("\n输入:")
        print(f"  role='Python专家'")
        print(f"  question='什么是列表?'")
        print("\n输出: ", end="", flush=True)

        result = llm.invoke(messages)
        print("\n")  # 换行

    except Exception as e:
        print(f"\n错误: {e}")
        print("提示: 需要配置 MODELVERSE_API_KEY")


def example_6_few_shot_with_llm():
    """示例 6: 少样本提示模板 + LLM（流式输出）"""
    print("\n" + "=" * 60)
    print("示例 6: 少样本提示模板 + LLM (DeepSeek-R1) - 流式输出")
    print("=" * 60)
    
    try:
        # 初始化 LLM（启用流式输出）
        llm = ChatOpenAI(
            temperature=0.3,  # 少样本学习使用较低温度
            model=get_modelverse_model(),
            openai_api_key=get_modelverse_api_key(),
            base_url=get_modelverse_api_base(),
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # 定义示例
        examples = [
            {"输入": "开心", "输出": "😊"},
            {"输入": "悲伤", "输出": "😢"}
        ]
        
        # 创建少样本模板
        example_prompt = PromptTemplate(
            input_variables=["输入", "输出"],
            template="输入: {输入}\n输出: {输出}"
        )
        
        few_shot = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="将情绪词转为表情:",
            suffix="输入: {input}\n输出:",
            input_variables=["input"]
        )
        
        # 创建链并调用（流式输出）
        chain = few_shot | llm
        
        print("\n示例:")
        print("  开心 -> 😊")
        print("  悲伤 -> 😢")
        print("\n输入: '兴奋'")
        print("\n输出: ", end="", flush=True)
        
        result = chain.invoke({"input": "兴奋"})
        print("\n")  # 换行
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("提示: 需要配置 MODELVERSE_API_KEY")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LangChain 三种提示模板示例")
    print("=" * 60)
    
    # 基础示例（不需要 API）
    example_1_basic_prompt()
    example_2_chat_prompt()
    example_3_few_shot_prompt()
    
    # 需要 API 的示例
    print("\n" + "=" * 60)
    print("以下示例需要 ModelVerse API (DeepSeek-R1)")
    print("=" * 60)
    
    example_4_basic_with_llm()
    example_5_chat_with_llm()
    example_6_few_shot_with_llm()
    
    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成!")
    print("=" * 60)

