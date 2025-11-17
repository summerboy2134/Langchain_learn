"""
Memory 完整示例（ModelVerse / DeepSeek-R1）

本文件展示了 LangChain 中五种常用的 Memory 类型：
1. BufferMemory（完整缓冲记忆）- 保留所有对话历史
2. WindowMemory（滑动窗口记忆）- 只保留最近 K 轮对话
3. SummaryMemory（摘要记忆）- 对历史进行智能摘要
4. SummaryBufferMemory（混合摘要记忆）- 最近完整 + 早期摘要
5. TokenBufferMemory（Token 限制记忆）- 严格限制 Token 数量

适合快速上手和理解不同 Memory 类型的特点。
详细对比请参考：04_memory/Memory对比说明.md
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
from langchain.schema import BaseMemory, AIMessage, HumanMessage
from pydantic import Field
from typing import Any, Dict, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    get_modelverse_api_key,
    get_modelverse_api_base,
    get_modelverse_model,
)


def example_1_buffer():
    """
    示例 1：ConversationBufferMemory（完整缓冲记忆）
    
    特点：
    - 保留所有对话历史，不做任何删减
    - 上下文最完整，适合短对话（5-10 轮）
    - Token 消耗随对话轮数线性增长
    
    适用场景：演示、短对话、需要完整上下文的场景
    """
    print("=" * 60)
    print("示例 1：BufferMemory（保存所有对话）")
    print("=" * 60)

    # 初始化 LLM（使用 ModelVerse 的 DeepSeek-R1）
    llm = ChatOpenAI(
        model=get_modelverse_model(),
        openai_api_key=get_modelverse_api_key(),
        base_url=get_modelverse_api_base(),
        temperature=0.7,
    )

    # 创建 BufferMemory：保留完整对话历史
    memory = ConversationBufferMemory()
    conv = ConversationChain(llm=llm, memory=memory, verbose=False)

    # 第一轮对话：告诉 AI 自己的名字
    conv.predict(input="你好，我叫小明")
    
    # 第二轮对话：告诉 AI 自己的爱好
    conv.predict(input="我喜欢 Python")
    
    # 第三轮对话：询问 AI 是否记得名字
    res = conv.predict(input="我叫什么名字？")
    print("回答：", res)
    print("\n✅ BufferMemory 能记住所有历史信息（名字和爱好）")


def example_2_window():
    """
    示例 2：ConversationBufferWindowMemory（滑动窗口记忆）
    
    特点：
    - 只保留最近 k 轮对话（本例中 k=2）
    - Token 消耗固定，适合中长对话（10-50 轮）
    - 早期信息会被自动丢弃
    
    适用场景：长时间运行的对话系统、只需近期上下文的场景
    """
    print("\n" + "=" * 60)
    print("示例 2：WindowMemory（只保留最近 2 轮）")
    print("=" * 60)

    llm = ChatOpenAI(
        model=get_modelverse_model(),
        openai_api_key=get_modelverse_api_key(),
        base_url=get_modelverse_api_base(),
        temperature=0.7,
    )

    # 创建 WindowMemory：k=2 表示只保留最近 2 轮对话
    # 1 轮 = 1 次用户输入 + 1 次 AI 回复
    memory = ConversationBufferWindowMemory(k=2)
    conv = ConversationChain(llm=llm, memory=memory, verbose=False)

    # 第 1 轮：告诉 AI 名字
    conv.predict(input="我叫王五")
    
    # 第 2 轮：告诉 AI 所在城市
    conv.predict(input="我在北京")
    
    # 第 3 轮：告诉 AI 爱好（此时第 1 轮"我叫王五"已超出窗口）
    conv.predict(input="我喜欢编程")
    
    # 第 4 轮：询问名字（第 1 轮信息已丢失）
    res = conv.predict(input="我叫什么名字？")
    print("回答：", res)



def example_3_summary():
    """
    示例 3：ConversationSummaryMemory（摘要记忆）
    
    特点：
    - 使用 LLM 对对话历史进行智能摘要
    - 大幅节省 Token，适合长对话（50+ 轮）
    - 保留关键信息，但可能丢失细节
    
    适用场景：长对话、需要长期记忆但要控制成本的场景
    注意：摘要过程本身需要额外 LLM 调用
    """
    print("\n" + "=" * 60)
    print("示例 3：SummaryMemory（自动摘要历史）")
    print("=" * 60)

    llm = ChatOpenAI(
        model=get_modelverse_model(),
        openai_api_key=get_modelverse_api_key(),
        base_url=get_modelverse_api_base(),
        temperature=0.7,
    )

    # 创建 SummaryMemory：使用 LLM 对历史进行摘要
    # 注意：需要传入 llm 参数用于生成摘要
    memory = ConversationSummaryMemory(llm=llm)
    conv = ConversationChain(llm=llm, memory=memory, verbose=False)

    # 进行多轮对话，告诉 AI 个人背景信息
    conv.predict(input="我是一名数据科学家")
    conv.predict(input="我在金融行业工作")
    conv.predict(input="我擅长 Python 和机器学习")
    
    # 查看 SummaryMemory 生成的摘要
    # 摘要会提取关键信息：职业、行业、技能
    print("\n📝 当前摘要（由 LLM 生成）：")
    print(memory.load_memory_variables({})["history"])

    # 询问 AI，看它能否基于摘要回答问题
    res = conv.predict(input="我从事哪个行业？")
    print("\n❓ 问答：我从事哪个行业？")
    print("💬 回答：", res)
    print("\n✅ SummaryMemory 基于摘要保留了关键信息（金融行业）")
    print("💡 Token 消耗远低于保留完整历史，适合长对话场景")


def example_4_summary_buffer():
    """
    示例 4：自定义 SummaryBufferMemory（混合摘要记忆）
    
    特点：
    - 最近 K 轮对话保留完整原文
    - 早期对话自动进行智能摘要
    - 兼容所有 LLM（包括 DeepSeek），不依赖 tiktoken
    - 可按字符数或轮数限制
    
    优点：
    ✅ 平衡了上下文质量和成本
    ✅ 最近对话保持完整细节
    ✅ 早期信息通过摘要保留关键点
    ✅ 完全兼容非 OpenAI 模型
    
    缺点：
    ❌ 需要额外的 LLM 调用生成摘要
    ❌ 摘要可能丢失部分细节
    
    适用场景：
    - 生产环境的首选方案（推荐⭐）
    - 需要长期记忆但要控制成本
    - 中长对话（30-100 轮）
    """
    print("\n" + "=" * 60)
    print("示例 4：SummaryBufferMemory（混合摘要记忆）")
    print("=" * 60)
    
    llm = ChatOpenAI(
        model=get_modelverse_model(),
        openai_api_key=get_modelverse_api_key(),
        base_url=get_modelverse_api_base(),
        temperature=0.7,
    )
    
    # 定义自定义 SummaryBufferMemory（兼容所有 LLM）
    class CustomSummaryBufferMemory(BaseMemory):
        """
        自定义混合摘要记忆（兼容所有 LLM，包括 DeepSeek）
        
        功能：
        - 按字符数限制原文区大小（可选）
        - 早期对话自动进行摘要
        - 不依赖官方 token 计数，使用字符数近似
        
        工作原理：
        - 从最新往前累计字符数，直到达到 max_chars
        - 超出部分的对话移入摘要区，由 LLM 生成摘要
        - 最终上下文 = 早期摘要 + 最近完整对话
        """
        
        llm: Any = Field(description="用于生成摘要的 LLM")
        max_chars: int = Field(default=0, description="原文区最大字符数（0表示按轮数）")
        keep_recent_turns: int = Field(default=3, description="保留最近几轮（max_chars=0时生效）")
        memory_key: str = Field(default="history", description="记忆在 prompt 中的键名")
        recent_messages: List[tuple] = Field(default_factory=list, description="最近的完整对话")
        summary: str = Field(default="", description="早期对话的摘要")
        
        model_config = {"arbitrary_types_allowed": True}
        
        @property
        def memory_variables(self) -> List[str]:
            return [self.memory_key]
        
        def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """加载记忆内容"""
            context = ""
            if self.summary:
                context += f"早期对话摘要：\n{self.summary}\n\n"
            if self.recent_messages:
                context += "最近对话：\n"
                for human_msg, ai_msg in self.recent_messages:
                    context += f"Human: {human_msg}\nAI: {ai_msg}\n"
            return {self.memory_key: context.strip()}
        
        def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
            """保存对话上下文"""
            human_msg = inputs.get("input", "")
            ai_msg = outputs.get("response", outputs.get("output", outputs.get("text", "")))
            self.recent_messages.append((human_msg, ai_msg))
            
            # 检查是否需要移出早期对话
            if self.max_chars > 0:
                # 按字符数限制
                while self._count_recent_chars() > self.max_chars and len(self.recent_messages) > 1:
                    old_human, old_ai = self.recent_messages.pop(0)
                    self._update_summary(old_human, old_ai)
            else:
                # 按轮数限制
                if len(self.recent_messages) > self.keep_recent_turns:
                    old_human, old_ai = self.recent_messages.pop(0)
                    self._update_summary(old_human, old_ai)
        
        def _count_recent_chars(self) -> int:
            """计算最近对话的字符总数"""
            total = 0
            for human_msg, ai_msg in self.recent_messages:
                total += len(human_msg) + len(ai_msg)
            return total
        
        def clear(self) -> None:
            """清空记忆"""
            self.recent_messages = []
            self.summary = ""
        
        def _update_summary(self, human_msg: str, ai_msg: str) -> None:
            """更新摘要（增量式）"""
            if not self.summary:
                # 首次生成摘要
                prompt = f"请简洁地总结以下对话的关键信息：\nHuman: {human_msg}\nAI: {ai_msg}\n\n总结："
                self.summary = self.llm.predict(prompt).strip()
            else:
                # 增量更新摘要
                prompt = f"当前摘要：{self.summary}\n\n新增对话：\nHuman: {human_msg}\nAI: {ai_msg}\n\n请更新摘要（保持简洁）："
                self.summary = self.llm.predict(prompt).strip()
    
    # 使用自定义 SummaryBufferMemory（按轮数限制，保留最近 3 轮）
    print("\n💡 配置：保留最近 3 轮完整对话，早期对话自动摘要")
    memory = CustomSummaryBufferMemory(llm=llm, keep_recent_turns=3)
    conv = ConversationChain(llm=llm, memory=memory, verbose=False)
    
    # 进行 6 轮对话
    print("\n📝 进行 6 轮对话...")
    conversations = [
        "你好，我叫张三",
        "我在上海工作",
        "我是产品经理",
        "我喜欢阅读和旅行",
        "我最近在学习 AI",
        "我想做一个智能助手",
    ]
    
    for i, user_input in enumerate(conversations, 1):
        print(f"第 {i} 轮：{user_input[:20]}... ", end="")
        conv.predict(input=user_input)
        print("✓")
    
    # 查看当前记忆状态
    print("\n" + "=" * 60)
    print("📊 当前记忆状态")
    print("=" * 60)
    
    if memory.summary:
        print(f"\n📝 摘要区（早期对话摘要）：")
        print(f"   {memory.summary}")
    else:
        print(f"\n📝 摘要区：（无摘要）")
    
    print(f"\n📄 原文区（最近 {len(memory.recent_messages)} 轮完整对话）：")
    for i, (h, a) in enumerate(memory.recent_messages, 1):
        turn_num = 6 - len(memory.recent_messages) + i
        print(f"   第 {turn_num} 轮 - Human: {h[:30]}...")
    
    # 测试记忆效果
    print("\n" + "=" * 60)
    print("🧪 测试记忆效果")
    print("=" * 60)
    
    res = conv.predict(input="总结一下我的基本信息")
    print(f"\n❓ 问题：总结一下我的基本信息")
    print(f"💬 回答：{res}")
    
    print("\n✅ SummaryBufferMemory 既保留了最近细节，又记住了早期关键信息")
    print("💡 适合需要长期记忆但又要控制成本的生产环境（推荐⭐）")


def example_5_token_buffer():
    """
    示例 5：ConversationTokenBufferMemory（Token 限制记忆）
    
    特点：
    - 严格限制保留的对话 Token 数量上限（本例中为 200 tokens）
    - 从最新消息开始逆向保留，直到达到 Token 上限
    - 自动截断早期消息，不进行智能摘要（直接丢弃）
    - 使用 tiktoken 实时计算 Token 数量
    
    优点：
    ✅ 成本可预测：Token 上限明确，不会超支
    ✅ 无额外调用：不需要额外的 LLM 调用生成摘要
    ✅ 响应速度快：实时计算，无延迟
    
    缺点：
    ❌ 截断策略简单：可能在对话中间切断，导致信息不完整
    ❌ 无智能摘要：直接丢弃历史，无法保留关键信息
    
    适用场景：
    - 严格预算控制的生产环境
    - 对话质量要求不高但成本敏感的场景
    - 需要精确预测 API 成本的应用
    
    注意：需要安装 tiktoken 库，且对非 OpenAI 模型可能抛出错误
    """
    print("\n" + "=" * 60)
    print("示例 5：TokenBufferMemory（Token 限制）")
    print("=" * 60)

    try:
        import tiktoken
    except ImportError:
        print("⚠️  需要安装 tiktoken: pip install tiktoken")
        print("跳过此示例...")
        return

    llm = ChatOpenAI(
        model=get_modelverse_model(),
        openai_api_key=get_modelverse_api_key(),
        base_url=get_modelverse_api_base(),
        temperature=0.7,
    )

    # 创建 TokenBufferMemory：最多保留 200 个 token
    # 超出部分直接截断（不进行摘要）
    # ⚠️ 警告：非 OpenAI 模型（如 DeepSeek）会抛出 NotImplementedError
    try:
        memory = ConversationTokenBufferMemory(
            llm=llm,
            max_token_limit=200
        )
        conv = ConversationChain(llm=llm, memory=memory, verbose=False)

        # 进行多轮对话（故意创建大量内容以触发截断）
        print("\n📝 进行多轮对话...")
        print("轮次 1：", end="")
        conv.predict(input="我叫李四，是一名软件工程师")
        print("✓")
        
        print("轮次 2：", end="")
        conv.predict(input="我在深圳的一家互联网公司工作")
        print("✓")
        
        print("轮次 3：", end="")
        conv.predict(input="我主要负责后端开发，使用 Python 和 Go")
        print("✓")
        
        print("轮次 4：", end="")
        conv.predict(input="我的团队有 10 个人")
        print("✓")
        
        print("轮次 5：", end="")
        conv.predict(input="我们正在开发一个电商平台")
        print("✓")
        
        # 查看当前保留的内容
        memory_content = memory.load_memory_variables({})
        print("\n📊 当前保留的对话（最近约 200 tokens 以内）：")
        print("─" * 60)
        print(memory_content["history"])
        print("─" * 60)
        
        # 计算实际保留的消息数量
        history_text = memory_content["history"]
        human_count = history_text.count("Human:")
        ai_count = history_text.count("AI:")
        print(f"\n📈 统计：保留了 {human_count} 轮用户消息 和 {ai_count} 轮 AI 回复")
        
        # 测试早期信息是否被截断
        print("\n❓ 测试记忆效果：")
        res = conv.predict(input="我叫什么名字？在哪个城市工作？")
        print("问题：我叫什么名字？在哪个城市工作？")
        print("💬 回答：", res)
        
        print("\n" + "=" * 60)
        print("📊 TokenBufferMemory 特点总结：")
        print("=" * 60)
        print("✅ 优点：")
        print("  • 严格控制 Token 上限，成本可预测")
        print("  • 不需要额外 LLM 调用生成摘要")
        print("  • 实时计算，响应速度快")
        print("\n❌ 缺点：")
        print("  • 早期信息会被直接截断（如'李四'、'深圳'可能已丢失）")
        print("  • 无智能摘要，无法保留关键信息")
        print("  • 可能在对话中间切断，导致上下文不完整")
        print("\n💡 适用场景：")
        print("  • 严格预算控制的生产环境")
        print("  • 需要精确预测 API 成本的应用")
        print("  • 对话质量要求不高但成本敏感的场景")
        
    except NotImplementedError as e:
        print("\n❌ 错误：当前 LLM 模型不支持 Token 计数")
        print(f"详细信息：{str(e)}")
        print("\n💡 说明：")
        print("  TokenBufferMemory 依赖 tiktoken 库进行 Token 计数，")
        print("  该库主要为 OpenAI 模型设计。使用非 OpenAI 模型（如 DeepSeek）")
        print("  时可能会抛出 NotImplementedError。")
        print("\n🔧 解决方案：自定义字符计数版本（见下方）")
        print("=" * 60)
        
        # 自定义字符计数版本的 TokenBuffer
        print("\n" + "=" * 60)
        print("🔧 替代方案：自定义字符计数版 TokenBuffer")
        print("=" * 60)
        print("💡 使用字符数近似 Token 数（中文 1:1，英文 4:1）")
        
        # 创建自定义 TokenBuffer Memory
        class CustomTokenBufferMemory(BaseMemory):
            """
            自定义 Token 限制记忆（使用字符数近似 Token）
            
            工作原理：
            - 设置 max_chars（最大字符数）近似 max_token_limit
            - 从最新消息开始逆向保留，直到达到字符上限
            - 自动截断早期消息（不进行摘要）
            
            字符数与 Token 的关系：
            - 中文：1 字符 ≈ 1 token
            - 英文：4 字符 ≈ 1 token
            """
            max_chars: int = Field(default=500, description="最大字符数")
            memory_key: str = Field(default="history", description="记忆键名")
            messages: List[tuple] = Field(default_factory=list, description="所有消息")
            
            model_config = {"arbitrary_types_allowed": True}
            
            @property
            def memory_variables(self) -> List[str]:
                return [self.memory_key]
            
            def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
                """加载记忆，只保留最近的消息（不超过 max_chars）"""
                # 策略：从最新往前保留，至少保留最后一轮完整对话
                selected = []
                total_chars = 0
                
                for human_msg, ai_msg in reversed(self.messages):
                    # 计算这一轮对话的完整长度
                    msg_text = f"Human: {human_msg}\nAI: {ai_msg}\n"
                    msg_chars = len(msg_text)
                    
                    # 如果还没有任何内容，至少保留最后一轮（即使超限）
                    if len(selected) == 0:
                        selected.insert(0, (human_msg, ai_msg))
                        total_chars += msg_chars
                    # 如果加上这一轮不超限，就加入
                    elif total_chars + msg_chars <= self.max_chars:
                        selected.insert(0, (human_msg, ai_msg))
                        total_chars += msg_chars
                    else:
                        # 超限了，停止添加
                        break
                
                # 构建历史文本
                history = ""
                for human_msg, ai_msg in selected:
                    history += f"Human: {human_msg}\nAI: {ai_msg}\n"
                
                return {self.memory_key: history.strip()}
            
            def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
                """保存对话上下文"""
                human_msg = inputs.get("input", "")
                # 尝试多个可能的键来获取 AI 回复
                ai_msg = outputs.get("response", outputs.get("output", outputs.get("text", "")))
                self.messages.append((human_msg, ai_msg))
            
            def clear(self) -> None:
                """清空记忆"""
                self.messages = []
        
        # 使用自定义 TokenBuffer（设置合理值以演示截断效果）
        # 中文约 600 字符 = 600 tokens，可以保留 1-2 轮完整对话
        custom_memory = CustomTokenBufferMemory(max_chars=600)
        conv = ConversationChain(llm=llm, memory=custom_memory, verbose=False)
        
        # 进行多轮对话（每条消息较长，容易触发截断）
        print("\n📝 进行多轮对话...")
        conversations = [
            ("我叫李四，是一名软件工程师，有5年的后端开发经验，擅长分布式系统设计", "第1轮"),
            ("我在深圳的一家大型互联网公司工作，主要负责电商平台的微服务架构设计和优化", "第2轮"),
            ("我主要使用 Python 和 Go 语言进行开发，对高并发、分布式系统、消息队列等技术很感兴趣", "第3轮"),
            ("我的团队有10个人，包括后端工程师、前端工程师和测试工程师，我们正在开发一个大型电商平台", "第4轮"),
            ("这个电商平台需要支持每秒10万级的并发请求，同时还要处理海量订单数据，技术挑战非常大", "第5轮"),
        ]
        
        for user_input, label in conversations:
            print(f"{label}：{user_input[:30]}... ", end="")
            response = conv.predict(input=user_input)
            print(f"✓ (输入 {len(user_input)} 字符, 回复 {len(response)} 字符)")
        
        print(f"\n💾 总共进行了 {len(conversations)} 轮对话")
        
        # 查看当前保留的内容
        memory_content = custom_memory.load_memory_variables({})
        history_text = memory_content["history"]
        
        print("\n📊 当前保留的对话（最近 600 字符以内）：")
        print("─" * 60)
        print(history_text)
        print("─" * 60)
        
        # 统计
        human_count = history_text.count("Human:")
        ai_count = history_text.count("AI:")
        actual_chars = len(history_text)
        total_conversations = len(custom_memory.messages)
        
        print(f"\n📈 统计信息：")
        print(f"  • 总对话数：{total_conversations} 轮")
        print(f"  • 保留轮数：{human_count} 轮（第 {total_conversations - human_count + 1}-{total_conversations} 轮）")
        print(f"  • 被截断：{total_conversations - human_count} 轮（第 1-{total_conversations - human_count} 轮）")
        print(f"  • 实际字符数：{actual_chars} / {custom_memory.max_chars} 字符")
        
        # 显示被截断的对话
        if total_conversations > human_count:
            print(f"\n🗑️  被截断的对话内容：")
            for i in range(total_conversations - human_count):
                h, a = custom_memory.messages[i]
                print(f"  第 {i+1} 轮: {h[:40]}...")
        
        # 测试记忆效果（测试早期被截断的信息）
        print("\n❓ 测试记忆效果（询问早期被截断的信息）：")
        res = conv.predict(input="我叫什么名字？")
        print("问题：我叫什么名字？")
        print(f"💬 回答：{res}")
        
        if "李四" in res or "不知道" in res or "没有提到" in res or "没说" in res:
            print("✅ 符合预期：第1轮的名字信息（李四）已被截断")
        else:
            print("⚠️  AI 可能根据上下文进行了推测")
        
        print("\n❓ 测试记忆效果（询问最近保留的信息）：")
        res2 = conv.predict(input="我在哪个城市工作？")
        print("问题：我在哪个城市工作？")
        print(f"💬 回答：{res2}")
        
        if "深圳" in res2:
            print("✅ 符合预期：第2轮的城市信息（深圳）被成功保留")
        
        print("\n" + "=" * 60)
        print("📊 自定义 TokenBuffer 特点总结：")
        print("=" * 60)
        print("✅ 优点：")
        print("  • 完全兼容 DeepSeek 等非 OpenAI 模型")
        print("  • 严格控制字符数/Token 上限，成本可预测")
        print("  • 不需要 tiktoken 库和额外 LLM 调用")
        print("  • 实现简单，性能高效")
        print("\n❌ 缺点：")
        print("  • 早期信息被直接截断（'李四'可能已丢失）")
        print("  • 字符数只是 Token 的近似值（中文较准，英文误差大）")
        print("  • 无智能摘要，无法保留关键信息")
        print("\n💡 字符数与 Token 换算：")
        print("  • 中文：600 字符 ≈ 600 tokens（误差 ±5%）")
        print("  • 英文：600 字符 ≈ 150 tokens（1 token ≈ 4 字符）")
        print("  • 建议：中文直接用字符数，英文用字符数 ÷ 4")
        print("\n⚠️  重要说明：")
        print("  • 字符数计算包含：用户输入 + AI回复 + 格式字符('Human:', 'AI:' 等)")
        print("  • 至少保留最后一轮完整对话（即使超过限制）")
        print("  • 从最新往前累加，尽可能多地保留最近的对话")
        print("  • 与官方 TokenBufferMemory 的统计方式一致")
        
    except Exception as e:
        print(f"\n❌ 运行出错：{str(e)}")
        print("💡 请检查 tiktoken 是否正确安装：pip install tiktoken")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧠 LangChain Memory 完整示例（ModelVerse / DeepSeek-R1）")
    print("=" * 70)
    print("\n本文件演示五种常用 Memory 类型的基本用法：")
    print("1️⃣  BufferMemory - 完整保留所有对话历史")
    print("2️⃣  WindowMemory - 只保留最近 K 轮对话")
    print("3️⃣  SummaryMemory - 智能摘要对话历史")
    print("4️⃣  SummaryBufferMemory - 最近完整 + 早期摘要（推荐）")
    print("5️⃣  TokenBufferMemory - 严格限制 Token 数量")
    print("\n详细对比和选择建议请参考：04_memory/Memory对比说明.md")
    print("=" * 70)
    
    # 运行五个示例
    example_1_buffer()
    example_2_window()
    example_3_summary()
    example_4_summary_buffer()
    example_5_token_buffer()
    
    # 总结
    print("\n" + "=" * 70)
    print("✅ 所有示例运行完成！")




