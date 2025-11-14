### LangChain 学习项目 - 安装与测试指南（conda 环境：langchain）

本项目包含 LangChain 的提示模板、LLM、Agents、Memory 等示例。建议在 conda 环境中运行本项目，按以下步骤即可快速恢复并验证运行。

### 一、准备环境
- Python 版本：建议 3.10+

### 二、安装依赖
1) 激活环境

```bash
conda activate env_name
```

2) 安装依赖

```bash
python -m pip install -U pip
pip install -r requirements.txt
```

如需加速，可配置国内镜像源（可选）。

### 三、配置环境变量
1) 复制示例环境文件并填写密钥

```bash
cp example_env.txt .env
```

2) 在 `.env` 中按需填写以下变量（推荐使用 ModelVerse）：
- 必填（使用 ModelVerse 时）：
  - `MODELVERSE_API_KEY`
  - `MODELVERSE_API_BASE`（默认 `https://api.modelverse.cn/v1`，可保持默认）
  - `MODELVERSE_MODEL`（如 `deepseek-ai/DeepSeek-R1`）
- 可选（若改用原生 OpenAI）：
  - `OPENAI_API_KEY`
  - `OPENAI_API_BASE`（默认 `https://api.openai.com/v1`）
  - `OPENAI_MODEL`

项目通过 `utils/config.py` 自动加载 `.env`。

### 四、快速自测
确保 `.env` 已正确配置后，运行以下示例脚本进行验证：

```bash
# 模版调用示例
python 01_prompt_templates/prompt_examples.py

# 基于 ModelVerse 的基础调用
python 02_llm/modelverse_llm.py

# Agents 示例
python 03_agents/basic_agent.py

# Memory 示例（总结记忆）
python 04_memory/summary_memory.py
```

若能看到模型返回或流式输出，即表示环境与依赖已正确配置。

### 五、常见问题
- 未找到密钥或 401/403：检查 `.env` 是否已填写并生效（终端重开或确认当前目录）。
- 网络报错或超时：检查网络是否可访问对应 API 域名，必要时配置代理（`HTTP_PROXY`/`HTTPS_PROXY`）。
- 依赖冲突：可尝试升级 pip、清理缓存并重新安装；或新建一个干净的 conda 环境重试。

### 六、项目结构（节选）
```
Langchain/
├── 01_prompt_templates/
│   ├── basic_prompt.py
│   ├── prompt_examples.py
├── 02_llm/
│   ├── basic_llm.py
│   ├── llm_examples.py
│   ├── modelverse_llm.py
│   └── streaming_llm.py
├── 03_agents/
│   ├── agent_extraction_examples.py
│   ├── basic_agent.py
│   ├── custom_agent.py
│   └── tools_agent.py
├── 04_memory/
│   ├── basic_memory.py
│   ├── conversation_memory.py
│   ├── memory_examples.py
│   └── summary_memory.py
├── utils/
│   ├── config.py
│   └── token_counter.py
├── example_env.txt
├── requirements.txt
└── README.md
```
