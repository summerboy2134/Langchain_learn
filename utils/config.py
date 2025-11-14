"""
配置管理工具
用于加载和管理环境变量
"""
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


def get_openai_api_key():
	"""获取 OpenAI API 密钥"""
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise ValueError(
			"未找到 OPENAI_API_KEY。请在 .env 文件中设置你的 API 密钥。"
		)
	return api_key


def get_openai_api_base():
	"""获取 OpenAI API 基础 URL（可选）"""
	return os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")


def get_model_name(default="gpt-3.5-turbo"):
	"""获取默认模型名称"""
	return os.getenv("OPENAI_MODEL", default)


def get_modelverse_api_key():
	"""获取 ModelVerse API 密钥"""
	api_key = os.getenv("MODELVERSE_API_KEY")
	if not api_key:
		raise ValueError(
			"未找到 MODELVERSE_API_KEY。请设置环境变量或在 .env 文件中设置你的 API 密钥。"
		)
	return api_key


def get_modelverse_api_base():
	"""获取 ModelVerse API 基础 URL"""
	return os.getenv("MODELVERSE_API_BASE", "https://api.modelverse.cn/v1")


def get_modelverse_model(default="deepseek-ai/DeepSeek-R1"):
	"""获取 ModelVerse 默认模型名称"""
	return os.getenv("MODELVERSE_MODEL", default)


