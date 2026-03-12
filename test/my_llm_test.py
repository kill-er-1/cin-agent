from pathlib import Path

from dotenv import load_dotenv
import sys

# 把项目根目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from core.llm import MyLLM

load_dotenv()


llm_deepseek = MyLLM(provider="deepseek")
llm_kimi = MyLLM(provider="kimi")
llm_qwen = MyLLM(provider="qwen")

message = [{"role": "user", "content": "你好，你是什么模型？"}]

response_deepseek = llm_deepseek.think(message)
response_kimi = llm_kimi.think(message)
response_qwen = llm_qwen.think(message)
print("DeepSeek模型的回答:")
for chunk in response_deepseek:
    # chunk在my_llm库中已经打印过一遍，这里只需要pass即可
    # print(chunk, end="", flush=True)
    pass

print("\nKimi模型的回答:")
for chunk in response_kimi:
    # chunk在my_llm库中已经打印过一遍，这里只需要pass即可
    # print(chunk, end="", flush=True)
    pass  
  
print("\nQwen模型的回答:")
for chunk in response_qwen:
    # chunk在my_llm库中已经打印过一遍，这里只需要pass即可
    # print(chunk, end="", flush=True)
    pass
  