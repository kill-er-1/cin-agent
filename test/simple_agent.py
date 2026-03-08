from hello_agents import SimpleAgent,HelloAgentsLLM
from dotenv import load_dotenv

load_dotenv()

llm = HelloAgentsLLM();

agent = SimpleAgent(name="Simple Agent", llm=llm, system_prompt="你是一个简单的智能体，负责回答用户的问题。")

response = agent.run("你好，智能体！你能介绍一下自己吗？")

print("智能体的回答：", response)

from hello_agents.tools import CalculatorTool
calculator = CalculatorTool()
agent.add_tool(calculator)
response = agent.run("请帮我计算一下 2 + 3 * 4 的结果。")
print("智能体的回答：", response)