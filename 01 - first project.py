from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
from dotenv import load_dotenv
load_dotenv()


model = HfApiModel(
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
provider=None,
)

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("How long would it take for an elephant to cross the united states from florida to california?")