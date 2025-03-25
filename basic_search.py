from huggingface_hub import list_models
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel

# Configuración del modelo LLM
model = LiteLLMModel(
    model_id="ollama_chat/mistral",
    api_base="http://172.17.0.2:11434",  # Cambia esto si usas un servidor remoto
    temperature=0.2,
    api_key="your-api-key",  # Cambia esto si es necesario
    num_ctx=8192,  # Ajusta según la capacidad de tu hardware
)

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")

