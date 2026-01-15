from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from langchain.tools import tool

load_dotenv()


# Output schema definition
class WeatherOutput(BaseModel):
    weather_info: str = Field(..., description="The weather information for the specified city.")

# Tool definition
def get_weather(city: str) -> WeatherOutput:
    """Get weather for a given city."""
    return WeatherOutput(weather_info=f"It's always sunny in {city}!")


# Initialize the language model
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="mistralai/devstral-2512:free",
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0
)


# Create the agent with the tool and system prompt
agent = create_agent(llm, tools=[get_weather],
    system_prompt="You are a helpful assistant. Call the get_weather tool to get weather information.", response_format=WeatherOutput)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

print(response['structured_response'].weather_info)