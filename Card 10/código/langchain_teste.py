from dotenv import load_dotenv
from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_community.tools.youtube.search import YouTubeSearchTool
from langchain_community.tools.google_trends import GoogleTrendsQueryRun
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_openai_functions_agent, AgentExecutor

load_dotenv()

# Tools
youtubeTool = YouTubeSearchTool()
google_trends = GoogleTrendsQueryRun(api_wrapper=GoogleTrendsAPIWrapper())
wikipediaTool = WikipediaQueryRun(api_wrapper= WikipediaAPIWrapper())
tools = [youtubeTool, google_trends, wikipediaTool]


# Prompt

prompt = hub.pull('hwchase17/openai-functions-agent')

# LLM

llm = ChatOllama(model ="mistral", temperature = 0)

# Juntar Tools, Prompt, LLM, Agent

meu_agente = create_openai_functions_agent(llm,tools, prompt)

# Executor

agent_executor = AgentExecutor(agent= meu_agente, tools= tools, verbose= True)

agent_executor.invoke({'input': 'me de uma pesquisa da wikipedia sobre bethoven'})