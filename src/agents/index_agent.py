import asyncio
from enum import StrEnum
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langgraph.graph import StateGraph, MessagesState
from langchain_community.document_loaders.github import GithubFileLoader


class GraphState(StrEnum):
  """
  The state of the graph.
  """
  LOADING_DOCUMENT = 'loading_document'
  INDEXING_DOCUMENT = 'indexing_document'
  EVALUATE = 'evaluate'
  ROLLBACK = 'rollback'
  REPORT = 'report'


class AgentState(MessagesState):
  """
  The state of the agent.
  """
  documents: list[Document]

async def load_document(state: AgentState, config: RunnableConfig):
  """
  Load a document from a git repository.
  process:
  1. load from github
  2. parse by notebook loader
  3. split by markdown header
  """
  loader = GithubFileLoader(config['repo'], config['branch'],)
  state['messages'].append(ToolMessage(content="Loading document...", tool_call_id=str(uuid.uuid4())))
  await asyncio.sleep(1) # Simulate loading time
  state['messages'].append(ToolMessage(content="Document loaded.", tool_call_id=str(uuid.uuid4())))
  text_splitter = MarkdownHeaderTextSplitter()
  documents = loader.load_and_split()


async def index_document(state, config):
  """
  Index a document.
  """

async def evaluate(state, config):
  """
  Evaluate the document.
  """

async def rollback(state, config):
  """
  Rollback the document.
  """

async def report(state, config):
  """
  Report the document.
  """

agent = StateGraph(AgentState)

agent.add_node()