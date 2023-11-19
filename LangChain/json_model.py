from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema import BaseOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def init_server_model(self, llm: ChatOpenAI) -> None:
    # create the data processing modek with access to the suggested names
    template = """
    {chat_history}
    User: {user_input}
    Chatbot:
    """

    prompt = PromptTemplate(
        input_variables=["chat_history", "user_input"], template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history")
    return LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )