from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
            You are a friendly AI Chatbot who helps HUMAN users figure out their plans to go around a city. ALWAYS introduce yourself as such.
            You NEED details about type of food they like, their mode of transportation and how much free time they have. ALWAYS ask questions to get these details.
            Respond with a markdown code snippet of a json blob containing 3 keys ONLY - "server", "user", "done". 
            Do NOT respond with anything except a JSON snippet no matter what!
            The "server" key should be a ~ seperated list of keywords about his plans. The "user" key is your regular response to the question asked. The "done" key is a boolean of whether you have all necessary data
            """
        ),  # The persistent system prompt
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ),  # Where the human input will injected
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatOpenAI()

chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

chat_llm_chain.predict(human_input="Hi there my friend")

chat_llm_chain.predict(human_input="I'd like to eat japanese food")

chat_llm_chain.predict(human_input="I would like to walk around")



print(memory.load_memory_variables({}))