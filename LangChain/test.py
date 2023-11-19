from langchain.agents import AgentExecutor, AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool


@tool
def GPlacesTool(query: str) -> str:
    """Useful to search for information about places or restaurants"""
    import requests

    places_url = "https://places.googleapis.com/v1/places:searchText"
    try:
        gplaces_api_key = os.environ.get("GPLACES_API_KEY")
    except ImportError:
        raise ImportError(
            "Could not import googlemaps python package. "
            "Please install it with `pip install googlemaps`."
        )
    headers = {
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.priceLevel",
        "X-Goog-Api-Key": gplaces_api_key,
        "Content-Type": "application/json",
    }
    payload = {"textQuery": query}
    response = requests.post(places_url, headers=headers, json=payload)
    print(response)
    if response.status_code == 200:
        results = response.json()["places"]
        print(results)
        items = [
            f"{i+1}. {item['displayName']['text']} | Address: {item['formattedAddress']}"
            for i, item in enumerate(results)
        ]
        result = "\n".join(items)
        return result + "\n\n"
    else:
        print(f"Error: {response.status_code} - {response.text}")


class UserLangChain:
    MEMORY_KEY = "chat_history"

    def __init__(self, model="gpt-4", temperature=0) -> None:
        llm = ChatOpenAI(model=model, temperature=temperature)
        self.__init_user_model(llm)
        llm2 = ChatOpenAI(model="gpt-4", temperature=temperature)
        self.__init_server_model(llm2)

    def __init_user_model(self, llm: ChatOpenAI) -> None:
        # create the class instance of a Agent executor with access to GPlaces API
        self.tools = [GPlacesTool]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a friendly AI Chatbot who helps HUMAN users figure out their outing plans. ALWAYS introduce yourself as such.
                    To help the user, you need THREE pieces of information from THEM - location, mode of transportation, things to do.
                    You always ask the user questions until you KNOW location, mode of transportation, and things to do.
                    IF they want to go to a restaurant or place to eat, DO ask for their preferences. You are bad at finding places. Use a tool only ONCE. ALWAYS give top 3 choices ONLY.
                    When giving suggestions of places to go, use numbered bullet points ONLY starting with the name of the place.
                    """,
                ),
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        self.llm_with_tools = llm.bind(
            functions=[format_tool_to_openai_function(t) for t in self.tools]
        )

        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | self.llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        # self.agent_executor = initialize_agent(
        #     self.tools, llm, agent=agent, verbose=True
        # )
        self.chat_history = []

    def __init_server_model(self, llm: ChatOpenAI) -> None:
        # create the data processing modek with access to the suggested names
        template = """
        You are an AI that looks out for two things: names of places and mode of transportation. 
        If found, you will output them in a JSON blob with 2 keys "mode" (string) and "places" (list of strings). The keys will be empty if nothing is found.
        Only output this JSON blob and nothing else.
        examples of mode are 'driving', 'walking', 'public transport'
        examples of places are ['Restaurant A', 'The Bean', 'Beach']
        NEVER output anything other than this JSON blob

        {chat_hist}
        user: {user_input}
        Chatbot:"""

        prompt = PromptTemplate(
            input_variables=["chat_hist", "user_input"], template=template
        )
        memory = ConversationBufferMemory(memory_key="chat_hist")
        self.server_executor = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=False,
            memory=memory,
        )

    def invoke(self, input: str):
        result = self.agent_executor.invoke(
            {"input": input, "chat_history": self.chat_history}
        )
        user_input = result["output"] + " " + input
        summary = self.server_executor.predict(user_input=user_input)
        self.chat_history.extend(
            [
                HumanMessage(content=input),
                AIMessage(content=result["output"]),
            ]
        )
        print(summary)
        return result


lc_model = UserLangChain()
lc_model.invoke("Hi!")
# lc_model.invoke("I'd like to plan a date. Can I get some help?")
# lc_model.invoke(
#     "I'd like to go to Chicago, and I have a car. Sightseeing and a restaurant will be good"
# )
# lc_model.invoke("no restrictions, my date likes Korean food though")

while True:
    a = input("Enter prompt: ")
    if a == "0":
        break
    lc_model.invoke(a)
