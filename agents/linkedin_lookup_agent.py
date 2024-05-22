from tools.tools import get_profile_url

from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import os


def lookup(name: str) -> str:
     openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
     llm = AzureChatOpenAI(
        deployment_name=os.environ.get("azure_deployment"),
        openai_api_version=os.environ.get("api_version"),
        openai_api_key=openai_api_key,
        openai_api_base=os.environ.get("azure_endpoint"),
    )
     template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                          Your answer should contain only a URL"""
     tools_for_agent1 = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url,
            description="useful for when you need get the Linkedin Page URL",
        ),
    ]
     agent = initialize_agent(
        tools_for_agent1, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
     prompt_template = PromptTemplate(
        input_variables=["name_of_person"], template=template
    )
     linkedin_username = agent.run(prompt_template.format_prompt(name_of_person=name))
     linkedin_username="https"+linkedin_username.split("https")[1]
     return linkedin_username
