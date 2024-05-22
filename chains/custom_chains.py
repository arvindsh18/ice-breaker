from langchain.prompts import PromptTemplate
from langchain.chat_models import AzureChatOpenAI

# from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os
from output_parsers import summary_parser, ice_breaker_parser, topics_of_interest_parser


def get_summary_chain() -> LLMChain:

    
    openai_api_key = "827883e9e67841f583d019494b915f8d"
    llm = AzureChatOpenAI(
        deployment_name="AzureModel",
        openai_api_version="2023-05-15",
        openai_api_key=openai_api_key,
        openai_api_base="https://hv-openai-lab31.openai.azure.com",
    )
    

    summary_template = """
         given the information about a person from linkedin {information} I want you to create:
         1. a short summary
         2. two interesting facts about them
         \n{format_instructions}
     """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": summary_parser.get_format_instructions()
        },
    )

    return LLMChain(llm=llm, prompt=summary_prompt_template)


def get_interests_chain() -> LLMChain:

    openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    llm = AzureChatOpenAI(
        deployment_name=os.environ.get("azure_deployment"),
        openai_api_version=os.environ.get("api_version"),
        openai_api_key=openai_api_key,
        openai_api_base=os.environ.get("azure_endpoint"),
    )
    
    interesting_facts_template = """
         given the information about a person from linkedin {information} I want you to create:
         3 topics that might interest them
        \n{format_instructions}
     """

    interesting_facts_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=interesting_facts_template,
        partial_variables={
            "format_instructions": topics_of_interest_parser.get_format_instructions()
        },
    )

    return LLMChain(llm=llm, prompt=interesting_facts_prompt_template)


def get_ice_breaker_chain() -> LLMChain:
    
    openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    llm_creative = AzureChatOpenAI(
        deployment_name=os.environ.get("azure_deployment"),
        openai_api_version=os.environ.get("api_version"),
        openai_api_key=openai_api_key,
        openai_api_base=os.environ.get("azure_endpoint"),
    )

    ice_breaker_template = """
         given the information about a person from linkedin {information} I want you to create:
         2 creative Ice breakers with them that are derived from their activity on Linkedin preferably on
        \n{format_instructions}
     """

    ice_breaker_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=ice_breaker_template,
        partial_variables={
            "format_instructions": ice_breaker_parser.get_format_instructions()
        },
    )

    return LLMChain(llm=llm_creative, prompt=ice_breaker_prompt_template)
