from langchain_openai import AzureOpenAIEmbeddings
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from settings import OPENAI_ENDPOINT


def init_embeddings():
    azure_credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        azure_credential, "https://cognitiveservices.azure.com/.default"
    )

    return AzureOpenAIEmbeddings(
        azure_ad_token_provider=token_provider,
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment="ada-002"
    )
