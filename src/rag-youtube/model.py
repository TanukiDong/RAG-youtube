from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from settings import (
    OPENAI_ENDPOINT,
    EMBEDDING_MODEL_ID,
    LLM_MODEL_ID,
    OPENAI_API_VERSION,
)
azure_credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    azure_credential, "https://cognitiveservices.azure.com/.default"
)

embeddings = AzureOpenAIEmbeddings(
        azure_ad_token_provider=token_provider,
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=EMBEDDING_MODEL_ID
    )
llm = AzureChatOpenAI(
        api_version=OPENAI_API_VERSION,
        azure_ad_token_provider=token_provider,
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=LLM_MODEL_ID,
    )