from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from settings import (
    OPENAI_ENDPOINT,
    EMBEDDING_MODEL_ID,
    LLM_MODEL_ID,
    OPENAI_API_VERSION,
    TEMPERATURE,
    SEED)

azure_credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    azure_credential, "https://cognitiveservices.azure.com/.default"
)

loader = TextLoader("src/youtube-rag/transcription-elden-lore.txt")
text_documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(text_documents)

embeddings = AzureOpenAIEmbeddings(
        azure_ad_token_provider=token_provider,
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=EMBEDDING_MODEL_ID
    )
model = AzureChatOpenAI(
        api_version=OPENAI_API_VERSION,
        azure_ad_token_provider=token_provider,
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=LLM_MODEL_ID,
        temperature=TEMPERATURE,
        model_kwargs={"seed": SEED},
        streaming=True,
    )
parser = StrOutputParser()

template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# translation_prompt = ChatPromptTemplate.from_template(
#     "Translate {answer} to {language}"
# )

# translation_chain = (
#     {"answer": chain, "language": itemgetter("language")} | translation_prompt | model | parser
# )

vectorstore = DocArrayInMemorySearch.from_documents(documents, embeddings)
setup = RunnableParallel(
    context=vectorstore.as_retriever(),
    question=RunnablePassthrough()
    )
chain = setup | prompt | model | parser

result = chain.invoke("What is an Empyrean?")
print()
print(result)