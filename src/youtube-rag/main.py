# https://github.com/svpino/youtube-rag/blob/main/rag.ipynb

from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from llm import init_llm
from embeddings import init_embeddings

with open("src/youtube-rag/transcription.txt") as file:
    transcription = file.read()

loader = TextLoader("src/youtube-rag/transcription.txt")
text_documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(text_documents)

embeddings = init_embeddings()
model = init_llm()
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

result = chain.invoke("Does Rellana love Messmer?")
print()
print(result)