from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pathlib import Path

from model import llm, embeddings

DIR = Path(__file__).parent / "transcription/transcription-elden-lore.txt"
loader = TextLoader(DIR)
text_documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(text_documents)

parser = StrOutputParser()

template = """
Answer the question based on the context below. If you can't 
answer the question, simply reply "I don't know".
Only use the information from the provided context, do not search the web.

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
chain = setup | prompt | llm | parser

question = input("What do you wish to know?\n")
result = chain.invoke(question)
print()
print(result)