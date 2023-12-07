# Install required dependencies from the requirements.txt file
# ! pip install -r requirements.txt

import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ctransformers import AutoModelForCausalLM
from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set the path to the data and the Faiss vector store
DATA_PATH = '/content/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Load documents from PDF files in the specified directory
loader = DirectoryLoader(DATA_PATH,
                          glob='*.pdf',
                          loader_cls=PyPDFLoader)
documents = loader.load()

# Split the documents into smaller chunks for processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Set the API token for Hugging Face Hub
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Define a custom prompt template for QA retrieval
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Create a prompt template object
PROMPT_template = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])

# Load a pre-trained language model for embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': 'cpu'})

# Create a Faiss vector store from the documents
db = FAISS.from_documents(texts, embeddings)

# Load a pre-trained language model from Hugging Face Hub
llm = HuggingFaceHub(
    repo_id='MBZUAI/LaMini-Flan-T5-248M', model_kwargs={"temperature": 0.5, "max_length": 64}
)

# Create an LLMChain with the specified prompt template and language model
llm_chain = LLMChain(prompt=PROMPT_template, llm=llm)

# Create a RetrievalQA chain using the Faiss vector store and the LLMChain
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': PROMPT_template})

# Save the model as a pickle file (commented out for now)
# joblib.dump(qa_chain, 'LLM.pkl')

# Load the model from the pickle file
LLM_Result = joblib.load('LLM.pkl')

# Define a function to interact with the loaded model and get results
def result(query):
    result = LLM_Result({'query': query})
    return result['result']

# Example queries and getting results
result("what is torsion")
result("give me formula for torsion")
result("which material is suitable for piston manufacturing")
result("why grey cast iron is used in piston manufacturing")
result("give me the compression formula")

# Define Chatlit code

# On chat start, initialize the Langchain and set it in the user session
@cl.on_chat_start
async def start():
    chain = result()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)

# On receiving a message, process it using the Langchain and return the result
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()
