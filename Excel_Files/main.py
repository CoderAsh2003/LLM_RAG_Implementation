from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_ollama import ChatOllama,OllamaEmbeddings
from langchain.vectorstores.faiss import FAISS
from IPython.display import display,Markdown
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import os
import faiss

current_dir = os.path.dirname(os.path.abspath(__file__))

db_dir = os.path.join(current_dir,'faiss_index_store')



def prepare_excel_for_rag(filepath,persistent_directory):
    loader = UnstructuredExcelLoader(filepath)
    data = loader.load()

    #Split the text into smaller chunks with overlap
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 10000,
    chunk_overlap = 100
    )  
    chunks = text_splitter.split_documents(data)

    #Create embeddings using Ollama 3.2
    embeddings = OllamaEmbeddings(model = 'llama3.2',num_gpu=1)
    
    #Use FAISS for vector store as a db
    db_faiss = FAISS.from_documents(chunks,embeddings)

    #Save
    faiss.write_index(db_faiss.index,persistent_directory)

    return db_faiss

def rag(db_faiss,query,k):
    output_retrieval = db_faiss.similarity_search(query,k = k)
    
    #Merge Chunks
    output_retrieval_merged = '\n'.join([doc.page_content for doc in output_retrieval])

    #Define prompt
    prompt = f"Based on the context {output_retrieval_merged}, Answer the following question: {query}.If you do not possess the information on the answer, respond with I dont know."

    #Call Ollama Model
    ollama_llm = ChatOllama(model = 'llama3.2', num_gpu=1,temperature=0)

    response = ollama_llm.invoke(prompt)

    return f"{response.content}"

#Test the model

#Create a vector database
db_excel = prepare_excel_for_rag('updated_sample_file.xlsx')

#Implement the RAG System
query = "What is Kathleen Hanner's gender?"
rag(db_excel,query,4)