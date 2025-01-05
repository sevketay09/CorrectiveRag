import os    
from pprint import pprint    
from typing import Any, List   
    
from dotenv import load_dotenv      
from langchain.text_splitter import RecursiveCharacterTextSplitter    
from langchain.document_loaders import PyPDFLoader     
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser    
from langchain_core.prompts import ChatPromptTemplate    
from langgraph.graph import StateGraph    
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  
from openai import AzureOpenAI  
import requests
from CorrectiveRag.prompt_template import GradePrompt, RewritePrompt,GeneratePrompt  
    
# Load the API key from the .env file    
load_dotenv()    
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")    
openai_api_key = os.getenv("AZURE_OPENAI_KEY")    
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")    
    
# Initialize the OpenAI client    
client =AzureOpenAI(    
    azure_endpoint=openai_endpoint,    
    api_key=openai_api_key,  
    api_version="2024-02-15-preview"    
)    
    
# Util function to get documents from PDF file(s) provided    
def get_document_from_pdf(sources: List[str]):    
    docs = []    
    for source in sources:    
        if source.lower().endswith('.pdf') and os.path.isfile(source):    
            loader = PyPDFLoader(source)    
            docs.extend(loader.load())    
        else:    
            raise ValueError(f"Unsupported source or file not found: {source}")    
    return docs    
    
# Util function to split the document into chunks    
def get_chunks_from_document(document, text_chunk_size: int = 250, overlap: int = 0):    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(    
        chunk_size=text_chunk_size, chunk_overlap=overlap    
    )    
    doc_splits = text_splitter.split_documents(document)    
    return doc_splits    


def get_retriever(embedding_model_name: str, document_chunks):
    # HuggingFaceEmbeddings kullanarak embedding işlemini yapılandır
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # FAISS vectorstore oluştur
    vectorstore = FAISS.from_documents(
        documents=document_chunks,
        embedding=embeddings
    )

    # Retriever'ı döndür 
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    return retriever
    
# Util function to check if the retriever is working    
def check_retriever(retriever: Any, query: str, top_k: int = 3):    
    results = retriever.get_relevant_documents(query, top_k=top_k)    
    assert len(results) > 0, "No results found"    
    return results    
    
# Util function to get the retrieval grading pipeline    
def get_retrieval_grading_pipeline():    
    grading_prompt = ChatPromptTemplate.from_messages(    
        [    
            ("system", GradePrompt),    
            (    
                "human",    
                "Bulunan belge: \n\n {document} \n\n Kullanıcı sorusu: {question}",    
            ),    
        ]    
    )    
    
    def grading_llm(document, question):    
        response = client.chat.completions.create(    
            model="gpt-4o",    
            messages=[    
                {"role": "system", "content": GradePrompt},    
                {"role": "user", "content": f"Bulunan belge: \n\n {document} \n\n Kullanıcı sorusu: {question}"}    
            ],    
            max_tokens=1024,    
            temperature=0    
        )    
        return response.choices[0].message.content
    
    def grading_llm_wrapper(inputs):    
        document = inputs["document"]    
        question = inputs["question"]    
        return grading_llm(document, question)    
    
    retrieval_grader = grading_llm_wrapper | JsonOutputParser()    
    return retrieval_grader    
    
def get_rag_pipeline():
    def generation_llm(inputs):
        context = inputs["context"]
        question = inputs["question"]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": GeneratePrompt},
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
            ],
            max_tokens=1024,
            temperature=0.7
        )
        return response.choices[0].message.content

    rag_chain_pipeline = generation_llm | StrOutputParser()
    return rag_chain_pipeline
    
# Util function to get the query rewriter    
def get_query_rewriter():    
    re_write_prompt = ChatPromptTemplate.from_messages(    
        [    
            ("system", RewritePrompt),    
            (    
                "human",    
                "İlk soru: \n\n {question} \n Daha iyi bir soru formüle edin.",    
            ),    
        ]    
    )    
    
    def rewriter_llm(inputs):    
        question = inputs["question"]    
        response = client.chat.completions.create(    
            model="gpt-4o",    
            messages=[    
                {"role": "system", "content": RewritePrompt},    
                {"role": "user", "content": f"İlk soru: \n\n {question} \n Daha iyi bir soru formüle edin."}    
            ],    
            max_tokens=1024,    
            temperature=0.7    
        )    
        return response.choices[0].message.content
      
    question_rewriter = rewriter_llm | StrOutputParser()    
    return question_rewriter    
    
# Util function to get the web search tool    
def get_web_search(k: int = 3):    
    class TavilySearchResults:    
        def __init__(self, max_results):    
            self.max_results = max_results    
    
        def invoke(self, inputs):    
            query = inputs["query"]    
            headers = {    
                "Authorization": f"Bearer {os.getenv('TAVILY_API_KEY')}",    
                "Content-Type": "application/json"    
            }    
            data = {    
                "query": query,    
                "max_results": self.max_results    
            }    
            response = requests.post(    
                "https://api.tavily.com/search",    
                headers=headers,    
                json=data    
            )    
            response.raise_for_status()    
            return response.json().get("results", [])    
    
    return TavilySearchResults(max_results=k)    
    
def get_crag_response(custom_graph: StateGraph, example: dict):    
    for output in custom_graph.stream(example):    
        for key, value in output.items():    
            pprint(f"Node '{key}':")    
        pprint("\n---\n")    
    return value["generation"]  