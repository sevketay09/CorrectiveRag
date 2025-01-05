import os  
import sys  
from typing import List, Union  
  
from langchain.schema import Document  
from langgraph.graph import END, START, StateGraph  
from typing_extensions import TypedDict  
import logging
  
sys.path.append(  
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  
)  
  
from CorrectiveRag.prompt_template import *  
from CorrectiveRag.utils import *  
# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

 
def crag_main(pdfs: List[str], embedding_model_name: str, max_results_k: int):  
    doc_from_pdf = get_document_from_pdf(pdfs)  
    document_chunks = get_chunks_from_document(doc_from_pdf)  
    retriever = get_retriever(embedding_model_name, document_chunks=document_chunks)  
    retrieval_grader = get_retrieval_grading_pipeline()  
    rag_chain = get_rag_pipeline()  
    question_rewriter = get_query_rewriter()  
    web_search_tool = get_web_search(k=max_results_k)  
  
    class GraphState(TypedDict):  
        question: str  
        generation: str  
        web_search: str  
        documents: List[str]  
  
    def retrieve(state):  
        question = state["question"]  
        documents = retriever.invoke(question)  
        return {"documents": documents, "question": question}  
  
    
    def grade_documents(state):
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        web_search = "No"
        
        logger.info(f"Grading documents for question: {question}")
        
        for d in documents:
            score = retrieval_grader.invoke(
                {"document": d.page_content, "question": question}
            )
            grade = score["binary_score"]
            
            if grade == "evet":
                logger.info(f"Document matched: {d.page_content[:100]}...")
                filtered_docs.append(d)
            else:
                web_search = "Yes"
                logger.info("Document not relevant, will use web search")
                continue
                
        logger.info(f"Found {len(filtered_docs)} relevant documents")
        
        return {
            "documents": filtered_docs,
            "question": question,
            "web_search": web_search,
        }

    # generate fonksiyonuna da logging ekleyelim
    def generate(state):
        question = state["question"]
        documents = state["documents"]
        
        logger.info(f"Generating answer for question: {question}")
        logger.info(f"Using {len(documents)} documents for generation")
        
        generation = rag_chain.invoke({"context": documents, "question": question})
        
        logger.info("Answer generated successfully")
        
        return {"documents": documents, "question": question, "generation": generation}
    def transform_query(state):  
        question = state["question"]  
        documents = state["documents"]  
        better_question = question_rewriter.invoke({"question": question})  
        return {"documents": documents, "question": better_question}  
  
    def web_search(state):  
        question = state["question"]  
        documents = state.get("documents", [])  
        docs = web_search_tool.invoke({"query": question})  
        documents.extend(  
            [  
                Document(page_content=d["content"], metadata={"url": d["url"]})  
                for d in docs  
                if isinstance(d, dict)  
            ]  
        )  
        return {"documents": documents, "question": question}  
  
    def decide_to_generate(state):  
        web_search = state["web_search"]  
        if web_search == "Yes":  
            return "transform_query"  
        else:  
            return "generate"  
  
    workflow = StateGraph(GraphState)  
    workflow.add_node("retrieve", retrieve)  
    workflow.add_node("grade_documents", grade_documents)  
    workflow.add_node("generate", generate)  
    workflow.add_node("transform_query", transform_query)  
    workflow.add_node("web_search_node", web_search)  
    workflow.add_edge(START, "retrieve")  
    workflow.add_edge("retrieve", "grade_documents")  
    workflow.add_conditional_edges(  
        "grade_documents",  
        decide_to_generate,  
        {  
            "transform_query": "transform_query",  
            "generate": "generate",  
        },  
    )  
    workflow.add_edge("transform_query", "web_search_node")  
    workflow.add_edge("web_search_node", "generate")  
    workflow.add_edge("generate", END)  
    custom_graph = workflow.compile()  
    return custom_graph  
  
def chat_with_crag(custom_graph: StateGraph):  
    while True:  
        question = input("Sorunuzu girin (çıkmak için 'stop' yazın): ")  
        if question.lower() == "stop":  
            print("CRAG pipeline'ından çıkılıyor. Hoşçakalın!")  
            break  
        example = {"question": question}  
        response = get_crag_response(custom_graph=custom_graph, example=example)  
        print("\nCevap:\n", response)  
        print("\n")  
  
def chat_with_crag_ui(custom_graph, question):  
    if question.lower() == "stop":  
        return "CRAG pipeline'ından çıkılıyor. Hoşçakalın!"  
    example = {"question": question}  
    response = get_crag_response(custom_graph=custom_graph, example=example)  
    return response  