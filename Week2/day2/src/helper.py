import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

from src.prompt import prompt_template, refine_template

load_dotenv()


def file_processing(file_path):
    """
    Loads a PDF and splits it into two different sets of chunks:
    - document_ques_gen: large chunks, used to generate broad, well-rounded questions
    - document_answer_gen: smaller chunks, used later to build the retriever
      that answers each generated question accurately
    """
    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ""
    for page in data:
        question_gen += page.page_content

    # Large chunks -> good for generating broad questions covering a full concept
    splitter_ques_gen = TokenTextSplitter(
        model_name="gpt-3.5-turbo",
        chunk_size=10000,
        chunk_overlap=200
    )
    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)
    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    # Smaller chunks -> better precision when retrieving context to answer a specific question
    splitter_ans_gen = TokenTextSplitter(
        model_name="gpt-3.5-turbo",
        chunk_size=1000,
        chunk_overlap=100
    )
    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

    return document_ques_gen, document_answer_gen


def llm_pipeline(file_path):
    """
    Full pipeline: process the file, generate ~10 interview questions,
    build a retriever over the document, and return everything needed
    to answer those questions on demand.
    """
    document_ques_gen, document_answer_gen = file_processing(file_path)

    # LLM used purely for question generation
    llm_ques_gen_pipeline = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    # "refine" chain type: generates questions from the first chunk, then
    # progressively refines/extends them as it reads through the remaining chunks
    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen_pipeline,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS
    )

    ques = ques_gen_chain.run(document_ques_gen)

    # Embeddings + vector store for answering questions later
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    # Separate LLM instance for answering (kept distinct from question-gen LLM
    # in case you want different temperature/settings for each task)
    llm_answer_gen = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Clean up the raw question text into a proper list
    ques_list = ques.split("\n")
    filtered_ques_list = [
        q.strip() for q in ques_list
        if q.strip().endswith('?') or q.strip().endswith('.')
    ]

    answer_generation_chain = RetrievalQA.from_chain_type(
        llm=llm_answer_gen,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    return answer_generation_chain, filtered_ques_list
