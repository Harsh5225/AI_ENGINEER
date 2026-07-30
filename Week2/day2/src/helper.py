import os
import re
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


def get_llm():
    """Initialize and return the Groq LLM."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def get_embeddings():
    """Initialize and return the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_pdf_text(file_path: str) -> str:
    """Load a PDF and concatenate all page content into a single string."""
    loader = PyPDFLoader(file_path)
    data = loader.load()

    full_text = ""
    for page in data:
        full_text += page.page_content

    return full_text


def split_text_for_question_gen(text: str) -> list[Document]:
    """Split raw text into large chunks for question generation."""
    splitter = TokenTextSplitter(
        model_name="gpt-3.5-turbo",
        chunk_size=10000,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]


def split_documents_for_answer_gen(documents: list[Document]) -> list[Document]:
    """Split question-gen documents further into smaller chunks for retrieval/answering."""
    splitter = TokenTextSplitter(
        model_name="gpt-3.5-turbo",
        chunk_size=1000,
        chunk_overlap=100,
    )
    return splitter.split_documents(documents)


def build_vectorstore(documents: list[Document], embeddings) -> FAISS:
    """Build a FAISS vector store from documents."""
    return FAISS.from_documents(documents, embeddings)


def generate_questions(llm, documents: list[Document]) -> str:
    """Run the refine-style summarization chain to generate interview questions."""
    question_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    refine_prompt = PromptTemplate(
        template=refine_template, input_variables=["existing_answer", "text"]
    )

    ques_gen_chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        verbose=True,
        question_prompt=question_prompt,
        refine_prompt=refine_prompt,
    )

    return ques_gen_chain.run(documents)


def parse_questions(raw_questions: str) -> list[str]:
    """Extract a clean list of questions from the numbered raw text output."""
    return re.findall(r"^\d+\.\s*(.*)", raw_questions, flags=re.MULTILINE)


def build_answer_chain(llm, vectorstore):
    """Build the retrieval QA chain used to answer each generated question."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
    )


def generate_and_save_answers(answer_chain, questions: list[str], output_path: str = "answers.txt"):
    """Loop through questions, generate answers, print + save them to file."""
    for question in questions:
        print("Question: ", question)
        answer = answer_chain.run(question)
        print("Answer: ", answer)
        print("********")

        with open(output_path, "a") as f:
            f.write("Question: " + question + "\n")
            f.write("Answer: " + answer + "\n")
            f.write("********\n")