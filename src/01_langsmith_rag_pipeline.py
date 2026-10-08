"""Bước 1: FAISS RAG và LangSmith tracing."""

import config  # Nạp .env trước khi import LangChain.

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import (
    load_knowledge_base,
    split_text,
    build_vectorstore,
)
from qa_pairs import SAMPLE_QUESTIONS


def setup_vectorstore():
    """Đọc KB -> chia chunks -> embedding -> FAISS."""
    embeddings = get_embeddings()
    text = load_knowledge_base()

    chunks = split_text(
        text,
        chunk_size=500,
        chunk_overlap=50,
    )

    print(f"Đã chia thành {len(chunks)} chunks")

    return build_vectorstore(chunks, embeddings)


RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer the question using only the supplied context. "
        "Do not add unsupported facts. "
        "If the context is insufficient, say so. "
        "Answer in the same language as the question.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])


def build_rag_chain(vectorstore):
    """Tạo chain retriever -> prompt -> LLM -> parser."""
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = get_llm()

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """Mỗi lần gọi tạo một trace rag-query."""
    return chain.invoke(question)


def main():
    if not config.validate():
        raise SystemExit(1)

    vectorstore = setup_vectorstore()
    chain, _ = build_rag_chain(vectorstore)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)

        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question}")
        print(f"A: {answer}\n")

    print(f"Đã xử lý {len(SAMPLE_QUESTIONS)} câu hỏi.")
    print(
        "Kiểm tra >=50 rag-query trên LangSmith project:",
        config.LANGSMITH_PROJECT,
    )


if __name__ == "__main__":
    main()