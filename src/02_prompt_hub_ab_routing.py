"""Bước 2: Prompt Hub và A/B routing tất định."""

import hashlib
import config

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import (
    load_knowledge_base,
    split_text,
    build_vectorstore,
)
from qa_pairs import SAMPLE_QUESTIONS


PROMPT_V1_NAME = "doquocan-2a202602892-day22-rag-v1"
PROMPT_V2_NAME = "doquocan-2a202602892-day22-rag-v2"

SYSTEM_V1 = (
    "You are a concise AI study assistant. "
    "Answer directly in 2-4 short sentences, "
    "using only facts supported by the context. "
    "Avoid unnecessary examples. "
    "If context is insufficient, clearly state that limitation. "
    "Use the same language as the question.\n\n"
    "Context:\n{context}"
)

SYSTEM_V2 = (
    "You are an AI technical educator. "
    "Give a structured answer in 3-5 sentences: "
    "start with the definition or main answer, "
    "then explain the mechanism, "
    "then its purpose or limitation when supported by the context. "
    "Use only context-supported facts; "
    "do not invent details or examples. "
    "If context is insufficient, clearly state that limitation. "
    "Use the same language as the question.\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human", "{question}"),
])

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human", "{question}"),
])


def push_prompts_to_hub(client):
    """Push hai prompt; chỉ bỏ qua trường hợp nội dung không đổi."""
    items = [
        (PROMPT_V1_NAME, PROMPT_V1, "V1: concise grounded answer"),
        (PROMPT_V2_NAME, PROMPT_V2, "V2: structured grounded explanation"),
    ]

    for name, prompt, description in items:
        try:
            url = client.push_prompt(
                name,
                object=prompt,
                description=description,
            )
            print(f"Đã push {name}: {url}")

        except Exception as exc:
            if "nothing to commit" in str(exc).lower():
                print(f"{name}: prompt không đổi, Nothing to commit")
            else:
                raise


def pull_prompts_from_hub(client) -> dict:
    """Pull thật từ Hub; không dùng local fallback."""
    prompts = {}

    for name in (PROMPT_V1_NAME, PROMPT_V2_NAME):
        prompt = client.pull_prompt(name)

        if set(prompt.input_variables) != {"context", "question"}:
            raise ValueError(
                f"{name}: phải có hai biến context và question"
            )

        prompts[name] = prompt
        print(f"↓ Đã pull '{name}' từ Hub")

    return prompts


def get_prompt_version(request_id: str) -> str:
    """Cùng request_id -> cùng nhánh prompt."""
    hash_int = int(
        hashlib.md5(request_id.encode("utf-8")).hexdigest(),
        16,
    )

    return (
        PROMPT_V1_NAME
        if hash_int % 2 == 0
        else PROMPT_V2_NAME
    )


@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(
    retriever,
    llm,
    prompt,
    question: str,
    version: str,
) -> dict:
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": "\n\n".join(contexts),
        "question": question,
    })

    return {
        "question": question,
        "answer": answer,
        "version": version,
        "contexts": contexts,
    }


def setup_vectorstore():
    chunks = split_text(
        load_knowledge_base(),
        chunk_size=500,
        chunk_overlap=50,
    )

    return build_vectorstore(chunks, get_embeddings())


def main():
    if not config.validate():
        raise SystemExit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)

    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    retriever = setup_vectorstore().as_retriever(
        search_kwargs={"k": 3}
    )

    llm = get_llm()
    counts = {"v1": 0, "v2": 0}

    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)

        version = (
            "v1" if version_key == PROMPT_V1_NAME else "v2"
        )

        result = ask_ab(
            retriever,
            llm,
            prompts[version_key],
            question,
            version,
            langsmith_extra={
                "metadata": {
                    "request_id": request_id,
                    "prompt_name": version_key,
                }
            },
        )

        counts[version] += 1

        print(
            f"[{i+1:02d}] [{request_id}] "
            f"[prompt-{version}] {question}"
        )
        print(f"A: {result['answer']}\n")

    print(
        f"Routing: V1={counts['v1']} | "
        f"V2={counts['v2']} | "
        f"Tổng={sum(counts.values())}"
    )


if __name__ == "__main__":
    main()