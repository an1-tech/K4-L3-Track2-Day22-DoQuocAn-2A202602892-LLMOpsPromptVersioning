"""Bước 3: đánh giá 50 QA cho mỗi prompt bằng RAGAS."""

import importlib
import json
from pathlib import Path

import config
import numpy as np

from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.run_config import RunConfig

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import (
    load_knowledge_base,
    split_text,
    build_vectorstore,
)
from qa_pairs import QA_PAIRS


prompt_module = importlib.import_module(
    "02_prompt_hub_ab_routing"
)

PROMPTS = {
    "v1": prompt_module.PROMPT_V1,
    "v2": prompt_module.PROMPT_V2,
}

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
]

ROOT = Path(__file__).resolve().parent.parent


def setup_vectorstore():
    chunks = split_text(
        load_knowledge_base(),
        chunk_size=500,
        chunk_overlap=50,
    )

    return build_vectorstore(chunks, get_embeddings())


def run_rag(retriever, llm, prompt, question: str) -> dict:
    docs = retriever.invoke(question)

    # Giữ riêng từng passage cho RAGAS.
    contexts = [doc.page_content for doc in docs]

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": "\n\n".join(contexts),
        "question": question,
    })

    return {
        "answer": answer,
        "contexts": contexts,
    }


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = get_llm()
    prompt = PROMPTS[prompt_version]
    results = []

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(
            retriever,
            llm,
            prompt,
            qa["question"],
        )

        results.append({
            "question": qa["question"],
            "reference": qa["reference"],
            "answer": out["answer"],
            "contexts": out["contexts"],
        })

        print(
            f"[{prompt_version}] "
            f"[{i:02d}/{len(QA_PAIRS)}] "
            f"{qa['question']}",
            flush=True,
        )

    # Dữ liệu thật để debug nếu phần evaluation gặp lỗi.
    output_path = ROOT / "data" / f"rag_outputs_{prompt_version}.json"

    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return results


def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


def run_ragas_eval(rag_results: list, version: str) -> dict:
    print(f"Đang chấm RAGAS {version}...", flush=True)

    result = evaluate(
        build_ragas_dataset(rag_results),
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=get_llm(temperature=0),
        embeddings=get_embeddings(),
        run_config=RunConfig(
            max_workers=2,
            timeout=180,
            max_retries=5,
        ),
        raise_exceptions=True,
    )

    scores = {}

    for key in METRIC_NAMES:
        raw = np.asarray(result[key], dtype=float)

        if (
            raw.size != len(rag_results)
            or not np.isfinite(raw).all()
        ):
            raise ValueError(
                f"{version}/{key}: thiếu điểm hoặc có NaN/Infinity"
            )

        scores[key] = float(np.mean(raw))

    return scores


def main():
    if not config.validate():
        raise SystemExit(1)

    if len(QA_PAIRS) != 50:
        raise ValueError("Bài yêu cầu 50 QA cho mỗi prompt")

    client = Client(api_key=config.LANGSMITH_API_KEY)

    hub_prompts = prompt_module.pull_prompts_from_hub(client)

    PROMPTS.update({
        "v1": hub_prompts[prompt_module.PROMPT_V1_NAME],
        "v2": hub_prompts[prompt_module.PROMPT_V2_NAME],
    })

    vectorstore = setup_vectorstore()

    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    print("\n" + "=" * 70)
    print(f"{'Metric':25s} {'V1':>9s} {'V2':>9s}   Winner")
    print("=" * 70)

    for metric in METRIC_NAMES:
        s1 = v1_scores[metric]
        s2 = v2_scores[metric]

        if abs(s1 - s2) < 1e-9:
            winner = "Tie"
        else:
            winner = "V1" if s1 > s2 else "V2"

        print(
            f"{metric:25s} "
            f"{s1:9.4f} {s2:9.4f}   {winner}"
        )

    best_faith = max(
        v1_scores["faithfulness"],
        v2_scores["faithfulness"],
    )

    print(
        f"Faithfulness tốt nhất: {best_faith:.4f}; "
        f"đạt >=0.8: {best_faith >= 0.8}"
    )

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
        "samples_per_version": {
            "v1": len(v1_results),
            "v2": len(v2_results),
        },
    }

    report_text = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

    paths = [
        ROOT / "data" / "ragas_report.json",
        ROOT / "evidence" / "03_ragas_report.json",
    ]

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_text, encoding="utf-8")
        print(f"Đã lưu: {path}")


if __name__ == "__main__":
    main()