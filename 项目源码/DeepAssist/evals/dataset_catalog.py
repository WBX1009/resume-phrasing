from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_id: str
    suite: str
    split: str
    source_url: str
    license: str = "see_dataset_card"
    config: Optional[str] = None
    notes: str = ""


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "beir_nfcorpus": DatasetSpec(
        name="beir_nfcorpus",
        hf_id="BeIR/nfcorpus",
        suite="retrieval",
        split="test",
        source_url="https://huggingface.co/datasets/BeIR/nfcorpus",
        notes="BEIR retrieval benchmark. Some HuggingFace mirrors expose corpus/queries/qrels via configs.",
    ),
    "beir_fiqa": DatasetSpec(
        name="beir_fiqa",
        hf_id="BeIR/fiqa",
        suite="retrieval",
        split="test",
        source_url="https://huggingface.co/datasets/BeIR/fiqa",
        notes="BEIR finance retrieval benchmark.",
    ),
    "duretrieval": DatasetSpec(
        name="duretrieval",
        hf_id="mteb/DuRetrieval",
        suite="retrieval",
        split="test",
        source_url="https://huggingface.co/datasets/mteb/DuRetrieval",
        notes="Chinese retrieval benchmark from C-MTEB/MTEB.",
    ),
    "squad": DatasetSpec(
        name="squad",
        hf_id="rajpurkar/squad",
        suite="rag",
        split="validation",
        source_url="https://huggingface.co/datasets/rajpurkar/squad",
        notes="Extractive QA benchmark used for deterministic answer EM/F1 and RAG context checks.",
    ),
    "hotpotqa": DatasetSpec(
        name="hotpotqa",
        hf_id="hotpotqa/hotpot_qa",
        suite="rag",
        split="validation",
        source_url="https://huggingface.co/datasets/hotpotqa/hotpot_qa",
        config="distractor",
        notes="Multi-hop QA benchmark for RAG answer and source support evaluation.",
    ),
    "cmrc2018": DatasetSpec(
        name="cmrc2018",
        hf_id="hfl/cmrc2018",
        suite="rag",
        split="validation",
        source_url="https://huggingface.co/datasets/hfl/cmrc2018",
        notes="Chinese machine reading comprehension benchmark.",
    ),
    "gsm8k": DatasetSpec(
        name="gsm8k",
        hf_id="openai/gsm8k",
        suite="agent",
        split="test",
        source_url="https://huggingface.co/datasets/openai/gsm8k",
        config="main",
        notes="Grade-school math benchmark for Python sandbox/tool execution tasks.",
    ),
    "bfcl": DatasetSpec(
        name="bfcl",
        hf_id="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        suite="agent",
        split="train",
        source_url="https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        notes="Function calling benchmark for tool selection and argument validity.",
    ),
}


def selected_specs(names: List[str]) -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []
    for name in names:
        normalized = name.strip().lower()
        if not normalized:
            continue
        if normalized not in DATASET_SPECS:
            available = ", ".join(sorted(DATASET_SPECS))
            raise KeyError(f"Unknown dataset '{name}'. Available datasets: {available}")
        specs.append(DATASET_SPECS[normalized])
    return specs
