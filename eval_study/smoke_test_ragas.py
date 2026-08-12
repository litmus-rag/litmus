"""Smoke test for the ragas + Azure gpt-5.4 + local-embeddings adapter, run once
before committing to full Study 2 generation. Writes a small testset to stdout."""

import asyncio
import os
import pathlib

import truststore

truststore.inject_into_ssl()

from langchain_core.documents import Document as LCDocument
from openai import AzureOpenAI
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import llm_factory
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from sentence_transformers import SentenceTransformer

from eval_study import config


class LocalEmbeddings(BaseRagasEmbeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.set_run_config(RunConfig())

    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True).tolist()[0]

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    async def aembed_query(self, text):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)

    async def aembed_documents(self, texts):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)


def build_ragas_llm():
    client = AzureOpenAI(
        api_key=os.environ["AZURE_API_KEY"],
        azure_endpoint=os.environ["AZURE_API_BASE"],
        api_version=os.environ["AZURE_API_VERSION"],
    )
    llm = llm_factory("gpt-5.4", provider="openai", client=client)
    # ragas's reasoning-model auto-detection breaks on dotted model ids like
    # "gpt-5.4" (int("5.4") raises inside its version-parsing regex fallback),
    # so max_tokens never gets remapped to max_completion_tokens automatically.
    # Bypass by setting model_args directly (same quirk CLAUDE.md documents
    # for azure_llm.py's direct calls).
    llm.model_args.pop("max_tokens", None)
    llm.model_args.pop("top_p", None)
    llm.model_args["max_completion_tokens"] = 3000
    llm.model_args["temperature"] = 1.0
    return llm


def main():
    llm = build_ragas_llm()
    embeddings = LocalEmbeddings()

    files = sorted(pathlib.Path(config.CORPUS_A_DOCS_DIR).glob("*.txt"))[:5]
    lc_docs = [LCDocument(page_content=f.read_text(), metadata={"filename": f.name}) for f in files]
    print(f"docs loaded: {len(lc_docs)}")

    gen = TestsetGenerator(llm=llm, embedding_model=embeddings)
    rc = RunConfig(seed=config.FIXED_SEED, max_workers=1, timeout=300)
    try:
        testset = gen.generate_with_langchain_docs(
            lc_docs, testset_size=3, run_config=rc, with_debugging_logs=True, raise_exceptions=True
        )
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    print("GENERATE RETURNED")
    df = testset.to_pandas()
    print(df.columns.tolist())
    print(df.head())


if __name__ == "__main__":
    main()
