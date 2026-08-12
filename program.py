import litmus

# Step 1: generate an eval set from your docs
eval_set = litmus.generate(
    docs_dir="./sample1",
    tier="minimal",       # "minimal" | "medium" | "exhaustive"
)
eval_set.save("eval_set.json")

# Step 2: evaluate your RAG system against it
def my_rag(question: str) -> dict:
    # ... your retrieval + generation logic ...
    return {"answer": "...", "contexts": ["chunk text 1", "chunk text 2"]}

results = litmus.evaluate(eval_set, rag=my_rag)
results.summary()
