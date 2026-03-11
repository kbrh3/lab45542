import json
import os
import sys

# Add root folder to sys.path so we can import from rag
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from rag.pipeline import init_pipeline, build_context

import erag

def text_generator(queries_and_documents):
    """
    Deterministically concatenates retrieved docs and returns a single lowercase string.
    queries_and_documents is a Dict[str, List[str]] (query -> list of documents)
    returns Dict[str, str] (query -> generated text)
    """
    outputs = {}
    for query, docs in queries_and_documents.items():
        outputs[query] = " ".join([str(d).lower() for d in docs])
    return outputs

def downstream_metric(generated_outputs, expected_outputs):
    """
    Returns a score in [0,1] using substring match: 
    1.0 if any expected answer appears in generated text, else 0.0.
    generated_outputs: Dict[str, str]
    expected_outputs: Dict[str, List[str]]
    returns: Dict[str, float]
    """
    scores = {}
    for query, gen_out in generated_outputs.items():
        exp_out = expected_outputs.get(query, [])
        matched = False
        for expected in exp_out:
            if expected.lower() in gen_out.lower():
                matched = True
                break
        scores[query] = 1.0 if matched else 0.0
    return scores

def run_erag_evaluation(gold_dataset_path: str, top_k: int = 5):
    """
    Script to run eRAG validation on our retriever outputs.
    """
    print(f"Loading eRAG gold set from: {gold_dataset_path}")
    with open(gold_dataset_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
        
    print(f"Loaded {len(gold_data)} queries.")
    
    # Initialize pipeline with project roots data directory to avoid missing data RuntimeError
    pipeline_state = init_pipeline(
        data_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data')),
        logs_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts', 'runs')),
    )
    
    retrieval_results = {}
    expected_outputs = {}
    
    for item in gold_data:
        q = item["query"]
        expected_concepts = item["expected"]
        
        # Save expected outputs for the evaluator
        expected_outputs[q] = expected_concepts
        
        # Call the actual retriever pipeline
        # `build_context` fetches text and img evidence using the pipeline state
        context = build_context(
            question=q, 
            top_k_text=top_k, 
            top_k_images=3, 
            top_k_evidence=8, 
            alpha=0.5, 
            use_multimodal=True
        )
        
        # We only want text chunks for the eRAG evaluator format
        texts = [ev["text"] for ev in context["evidence"] if ev["modality"] == "text"]
        retrieval_results[q] = texts
        
    print(f"\nSuccessfully evaluated {len(retrieval_results)} queries.")
    
    # Print a small preview of the first query for sanity check
    first_query = gold_data[0]["query"]
    print("\n--- Preview ---")
    print(f"Query: {first_query}")
    print(f"Expected Concepts: {expected_outputs[first_query]}")
    print("\nRetrieved snippets:")
    for i, snippet in enumerate(retrieval_results[first_query]):
        # Print a short truncated version of the snippet
        trunc_snippet = snippet[:150].replace('\n', ' ') + "..." if len(snippet) > 150 else snippet.replace('\n', ' ')
        print(f" [{i+1}] {trunc_snippet}")
        
    print("\n----------------\n")
    print("eRAG retrieval completed, running erag.eval()...")
    
    # Run erag.eval with the text_generator and downstream_metric
    eval_results = erag.eval(
        retrieval_results=retrieval_results,
        expected_outputs=expected_outputs,
        text_generator=text_generator,
        downstream_metric=downstream_metric,
        retrieval_metrics={"P_1", "P_3", "P_5"}
    )
    
    print("\n--- eRAG Evaluation Results ---")
    import pprint
    pprint.pprint(eval_results)
    
    return retrieval_results, expected_outputs, eval_results

if __name__ == "__main__":
    gold_path = os.path.join(os.path.dirname(__file__), "gold_erag.json")
    run_erag_evaluation(gold_path, top_k=5)
