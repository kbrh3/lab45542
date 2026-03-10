# eRAG Reproduction Steps

This folder contains scripts to run an eRAG-style (evaluative RAG) verification on our base retriever. 

## Steps
1. **Configure Data**: Make sure the actual vector database or TF-IDF matrices (if running the mock DB) are available to the script.
2. **Review Gold Set**: Update `gold_erag.json` if necessary to match the actual knowledge base you are testing against. It expects simple queries and a list of `required_concepts`.
3. **Execute Evaluation**: 
   Run from this directory:
   ```bash
   python erag_run_on_our_retrieval.py
   ```
4. **Analyze Output**: Check the output to see if the required concepts were actually retrieved by `rag/retriever.py:build_context`.
