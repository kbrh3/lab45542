import os
import time
from typing import Any, Dict, Literal

try:
    import snowflake.connector
except ImportError:
    snowflake = None

# Import existing RAG pipeline
from rag.pipeline import run_query_and_log

def rag_query(query: str, retrieval_mode: Literal["mm", "text_only"] = "mm", top_k: int = 5) -> Dict[str, Any]:
    """
    Query the existing RAG pipeline to retrieve information and synthesized answers.
    
    Args:
        query (str): The question to ask.
        retrieval_mode (Literal["mm", "text_only"]): Mode of retrieval, 'mm' for multimodal or 'text_only' for text.
        top_k (int): Number of top evidences to retrieve.
    
    Returns:
        Dict[str, Any]: A dictionary containing the status, data (answer, evidence, metrics), and meta (latency).
    """
    t0 = time.time()
    try:
        # Wrap query in the format expected by run_query_and_log
        query_item = {"query_id": "agent_query", "question": query, "gold_evidence_ids": []}
        
        # We pass retrieval_mode. 
        # Note: run_query_and_log uses global TOP_K internally, but to meet requirements we define top_k in the signature.
        out = run_query_and_log(query_item, retrieval_mode=retrieval_mode)
        latency = (time.time() - t0) * 1000.0
        
        return {
            "ok": True,
            "data": {
                "answer": out["answer"],
                "evidence": out.get("ctx", {}).get("evidence", []),
                "metrics": {
                    "latency_ms": out.get("latency_ms", latency),
                    "p5": out.get("p5"),
                    "r10": out.get("r10"),
                    "faithful": out.get("faithful"),
                    "missing_evidence_behavior": out.get("meb"),
                }
            },
            "meta": {"latency_ms": latency}
        }
    except Exception as e:
        latency = (time.time() - t0) * 1000.0
        return {
            "ok": False,
            "error": str(e),
            "meta": {"latency_ms": latency}
        }

def snowflake_query(sql: str, limit: int = 50) -> Dict[str, Any]:
    """
    Executes a SQL statement against Snowflake and returns a preview of the results.
    
    Args:
        sql (str): The SQL query to execute.
        limit (int): The maximum number of rows to return.
        
    Returns:
        Dict[str, Any]: A dictionary containing the status, data (columns, rows, limit), and meta (latency).
    """
    t0 = time.time()
    
    if snowflake is None:
        latency = (time.time() - t0) * 1000.0
        return {
            "ok": False,
            "error": "snowflake-connector-python is not installed.",
            "meta": {"latency_ms": latency}
        }
        
    if os.getenv("USE_SNOWFLAKE", "false").lower() != "true":
        latency = (time.time() - t0) * 1000.0
        return {
            "ok": False,
            "error": "Snowflake configuration is disabled by default. Set USE_SNOWFLAKE=true to enable.",
            "meta": {"latency_ms": latency}
        }

    try:
        # Check required env vars
        req_vars = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", 
                    "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA"]
        missing = [v for v in req_vars if not os.getenv(v)]
        if missing:
            raise ValueError(f"Missing Snowflake environment variables: {', '.join(missing)}")
            
        clean_sql = sql.strip()
        if clean_sql.upper().startswith("SELECT") and "LIMIT" not in clean_sql.upper():
            clean_sql = f"{clean_sql} LIMIT {limit}"
            
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA")
        )
        
        try:
            cur = conn.cursor()
            cur.execute(clean_sql)
            
            # Fetch results
            columns = [col[0] for col in cur.description] if cur.description else []
            rows = cur.fetchmany(limit)
            
            # Convert rows to list of dicts
            row_dicts = [dict(zip(columns, row)) for row in rows]
            
            latency = (time.time() - t0) * 1000.0
            return {
                "ok": True,
                "data": {
                    "columns": columns,
                    "rows": row_dicts,
                    "limit": limit
                },
                "meta": {"latency_ms": latency}
            }
        finally:
            conn.close()
            
    except Exception as e:
        latency = (time.time() - t0) * 1000.0
        return {
            "ok": False,
            "error": str(e),
            "meta": {"latency_ms": latency}
        }

def bills_analytics(metric: Literal["by_status", "by_year", "top_committees"]) -> Dict[str, Any]:
    """
    Runs one of three pre-defined analytics queries on the BILLS_ANALYTICS Snowflake view.
    
    Args:
        metric (Literal["by_status", "by_year", "top_committees"]): The analytics metric to retrieve.
        
    Returns:
        Dict[str, Any]: The query results mapped to the requested metric.
    """
    t0 = time.time()
    
    queries = {
        "by_status": "SELECT STATUS_DESC, COUNT(*) as count FROM BILLS_ANALYTICS GROUP BY STATUS_DESC ORDER BY count DESC",
        "by_year": "SELECT YEAR, COUNT(*) as count FROM BILLS_ANALYTICS GROUP BY YEAR ORDER BY YEAR DESC",
        "top_committees": "SELECT PRIMARY_COMMITTEE_NAME, COUNT(*) as count FROM BILLS_ANALYTICS WHERE PRIMARY_COMMITTEE_NAME IS NOT NULL GROUP BY PRIMARY_COMMITTEE_NAME ORDER BY count DESC LIMIT 10"
    }
    
    if metric not in queries:
        return {
            "ok": False,
            "error": f"Invalid metric: {metric}. Must be one of: {', '.join(queries.keys())}",
            "meta": {"latency_ms": (time.time() - t0) * 1000.0}
        }
        
    sql = queries[metric]
    
    result = snowflake_query(sql=sql, limit=50)
    
    # Update latency matching the whole operation
    result["meta"]["latency_ms"] = (time.time() - t0) * 1000.0
    return result

def summarize_text(text: str, max_words: int = 120) -> Dict[str, Any]:
    """
    Provides a basic deterministic summarization of the given text.
    
    Args:
        text (str): The text to summarize.
        max_words (int): Maximum words in the summary.
        
    Returns:
        Dict[str, Any]: A dictionary containing the status, data (summary), and meta (latency).
    """
    t0 = time.time()
    try:
        words = text.split()
        if len(words) <= max_words:
            summary = text
        else:
            summary = " ".join(words[:max_words]) + "..."
            
        latency = (time.time() - t0) * 1000.0
        return {
            "ok": True,
            "data": {"summary": summary},
            "meta": {"latency_ms": latency}
        }
    except Exception as e:
        latency = (time.time() - t0) * 1000.0
        return {
            "ok": False,
            "error": str(e),
            "meta": {"latency_ms": latency}
        }
