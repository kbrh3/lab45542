"""
rag/snowflake_retriever.py

Connects to Snowflake to retrieve bill data from POLICYPULSE_DB.PUBLIC.BILLS
based on keyword match on TITLE and DESCRIPTION.
"""

import os
import snowflake.connector
from snowflake.connector.errors import DatabaseError, ProgrammingError
from typing import List, Dict, Any

def get_snowflake_connection():
    """
    Helper function to establish a connection to Snowflake.
    Uses explicit SNOWFLAKE_* environment variables for credentials and configuration.
    """
    try:
        # Use only standard SNOWFLAKE_* prefixes
        account = os.getenv("SNOWFLAKE_ACCOUNT")
        user = os.getenv("SNOWFLAKE_USER")
        password = os.getenv("SNOWFLAKE_PASSWORD")
        warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
        database = os.getenv("SNOWFLAKE_DATABASE")
        schema = os.getenv("SNOWFLAKE_SCHEMA")

        if not all([account, user, password, warehouse, database, schema]):
            print("Warning: Missing one or more explicit SNOWFLAKE_* environment variables.")

        conn = snowflake.connector.connect(
            account=account,
            user=user,
            password=password,
            warehouse=warehouse,
            database=database,
            schema=schema
        )
        return conn
    except Exception as e:
        print(f"Error connecting to Snowflake: {e}")
        return None

def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieves the top_k matching bills from Snowflake using keyword matching.
    
    Args:
        query (str): The search query to match against BILL_NUMBER, TITLE, DESCRIPTION,
                     STATUS_DESC, or COMMITTEE_ID.
        top_k (int): Result limit.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries formatted exactly as expected by the RAG pipeline.
    """
    results = []
    
    if not query or not query.strip():
        return results
        
    conn = get_snowflake_connection()
    if not conn:
        print("Failed to procure a Snowflake connection. Retrieval aborted.")
        return results

    cursor = None
    try:
        cursor = conn.cursor()
        
        # Use ILIKE with wildcards for simple keyword matching.
        # Query split into individual keywords to increase recall.
        keywords = [w for w in query.strip().split() if len(w) > 2]
        if not keywords:
            keywords = [query.strip()]
        
        # Build OR conditions per keyword across searchable text columns
        conditions = []
        params: List[Any] = []
        for kw in keywords:
            pattern = f"%{kw}%"
            conditions.append(
                "(BILL_NUMBER ILIKE %s OR TITLE ILIKE %s OR DESCRIPTION ILIKE %s OR STATUS_DESC ILIKE %s OR COMMITTEE_ID ILIKE %s)"
            )
            params.extend([pattern, pattern, pattern, pattern, pattern])
        
        params.append(top_k)
        
        sql = f"""
            SELECT 
                BILL_ID,
                BILL_NUMBER,
                TITLE,
                DESCRIPTION,
                STATUS,
                STATUS_DESC,
                STATUS_DATE,
                COMMITTEE_ID
            FROM BILLS
            WHERE {" OR ".join(conditions)}
            ORDER BY STATUS_DATE DESC NULLS LAST
            LIMIT %s
        """
        
        cursor.execute(sql, params)
        
        for row in cursor.fetchall():
            bill_id    = row[0]
            bill_number = row[1] or ""
            title       = row[2] or ""
            description = row[3] or ""
            status      = row[4] or ""
            status_desc = row[5] or ""
            status_date = str(row[6]) if row[6] else ""
            committee_id = row[7] or ""
            
            retrieval_text = (
                f"Bill Number: {bill_number}\n"
                f"Title: {title}\n"
                f"Description: {description}\n"
                f"Status: {status_desc} ({status})\n"
                f"Committee: {committee_id}\n"
                f"Status Date: {status_date}"
            ).strip()
            
            db_name = os.getenv("SNOWFLAKE_DATABASE", "POLICYPULSE_DB")
            schema_name = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
            result_item = {
                "modality": "text",
                "id": f"bill::{bill_id}",
                "raw_score": 1.0,
                "fused_score": 1.0,
                "text": retrieval_text,
                "path": None,
                "citation_tag": f"[bill::{bill_id}]",
                "source": f"{db_name}.{schema_name}.BILLS",
                "page_num": None
            }
            results.append(result_item)
            
    except (DatabaseError, ProgrammingError) as db_err:
        print(f"Snowflake query execution failed: {db_err}")
    except Exception as e:
        print(f"Unexpected error during retrieval: {e}")
    finally:
        # Ensure resources are completely closed down
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            
    return results
