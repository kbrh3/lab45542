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
            schema=schema,
            role="TRAINING_ROLE"
        )
        return conn
    except DatabaseError as db_err:
        err_msg = str(db_err).lower()
        if "incorrect username or password" in err_msg or "authentication" in err_msg or "not found: post url" in err_msg or "404" in err_msg:
            print(f"[Snowflake Auth Error] Login/authentication failed: {db_err}")
        elif "warehouse" in err_msg:
            print(f"[Snowflake Warehouse Error] Warehouse not found or not authorized: {db_err}")
        elif "database" in err_msg or "catalog" in err_msg:
            print(f"[Snowflake Database Error] Database not found or not authorized: {db_err}")
        elif "schema" in err_msg:
            print(f"[Snowflake Schema Error] Schema not found or not authorized: {db_err}")
        else:
            print(f"[Snowflake Connection Error] {db_err}")
        return None
    except Exception as e:
        print(f"Error connecting to Snowflake: {e}")
        return None

def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieves the top_k matching bills from Snowflake using keyword matching.
    
    Args:
        query (str): The search query strings to match against BILL_NUMBER, TITLE, DESCRIPTION, STATUS_DESC, COMMITTEE, or LAST_ACTION.
        top_k (int): Result limit.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries formatted exactly as expected by the RAG pipeline.
    """
    results = []
    
    print(f"--- DEBUG: rag/snowflake_retriever.py -> retrieve() called querying Snowflake for: {query}")
    print(f"--- DEBUG: Original query: {query}")
    
    if not query or not query.strip():
        return results
        
    conn = get_snowflake_connection()
    if not conn:
        print("Failed to procure a Snowflake connection. Retrieval aborted.")
        return results

    cursor = None
    try:
        cursor = conn.cursor()
        
        # Extract meaningful keywords
        stop_words = {"what", "are", "related", "to", "the", "is", "of", "in", "for", "on", "a", "an", "and", "or", "with", "bills", "bill"}
        words = query.strip().lower().replace("?", "").replace(".", "").split()
        keywords = [w for w in words if w not in stop_words]
        
        print(f"--- DEBUG: Extracted keywords: {keywords}")
        
        # Build parameters like: "%education%"
        if keywords:
            keyword = f"%{keywords[0]}%" 
        else:
            keyword = f"%{query.strip().lower()}%"
            
        sql = """
            SELECT
                BILL_ID,
                BILL_NUMBER,
                TITLE,
                DESCRIPTION,
                STATUS_DESC,
                COMMITTEE,
                LAST_ACTION_DATE
            FROM POLICYPULSE_DB.PUBLIC.BILLS
            WHERE
                TITLE ILIKE %s OR
                DESCRIPTION ILIKE %s OR
                STATUS_DESC ILIKE %s OR
                COMMITTEE ILIKE %s
            ORDER BY STATUS_DATE DESC
            LIMIT %s
        """
        
        # Exact params mapping to the 4 ILIKE conditions + LIMIT
        params = (keyword, keyword, keyword, keyword, top_k)
        
        print(f"--- DEBUG: Final SQL parameters: {params}")
        
        cursor.execute(sql, params)
        print("--- DEBUG: cursor.execute() succeeded.")
        
        all_rows = cursor.fetchall()
        print(f"--- DEBUG: Number of rows returned: {len(all_rows)}")
        print(f"--- DEBUG: First 2 raw rows: {all_rows[:2]}")
        
        for row in all_rows:
            # Match schema: BILL_ID, BILL_NUMBER, TITLE, DESCRIPTION, STATUS_DESC, COMMITTEE, LAST_ACTION_DATE
            bill_id = row[0]
            bill_number = row[1] or ""
            title = row[2] or ""
            description = row[3] or ""
            status_desc = row[4] or ""
            committee = row[5] or ""
            last_action_date = row[6] or ""
            
            # Construct formatted RETRIEVAL_TEXT requiring TITLE, DESCRIPTION, STATUS_DESC
            retrieval_text = (
                f"Bill Number: {bill_number}\n"
                f"Title: {title}\n"
                f"Description: {description}\n"
                f"Status: {status_desc}\n"
                f"Committee: {committee}\n"
                f"Last Action Date: {last_action_date}"
            ).strip()
            
            # Format to pipeline context requirement
            db_name = os.getenv("SNOWFLAKE_DATABASE", "UNKNOWN")
            schema_name = os.getenv("SNOWFLAKE_SCHEMA", "UNKNOWN")
            result_item = {
                "modality": "text",
                "id": f"bill::{bill_id}",
                "raw_score": 1.0,
                "fused_score": 1.0, # Required
                "text": retrieval_text,
                "path": None,
                "citation_tag": f"[bill::{bill_id}]", # Required
                "source": f"{db_name}.{schema_name}.BILLS",
                "page_num": None
            }
            results.append(result_item)
            
        print(f"--- DEBUG: Number of evidence objects built: {len(results)}")
            
    except snowflake.connector.errors.ProgrammingError as db_err:
        print(f"--- DEBUG: Snowflake SQL Compilation/Execution Error: {db_err}")
    except snowflake.connector.errors.DatabaseError as db_err:
        print(f"--- DEBUG: Snowflake Database Error: {db_err}")
    except Exception as e:
        print(f"Unexpected error during retrieval: {e}")
    finally:
        # Ensure resources are completely closed down
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            
    return results
