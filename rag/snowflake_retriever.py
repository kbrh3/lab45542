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
    Uses environment variables for credentials and configuration.
    """
    try:
        # Fallbacks accommodate both standard SNOWFLAKE_* prefixes and direct generic names
        account = os.getenv("SNOWFLAKE_ACCOUNT")
        user = os.getenv("SNOWFLAKE_USER") or os.getenv("USER")
        password = os.getenv("SNOWFLAKE_PASSWORD") or os.getenv("PASSWORD")
        warehouse = os.getenv("SNOWFLAKE_WAREHOUSE") or os.getenv("WAREHOUSE")
        database = os.getenv("SNOWFLAKE_DATABASE") or os.getenv("DATABASE")
        schema = os.getenv("SNOWFLAKE_SCHEMA") or os.getenv("SCHEMA")

        if not all([account, user, password, warehouse, database, schema]):
            print("Warning: Missing one or more Snowflake environment variables.")

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
        query (str): The search query strings to match against TITLE or DESCRIPTION.
        top_k (int): Result limit.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries formatted exactly as expected by the RAG pipeline.
    """
    results = []
    conn = get_snowflake_connection()
    if not conn:
        print("Failed to procure a Snowflake connection. Retrieval aborted.")
        return results

    cursor = conn.cursor()
    try:
        # We use ILIKE with wildcards for simple keyword matching
        # Clean and wrap the query for SQL wildcard search
        keyword = f"%{query.strip()}%"
        
        sql = """
            SELECT 
                BILL_ID,
                BILL_NUMBER,
                TITLE,
                DESCRIPTION,
                STATUS,
                COMMITTEE,
                LAST_ACTION
            FROM POLICYPULSE_DB.PUBLIC.BILLS
            WHERE TITLE ILIKE %s OR DESCRIPTION ILIKE %s
            LIMIT %s
        """
        
        # Execute parameterized query to safely prevent SQL injection
        cursor.execute(sql, (keyword, keyword, top_k))
        
        for row in cursor.fetchall():
            # Extract row fields gracefully (handling potential Null values)
            bill_id = row[0]
            bill_number = row[1] or ""
            title = row[2] or ""
            description = row[3] or ""
            status = row[4] or ""
            committee = row[5] or ""
            last_action = row[6] or ""
            
            # Construct formatted RETRIEVAL_TEXT
            retrieval_text = (
                f"Bill Number: {bill_number}\n"
                f"Title: {title}\n"
                f"Description: {description}\n"
                f"Status: {status}\n"
                f"Committee: {committee}\n"
                f"Last Action: {last_action}"
            ).strip()
            
            # Format to pipeline context requirement
            result_item = {
                "modality": "text",
                "id": f"bill::{bill_id}",
                "raw_score": 1.0,
                "fused_score": 1.0, # Given equal weight conceptually 
                "text": retrieval_text,
                "path": None,
                "citation_tag": f"[bill::{bill_id}]",
                "source": "BILLS",
                "page_num": None
            }
            results.append(result_item)
            
    except (DatabaseError, ProgrammingError) as db_err:
        print(f"Snowflake query execution failed: {db_err}")
    except Exception as e:
        print(f"Unexpected error during retrieval: {e}")
    finally:
        # Ensure resources are completely closed down
        cursor.close()
        conn.close()
            
    return results
