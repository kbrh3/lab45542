"""
tool_schemas.py

Defines the OpenAI function calling / JSON schema style definitions for tools.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_query",
            "description": "Query the existing RAG pipeline to retrieve information and synthesized answers based on a knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to ask the RAG pipeline."
                    },
                    "retrieval_mode": {
                        "type": "string",
                        "enum": ["mm", "text_only"],
                        "description": "Mode of retrieval, 'mm' for multimodal or 'text_only' for text. Defaults to 'mm'."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top evidences to retrieve. Defaults to 5."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "snowflake_query",
            "description": "Executes a SQL statement against Snowflake and returns a preview of the results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL query to execute."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "The maximum number of rows to return. Defaults to 50."
                    }
                },
                "required": ["sql"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bills_analytics",
            "description": "Runs one of three pre-defined analytics queries on the BILLS_ANALYTICS Snowflake view.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["by_status", "by_year", "top_committees"],
                        "description": "The analytics metric to retrieve."
                    }
                },
                "required": ["metric"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_text",
            "description": "Provides a basic deterministic summarization of the given text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to summarize."
                    },
                    "max_words": {
                        "type": "integer",
                        "description": "Maximum words in the summary."
                    }
                },
                "required": ["text"]
            }
        }
    }
]
