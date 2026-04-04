import sys, os
from rag.snowflake_retriever import retrieve

if __name__ == "__main__":
    print(retrieve("What bills are related to education?"))
