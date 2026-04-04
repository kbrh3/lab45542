import os
import snowflake.connector
from dotenv import load_dotenv

load_dotenv("c:/Users/kaily/lab45542/.env")

def main():
    try:
        conn = snowflake.connector.connect(
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA"),
            role="TRAINING_ROLE"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM POLICYPULSE_DB.PUBLIC.BILLS LIMIT 1")
        columns = [desc[0] for desc in cursor.description]
        print("Columns:", columns)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
