import unittest
import os
import sys
import shutil
from unittest.mock import patch, MagicMock

# Base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Mock out snowflake dependencies to prevent ModuleNotFoundError in testing environments
sys.modules['snowflake'] = MagicMock()
sys.modules['snowflake.connector'] = MagicMock()
sys.modules['snowflake.connector.errors'] = MagicMock()

class TestSmoke(unittest.TestCase):
    
    def test_imports_resolve(self):
        """Verify that backend modules and agents import correctly without circular dependencies."""
        try:
            import rag
            from rag.pipeline import init_pipeline, run_pipeline, build_context
            from rag.state import get_state
            import agent.tool_registry
            import agent.runner
            from rag import snowflake_retriever
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Imports failed: {e}")

    def test_state_initialization(self):
        """Verify init_pipeline and get_state run successfully without local data directories."""
        from rag.state import init_pipeline, get_state
        
        # Test directory for logs
        test_dir = os.path.join(os.path.dirname(__file__), "test_data")
        logs_dir = os.path.join(test_dir, "logs")
        
        # rag.state.init_pipeline builds log routes natively
        state = init_pipeline(logs_dir=logs_dir)
        self.assertTrue(state.get("initialized"))
        self.assertEqual(state.get("logs_dir"), logs_dir)
        
        # get_state should return same structure
        global_state = get_state()
        self.assertTrue(global_state.get("initialized"))
        self.assertEqual(global_state.get("logs_dir"), logs_dir)
        
        # cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            
    @patch('rag.snowflake_retriever.retrieve')
    def test_pipeline_smoke(self, mock_retrieve):
        """Verify classic retrieval pipeline builds correctly via Snowflake mocks."""
        from rag.pipeline import run_pipeline
        
        # Mock what the snowflake retriever would return
        mock_retrieve.return_value = [
            {"id": "test_1", "text": "Mocked policy text.", "modality": "text", "citation_tag": "[TEST1]"}
        ]
        
        res = run_pipeline("What is test policy?", retrieval_mode="mm")
        
        self.assertIn("answer", res)
        self.assertIn("ctx", res)
        self.assertEqual(len(res["ctx"]["evidence"]), 1)
        self.assertIn("Mocked policy text", res["ctx"]["context"])
        
    @patch('agent.runner.chat')
    def test_agent_smoke(self, mock_chat):
        """Verify agent runner correctly interacts with mocked LLM."""
        from agent.runner import run_agent
        
        # Mock LLM avoiding network calls
        mock_chat.return_value = {
            "ok": True,
            "content": "This is a mocked answer from the agent.",
            "tool_calls": []
        }
        res = run_agent("Hello agent!")
        self.assertIn("answer", res)
        self.assertEqual(res["answer"], "This is a mocked answer from the agent.")
        self.assertTrue(res["metrics"].get("agent_ok"))

if __name__ == '__main__':
    unittest.main()
