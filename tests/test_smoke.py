import unittest
import os
import sys
import shutil
from unittest.mock import patch

# Setup sys.path to easily import rag and agent modules
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import fitz
from PIL import Image

from rag.pipeline import init_pipeline, run_pipeline, _STATE
from agent.runner import run_agent

class TestSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(os.path.dirname(__file__), "test_data")
        cls.pdf_dir = os.path.join(cls.test_dir, "pdfs")
        cls.fig_dir = os.path.join(cls.test_dir, "figures")
        os.makedirs(cls.pdf_dir, exist_ok=True)
        os.makedirs(cls.fig_dir, exist_ok=True)
        
        # Create a tiny pdf
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "SQLENS pipeline is a test system.")
        doc.save(os.path.join(cls.pdf_dir, "dummy_doc.pdf"))
        doc.close()
        
        # Create a tiny image
        img = Image.new('RGB', (10, 10), color='white')
        img.save(os.path.join(cls.fig_dir, "dummy_fig.png"))
        
        # Reset state just in case
        global _STATE
        _STATE["initialized"] = False
        
        # Initialize pipeline pointing to test data
        init_pipeline(data_dir=cls.test_dir, logs_dir=os.path.join(cls.test_dir, "logs"), use_run_id=False)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        global _STATE
        _STATE["initialized"] = False

    def test_pipeline_smoke(self):
        # Basic run_pipeline test
        res = run_pipeline("What is SQLENS?", retrieval_mode="mm")
        self.assertIn("answer", res)
        self.assertIn("ctx", res)
        self.assertTrue(len(res["ctx"]["evidence"]) > 0)
        
    @patch('agent.runner.chat')
    def test_agent_smoke(self, mock_chat):
        # Mock LLM avoiding network calls
        mock_chat.return_value = {
            "ok": True,
            "content": "This is a mocked answer from the agent.",
            "tool_calls": []
        }
        res = run_agent("Hello agent!")
        self.assertIn("answer", res)
        self.assertEqual(res["answer"], "This is a mocked answer from the agent.")
        self.assertTrue(res["metrics"]["agent_ok"])

if __name__ == '__main__':
    unittest.main()
