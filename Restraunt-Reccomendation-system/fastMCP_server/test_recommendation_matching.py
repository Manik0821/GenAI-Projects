import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


module_path = Path(__file__).with_name("server.py")

spec = importlib.util.spec_from_file_location("restaurant_server", module_path)
restaurant_server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restaurant_server)


class RecommendationMatchingTests(unittest.TestCase):
    def test_get_article_db_falls_back_when_chroma_init_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(restaurant_server, "CHROMA_DB_PATH", Path(tmp_dir)), patch.object(restaurant_server, "Chroma", side_effect=RuntimeError("boom")):
                db = restaurant_server._get_article_db()
                self.assertIsNotNone(db, "Expected a fallback vector collection when Chroma initialization fails")
                self.assertTrue(hasattr(db, "_collection"))

    def test_indian_query_returns_structured_matches(self):
        payload = json.loads(restaurant_server.recommend_by_vibe("indian"))
        self.assertTrue(payload["structured_matches"], "Expected Indian query to return structured matches")

    def test_spicy_query_returns_structured_matches(self):
        payload = json.loads(restaurant_server.recommend_by_vibe("spicy"))
        self.assertTrue(payload["structured_matches"], "Expected spicy query to return structured matches")


if __name__ == "__main__":
    unittest.main()
