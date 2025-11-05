import unittest
from utils.scorer import score_transcript_items, select_highlight_windows


class TestScorer(unittest.TestCase):
    def test_scoring_and_windows(self):
        # Synthetic transcript with a clear laugh/punchline
        items = [
            {"start": 0.0, "duration": 4.0, "text": "Halo semua, hari ini kita ngobrol santai."},
            {"start": 4.0, "duration": 3.0, "text": "Cerita sedikit tentang kejadian lucu kemarin!"},
            {"start": 7.0, "duration": 2.0, "text": "HAHA ngakak semua."},
            {"start": 9.0, "duration": 5.0, "text": "Oke lanjut ke topik berikutnya."},
        ]
        scores = score_transcript_items(items)
        self.assertEqual(len(scores), len(items))
        # The laugh line should have highest score
        self.assertGreater(scores[2], scores[0])
        self.assertGreater(scores[2], scores[1])

        windows = select_highlight_windows(items, target_duration=10, top_k=1)
        self.assertEqual(len(windows), 1)
        w = windows[0]
        self.assertGreater(w["end"], w["start"])  # positive window
        self.assertLessEqual(w["duration"], 14.0)  # with rolls shouldn't explode


if __name__ == "__main__":
    unittest.main()
