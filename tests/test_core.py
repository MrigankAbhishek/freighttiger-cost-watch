import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from adversarial_cases import run_adversarial  # noqa: E402
from freight_watch.costlog import CostLog  # noqa: E402
from freight_watch.llm import Rewriter, is_faithful  # noqa: E402
from freight_watch.metrics import load_shipments, weekly_metrics  # noqa: E402
from freight_watch.notes import load_notes  # noqa: E402
from freight_watch.pipeline import run_pipeline  # noqa: E402


def make_shipments(rows):
    """rows: (route_origin, dest, type, date, qty, km, cost). Returns a loaded, validated frame."""
    df = pd.DataFrame([dict(shipment_id=f"S{i}", origin=o, destination=d, route_type=t, material="x",
                            quantity_tonnes=q, distance_km=k, freight_cost_inr=c, shipment_date=dt, transporter="t")
                       for i, (o, d, t, dt, q, k, c) in enumerate(rows)])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.csv"
        df.to_csv(path, index=False)
        return load_shipments(path)


class MetricsTests(unittest.TestCase):
    def test_week_is_monday_to_sunday(self):
        s = make_shipments([("A", "B", "Short", "2024-01-07", 10, 100, 1000),   # Sunday -> week of Jan 1
                            ("A", "B", "Short", "2024-01-08", 10, 100, 1000)])  # Monday -> week of Jan 8
        self.assertEqual(sorted(s["week_of"].dt.date.astype(str)), ["2024-01-01", "2024-01-08"])

    def test_cost_is_ratio_of_sums_not_mean_of_ratios(self):
        s = make_shipments([("A", "B", "Short", "2024-01-01", 10, 100, 1000), ("A", "B", "Short", "2024-01-02", 20, 100, 6000)])
        m = weekly_metrics(s)
        self.assertAlmostEqual(m.loc[0, "cost_per_tonne_km"], 7000 / 3000)

    def test_own_history_uses_only_prior_weeks_and_all_available_when_fewer_than_8(self):
        costs = [1000, 2000, 3000]  # one shipment per week, tonne_km = 1000 -> 1,2,3 per tonne-km
        s = make_shipments([("A", "B", "Short", f"2024-01-{1 + 7 * i:02d}", 10, 100, c) for i, c in enumerate(costs)])
        m = weekly_metrics(s)
        self.assertTrue(pd.isna(m.loc[0, "own_avg"]))
        self.assertAlmostEqual(m.loc[2, "own_avg"], 1.5)  # mean of weeks 1-2 only (not week 3 itself)
        self.assertEqual(int(m.loc[2, "n_prior_weeks"]), 2)

    def test_no_look_ahead(self):
        base = [("A", "B", "Short", f"2024-{1 + i // 4:02d}-{1 + 7 * (i % 4):02d}", 10, 100, 1000) for i in range(12)]
        changed = base[:-1] + [("A", "B", "Short", base[-1][3], 10, 100, 99999)]
        a, b = weekly_metrics(make_shipments(base)), weekly_metrics(make_shipments(changed))
        pd.testing.assert_series_equal(a["own_avg"].iloc[:-1], b["own_avg"].iloc[:-1])

    def test_rolling_window_is_exactly_8_weeks(self):
        s = make_shipments([("A", "B", "Short", (pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=i)).date().isoformat(),
                             10, 100, 1000 * (i + 1)) for i in range(10)])
        m = weekly_metrics(s)
        self.assertAlmostEqual(m.loc[9, "own_avg"], sum(range(2, 10)) / 8)  # weeks 2..9 (1-indexed values 2..9)

    def test_peer_average_excludes_the_route_itself(self):
        rows = []
        for route, cost in [("A", 1000), ("B", 2000), ("C", 4000)]:
            rows.append((route, "Z", "Medium", "2024-01-01", 10, 100, cost))
        m = weekly_metrics(make_shipments(rows)).set_index("route")
        self.assertAlmostEqual(m.loc["A-Z", "peer_avg"], (2 + 4) / 2)
        self.assertAlmostEqual(m.loc["C-Z", "peer_avg"], (1 + 2) / 2)

    def test_peer_only_same_route_type(self):
        rows = [("A", "Z", "Short", "2024-01-01", 10, 100, 1000), ("B", "Z", "Long", "2024-01-01", 10, 100, 9000)]
        m = weekly_metrics(make_shipments(rows))
        self.assertTrue(m["peer_avg"].isna().all())


class NotesAndGroundingTests(unittest.TestCase):
    def test_given_notes_are_parsed_as_expected(self):
        notes = {n.note_id: n for n in load_notes(ROOT / "data" / "context_notes.csv")}
        self.assertEqual((notes["N001"].window_start, notes["N001"].window_end), (date(2025, 2, 24), date(2025, 3, 8)))
        self.assertEqual(notes["N003"].window_kind, "step_change")
        self.assertEqual(notes["N003"].magnitude, (5.0, 7.0))
        self.assertEqual(notes["N002"].window_kind, "single_week")

    def test_adversarial_traps_and_controls(self):
        for cid, ok, desc, got in run_adversarial():
            self.assertTrue(ok, f"{cid} {desc} (matched={got})")


class LLMGuardrailTests(unittest.TestCase):
    DRAFT = "Justified. Matches note N002 dated 2025-01-20: festival surcharge. The cost rise has a clear explanation."

    def test_faithfulness_rules(self):
        self.assertTrue(is_faithful(self.DRAFT, "Justified. Note N002 (2025-01-20) reports a festival surcharge."))
        self.assertFalse(is_faithful(self.DRAFT, "Justified. Note N003 explains it."))            # other note id
        self.assertFalse(is_faithful(self.DRAFT, "Justified. N002 on 2025-01-20 raised costs 40%."))  # invented number
        self.assertFalse(is_faithful(self.DRAFT, "Unexplained. N002 2025-01-20."))                 # flipped verdict
        self.assertFalse(is_faithful(self.DRAFT, "Justified. A festival surcharge applied."))       # dropped the note id

    def _rewriter(self, reply):
        log = CostLog()
        fake = lambda payload: {"choices": [{"message": {"content": reply}}], "usage": {"prompt_tokens": 100, "completion_tokens": 20}}  # noqa: E731
        return Rewriter("live", "fake-model", log, transport=fake), log

    def test_good_rewrite_is_used_and_tokens_counted(self):
        rw, log = self._rewriter("Justified. N002 (2025-01-20) reports a festival surcharge.")
        self.assertIn("festival surcharge", rw.rewrite(self.DRAFT))
        self.assertEqual((log.llm_calls, log.input_tokens, log.output_tokens), (1, 100, 20))

    def test_bad_rewrite_falls_back_to_template(self):
        rw, log = self._rewriter("Justified. N009 explains this with 99%.")
        self.assertEqual(rw.rewrite(self.DRAFT), self.DRAFT)
        self.assertEqual(log.fallbacks_to_template, 1)

    def test_transport_failure_falls_back_to_template(self):
        def boom(payload):
            raise OSError("no server")
        log = CostLog()
        self.assertEqual(Rewriter("live", "m", log, transport=boom).rewrite(self.DRAFT), self.DRAFT)


class DeterminismTest(unittest.TestCase):
    def test_two_runs_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            run_pipeline(ROOT / "data", a)
            run_pipeline(ROOT / "data", b)
            for name in ["flagged_routes.csv", "weekly_metrics_all.csv", "evidence_trace.csv"]:
                self.assertEqual((Path(a) / name).read_bytes(), (Path(b) / name).read_bytes(), name)


if __name__ == "__main__":
    unittest.main()
