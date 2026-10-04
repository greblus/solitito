import unittest

import numpy as np

from diagnose_short_onset import latch_trace, diagnose_event


class OnsetDiagnosisTests(unittest.TestCase):
    def test_low_response_distinguished_from_unrearmed_second_attack(self):
        values = np.zeros((80, 12), np.float32)
        values[5, 2] = .9
        values[6:, 2] = .4
        values[40:44, 2] = .8
        event = {"id": "new-D", "pc": 2, "t": .64}
        trace = latch_trace(values, .7)
        result = diagnose_event(event, values, trace, .7)
        self.assertEqual(result["classification"], "latch_blocked")
        self.assertFalse(result["armed_at_peak"])
        values[40:44, 2] = .6
        result = diagnose_event(event, values, latch_trace(values, .7), .7)
        self.assertEqual(result["classification"], "response_below_threshold")

    def test_rearm_preserves_true_repluck_and_trace_matches_existing_latch(self):
        values = np.zeros((80, 12), np.float32)
        values[5, [2, 9]] = .9
        values[40, [2, 9]] = .8
        trace = latch_trace(values, .7)
        self.assertEqual(len(trace["events"]), 4)
        result = diagnose_event({"id": "new-D", "pc": 2, "t": .64}, values, trace, .7)
        self.assertEqual(result["classification"], "event_present_check_matching")
        self.assertTrue(result["armed_before_attack"])

    def test_late_response_is_not_forgiven_as_an_in_window_detection(self):
        values = np.zeros((80, 12), np.float32)
        values[55, 2] = .9
        result = diagnose_event({"id": "new-D", "pc": 2, "t": .64}, values, latch_trace(values, .7), .7)
        self.assertEqual(result["classification"], "response_below_threshold")
        self.assertEqual(len(result["same_pc_events"]), 1)


if __name__ == "__main__":
    unittest.main()
