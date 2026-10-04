"""Sustained-class attribution must not punish true re-plucks or strums."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from onset_events import Event
from onset_ringing import ringing_annotations, ringing_mask, ringing_event_counts, ringing_acceptance
from train_short_onset import onset_targets, OnsetBlocks, onset_metrics, FEATURE_DIM


def note(name, t, pc, end=4., string=0):
    return {"id": name, "t": t, "pc": pc, "end": end, "string": string}


def source(events, domain="synthetic"):
    return {"id": "fixture", "duration": 4., "events": events, "domain": domain,
            "case": "root_plus_fifth", "group": "background1"}


class RingingTests(unittest.TestCase):
    def test_new_a_weights_only_old_d_in_existing_target_window(self):
        s = source([note("D", .5, 2), note("A", 1., 9)])
        target = onset_targets(s["events"], 250)
        mask, audit = ringing_mask(s, 250, target)
        self.assertEqual(audit["eligible"], 1)
        self.assertEqual(np.flatnonzero(mask[:, 2]).tolist(), list(range(62, 68)))
        self.assertFalse(mask[:, 9].any())
        self.assertFalse(mask[target > 0].any())
        self.assertFalse(mask[68:].any())

    def test_repluck_octave_unison_and_strum_are_not_negative(self):
        for delay in (0., .012, .025, .096, .120):
            for string in (0, 3):
                s = source([note("D", .5, 2), note("A", 1., 9, string=1),
                            note("new-D-or-octave", 1. + delay, 2, string=string)])
                mask, audit = ringing_mask(s, 250, onset_targets(s["events"], 250))
                self.assertFalse(mask[:70, 2].any())
                self.assertGreater(audit["excluded_same_pc_attack"], 0)

    def test_ended_or_unknown_guitarset_note_not_assumed_ringing(self):
        for end in (.8, 1.):
            s = source([note("D", .5, 2, end), note("A", 1., 9, string=1)], "guitarset")
            self.assertEqual(ringing_annotations(s)[0], [])
        s["events"][0].pop("end")
        opportunities, audit = ringing_annotations(s)
        self.assertEqual(opportunities, [])
        self.assertEqual(audit["missing_activity_end"], 1)

    def test_string_conflict_excluded_but_independent_string_is_valid(self):
        s = source([note("D", .5, 2), note("A", 1., 9)], "guitarset")
        self.assertEqual(ringing_annotations(s)[1]["excluded_string_conflict"], 1)
        s["events"][1]["string"] = 1
        self.assertEqual(ringing_annotations(s)[1]["eligible"], 1)
        s["events"].insert(1, note("other-D", .5, 2, string=3))
        self.assertEqual(len(ringing_annotations(s)[0]), 1)  # PC, not string count

    def test_repeat_and_late_first_and_foreign_classes_are_separate(self):
        s = source([note("D", .5, 2), note("A", 1., 9)])
        d = Event("detection-D", .528, 2)
        repeat = Event("repeat-D", 1.024, 2)
        foreign = Event("foreign", 1.024, 4)
        pair = (Event("D", .5, 2), d)
        c, _ = ringing_event_counts(s, [d, repeat, foreign], [pair])
        self.assertEqual((c["ringing_false_events"], c["ringing_repeat_events"], c["foreign_pc_events"]), (1, 1, 1))
        c, _ = ringing_event_counts(s, [repeat], [])
        self.assertEqual(c["ringing_late_first_events"], 1)
        # A valid matched re-pluck cannot become an error, even with cached opportunities.
        c, _ = ringing_event_counts(s, [repeat], [(Event("new-D", 1., 2), repeat)])
        self.assertEqual(c["ringing_false_events"], 0)

    def test_later_conflicting_pitch_truncates_earlier_weight_window(self):
        s = source([note("D", .5, 2, string=0), note("A", 1., 9, string=1),
                    note("F-on-old-string", 1.05, 5, string=0)], "guitarset")
        mask, _ = ringing_mask(s, 250, onset_targets(s["events"], 250))
        self.assertTrue(mask[62:65, 2].all())
        self.assertFalse(mask[65:, 2].any())

    def test_simultaneous_other_notes_count_one_opportunity_and_event(self):
        s = source([note("D", .5, 2), note("A", 1., 9), note("F", 1., 5), note("C", 1.03, 0)])
        c, _ = ringing_event_counts(s, [Event("false-D", 1.04, 2)], [])
        self.assertEqual(c["ringing_opportunities"], 2)
        self.assertEqual(c["ringing_false_events"], 1)
        self.assertEqual(c["ringing_affected_opportunities"], 2)

    def test_delaying_false_event_past_target_cannot_hide_it(self):
        s = source([note("D", .5, 2), note("A", 1., 9)])
        c, _ = ringing_event_counts(s, [Event("late-D", 1.112, 2)], [])
        self.assertEqual(c["ringing_false_events"], 1)
        c, _ = ringing_event_counts(s, [Event("much-later-D", 1.8, 2)], [])
        self.assertEqual(c["held_pc_false_events_any_time"], 1)
        self.assertEqual(c["ringing_false_events"], 0)

    def test_groups_are_unique_across_clips_and_midfile_errors_not_tail_only(self):
        s = source([note("D", .5, 2), note("A", 1., 9), note("E", 3., 4)])
        values = np.zeros((250, 12), dtype=np.float32)
        values[32, 2] = .95
        values[63, [2, 9]] = .95
        values[189, 4] = .95
        result, _ = onset_metrics([(s, values), (dict(s, id="paired-clip"), values)], .8)
        c = result["groups"]["all"]
        self.assertEqual(c["ringing_repeat_events"], 2)
        self.assertEqual(len(c["error_source_groups"]), 1)
        self.assertEqual(c["tail_extra"], 0)

    def test_block_boundary_keeps_loss_mask_and_padding_unweighted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npy"
            np.save(path, np.ones((140, FEATURE_DIM), np.float16))
            s = source([note("D", .5, 2), note("A", 2., 9)])
            s.update(features=str(path), frames=140, split="train")
            weighted = OnsetBlocks([s], 4.)
            control = OnsetBlocks([s], 1.)
            self.assertTrue(np.all(control[0][3] == 1))
            self.assertEqual(weighted[0][3][2, 125:128].tolist(), [4.] * 3)
            self.assertEqual(weighted[1][3][2, :3].tolist(), [4.] * 3)
            self.assertTrue(np.all(weighted[1][3][:, 12:] == 1))
            empty = dict(s, events=[])
            self.assertFalse(OnsetBlocks([empty], 4.).ringing[0].any())

    def test_acceptance_does_not_reward_silence_or_empty_control(self):
        baseline = {"groups": {k: {"ringing_false_events": 10, "ringing_opportunities": 20,
                                    "held_pc_false_events_any_time": 12,
                                    "challenge_tp": 10, "recall": .7, "latency_p95": .064}
                                for k in ("all", "synthetic", "synthetic/root_repluck",
                                          "synthetic/triad_repluck", "guitarset/comp")}}
        candidate = copy.deepcopy(baseline)
        candidate["groups"]["all"]["ringing_false_events"] = 5
        self.assertTrue(ringing_acceptance(candidate, baseline)["accepted"])
        candidate["groups"]["synthetic/root_repluck"]["challenge_tp"] = 9
        self.assertFalse(ringing_acceptance(candidate, baseline)["accepted"])
        baseline["groups"]["all"]["ringing_false_events"] = 0
        candidate = copy.deepcopy(baseline)
        self.assertFalse(ringing_acceptance(candidate, baseline)["accepted"])


if __name__ == "__main__":
    unittest.main()
