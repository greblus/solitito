"""Annotation-only diagnostics and loss masks; never an inference input.

Opportunities are (new attack time, old pitch class), not individual strings.
Synthetic stems continue to the file end; GuitarSet activity uses note ends.
Neither definition establishes the acoustic audibility of a decaying tail.
"""

from bisect import bisect_left, bisect_right

import numpy as np


RINGING_WINDOW = .096
RINGING_GUARD = .032
RINGING_EVAL_WINDOW = .128
RINGING_NEGATIVE_WEIGHT = 4.
RINGING_SPEC = {
    "version": "held-pc-negative-v1", "negative_weight": RINGING_NEGATIVE_WEIGHT,
    "training_window_seconds": RINGING_WINDOW, "evaluation_window_seconds": RINGING_EVAL_WINDOW,
    "same_pc_exclusion": "onset in [attack-96ms, attack+128ms]; positives always protected",
    "activity": "GuitarSet note end; synthetic excitation support until file end, not audibility",
    "opportunity": "unique other-attack time and previously active pitch class, older than 96ms",
    "ambiguity": "exclude conflicting/missing GuitarSet string activity from extra loss only",
    "selection": "validation only: half held-PC errors, no increase at other times, preserved re-plucks, comp recall -2pp, P95 +16ms",
    "empty_control": "zero control held-PC errors is inconclusive, never an accepted improvement",
}


def ringing_annotations(source):
    events = sorted(source["events"], key=lambda e: (e["t"], e["id"]))
    event_times = [e["t"] for e in events]
    by_pc = [[e for e in events if e["pc"] == pc] for pc in range(12)]
    times = [[e["t"] for e in group] for group in by_pc]
    synthetic = source.get("domain") == "synthetic"
    active, opportunities = [], []
    audit = {"eligible": 0, "excluded_same_pc_attack": 0,
             "excluded_string_conflict": 0, "missing_activity_end": 0}
    for event in events:
        if not synthetic and "end" not in event:
            audit["missing_activity_end"] += 1
    for attack in sorted({e["t"] for e in events}):
        active = [e for e in active if e.get("end", source["duration"] if synthetic else e["t"]) > attack]
        # Includes all strings at this instant. Short strums are protected below.
        new = events[bisect_left(event_times, attack):bisect_right(event_times, attack)]
        old_pcs = {e["pc"] for e in active if e["t"] + RINGING_WINDOW < attack - 1e-9}
        for pc in sorted(old_pcs):
            # Exclude the entire opportunity near a real same-PC start, including
            # an upcoming strum member. All ordinary targets/scoring remain intact.
            nearby = times[pc][bisect_left(times[pc], attack - RINGING_WINDOW - 1e-9):
                               bisect_right(times[pc], attack + RINGING_WINDOW + RINGING_GUARD + 1e-9)]
            if nearby:
                audit["excluded_same_pc_attack"] += 1
                continue
            held = [e for e in active if e["pc"] == pc]
            if not synthetic:
                # Conflicting pitches on one annotated string do not prove sustain.
                held = [e for e in held if "string" in e and not any(
                    o.get("string") == e["string"] and o["pc"] != pc
                    for o in active + new)]
            if not held:
                audit["excluded_string_conflict"] += 1
                continue
            ends = []
            for old in held:
                end = old.get("end", source["duration"])
                if not synthetic:
                    # Another pitch later in this window can terminate this string
                    # before its overlapping annotation ends. Do not weight that gap.
                    end = min([end] + [e["t"] for e in events if attack < e["t"] < end
                                       and e.get("string") == old["string"] and e["pc"] != pc])
                ends.append(end)
            support_end = max(ends)
            end = min(attack + RINGING_WINDOW, support_end)
            opportunities.append({"t": attack, "end": end, "pc": pc,
                                  "evaluation_end": min(attack + RINGING_EVAL_WINDOW, support_end),
                                  "old_ids": [e["id"] for e in held]})
            audit["eligible"] += 1
        active.extend(new)
    return opportunities, audit


def ringing_mask(source, frames, targets, hop_seconds=.016):
    opportunities, audit = ringing_annotations(source)
    times = (np.arange(frames) + 1) * hop_seconds
    mask = np.zeros((frames, 12), dtype=bool)
    for opportunity in opportunities:
        mask[(times > opportunity["t"] + 1e-9) &
             (times <= opportunity["end"] + 1e-9), opportunity["pc"]] = True
    # A positive on ANY string/octave always wins over the extra negative weight.
    mask &= targets == 0
    return mask, {**audit, "weighted_frames_classes": int(mask.sum())}


def ringing_event_counts(source, predicted, pairs):
    opportunities, audit = source.get("ringing_annotations", (None, None))
    if opportunities is None:
        opportunities, audit = ringing_annotations(source)
    matched = {p.id for _, p in pairs}
    detection_by_ref = {r.id: p.t for r, p in pairs}
    counts = {"ringing_opportunities": len(opportunities), "ringing_false_events": 0,
              "ringing_repeat_events": 0, "ringing_late_first_events": 0,
              "ringing_affected_opportunities": 0, "foreign_pc_events": 0,
              "held_pc_false_events_any_time": 0}
    affected = set()
    for prediction in predicted:
        if prediction.id in matched:
            continue
        held = [e for e in source["events"] if e["pc"] == prediction.pc
                and e["t"] + RINGING_WINDOW < prediction.t
                and prediction.t < e.get("end", source["duration"] if source.get("domain") == "synthetic" else e["t"])]
        counts["held_pc_false_events_any_time"] += bool(held)
        hits = [(i, o) for i, o in enumerate(opportunities)
                if o["pc"] == prediction.pc and o["t"] < prediction.t <= o["evaluation_end"] + 1e-9]
        if hits:
            counts["ringing_false_events"] += 1
            repeated = any(detection_by_ref.get(eid, float("inf")) < prediction.t
                           for _, o in hits for eid in o["old_ids"])
            counts["ringing_repeat_events" if repeated else "ringing_late_first_events"] += 1
            affected.update(i for i, _ in hits)
        elif not any(e["pc"] == prediction.pc and e["t"] <= prediction.t for e in source["events"]):
            counts["foreign_pc_events"] += 1
    counts["ringing_affected_opportunities"] = len(affected)
    return counts, audit


def ringing_acceptance(candidate, control):
    """Predeclared validation constraints; an inconclusive control cannot pass."""
    c, b = candidate["groups"], control["groups"]
    required = ("all", "synthetic", "synthetic/root_repluck", "synthetic/triad_repluck", "guitarset/comp")
    if any(key not in c or key not in b for key in required):
        return {"accepted": False, "checks": {"required_groups_present": False}}
    old = b["all"]["ringing_false_events"]
    checks = {
        "measurable_control_errors": old > 0,
        "half_as_many_held_pc_errors": c["all"]["ringing_false_events"] <= old / 2,
        "same_opportunity_count": c["all"]["ringing_opportunities"] == b["all"]["ringing_opportunities"],
        "no_increase_in_all_held_pc_errors": c["all"]["held_pc_false_events_any_time"] <= b["all"]["held_pc_false_events_any_time"],
        "root_repluck_preserved": c["synthetic/root_repluck"]["challenge_tp"] >= b["synthetic/root_repluck"]["challenge_tp"],
        "triad_repluck_preserved": c["synthetic/triad_repluck"]["challenge_tp"] >= b["synthetic/triad_repluck"]["challenge_tp"],
        "comp_recall_preserved": c["guitarset/comp"]["recall"] >= b["guitarset/comp"]["recall"] - .02 - 1e-9,
    }
    for domain in ("all", "synthetic", "guitarset/comp"):
        before, after = b[domain]["latency_p95"], c[domain]["latency_p95"]
        checks[f"{domain}_latency_preserved"] = (before is not None and after is not None
                                                and after <= before + .016 + 1e-9)
    return {"accepted": all(checks.values()), "checks": checks}
