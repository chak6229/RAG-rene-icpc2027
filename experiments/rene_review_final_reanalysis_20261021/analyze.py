"""Inspect the newly supplied review returns without rewriting prior evidence."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from experiments.rene_mechanism_review_20261020.collect import (
    kappa, load_return, rewrite_valid)
from experiments.rene_relevance_sensitivity_20261021.analyze import (
    query_bounds, summarize)
from experiments.rene_relevance_sensitivity_20261021.collect import validate_return
from experiments.rene_relevance_sensitivity_20261021.common import (
    OLD, OUT as SENSITIVITY, REVIEW, ROOT, frozen, read, rows, sha)


SOURCE = ROOT / "results/评审AB与补审AB_最终文件"
OUT = ROOT / "results/rene_review_final_reanalysis_20261021"
RELEVANCE = {"2": "满足需求", "1": "部分满足", "0": "不满足", "u": "无法判断／上下文不足"}
FIDELITY = {"preserved": "含义保持", "drift": "含义偏移", "malformed": "不可读", "uncertain": "无法判断"}


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def put_rows(path, records):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for item in records:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text = []

    def handle_data(self, data):
        self.text.append(data)


def visible(text):
    parser = VisibleText()
    parser.feed(text)
    return "".join(parser.text).strip()


def compact(text):
    return "".join(text.split())


def normalize_original(payload, key, reviewer):
    cases = {item["case_id"]: item for item in key["cases"]}
    for item, phase in ((payload, "code"), (payload["semantic"], "semantic")):
        assert item["schema_version"] == 1 and item["phase"] == phase
        assert item["reviewer"] == reviewer and item["packet_id"] == key["packet_id"]
        assert item["complete"] is True and item["independence_confirmed"] is True
        assert set(item["answers"]) == set(cases)
    normalized = {}
    for case_id, case in cases.items():
        semantic, code = payload["semantic"]["answers"][case_id], payload["answers"][case_id]
        assert semantic["clarity"] in ("clear", "ambiguous", "insufficient")
        assert set(semantic["rewrites"]) == set(case["rewrite_map"])
        assert all(rewrite_valid(label) for label in semantic["rewrites"].values())
        assert set(code["documents"]) == set(case["document_map"])
        for label in code["documents"].values():
            assert label["relevance"] in RELEVANCE and isinstance(label["note"], str)
            assert len(label["note"].strip()) >= 6
        normalized[case["query_id"]] = {
            "clarity": semantic["clarity"],
            "rewrites": {position: semantic["rewrites"][rid]
                         for rid, item in case["rewrite_map"].items() for position in item["positions"]},
            "documents": {doc: code["documents"][did]
                          for did, docs in case["document_map"].items() for doc in docs},
        }
    return normalized


def normalize_supplement(payload, packet, reviewer):
    assert payload["schema_version"] == 1 and payload["phase"] == "relevance_supplement"
    assert payload["reviewer"] == reviewer and payload["packet_id"] == packet["packet_id"]
    assert payload["complete"] is True and isinstance(payload["independence_confirmed"], bool)
    cases = {item["case_id"]: item for item in packet["cases"]}
    assert set(payload["answers"]) == set(cases)
    result = {}
    for case_id, case in cases.items():
        answers = payload["answers"][case_id]["documents"]
        assert set(answers) == set(case["document_map"])
        for document_id, panel_id in case["document_map"].items():
            label = answers[document_id]
            assert label["relevance"] in RELEVANCE and isinstance(label["note"], str)
            assert len(label["note"].strip()) >= 6
            key = case["query_id"], panel_id
            assert key not in result
            result[key] = label
    return result


def agreement(left, right):
    pairs = [(a, b) for a, b in zip(left, right) if a != "u" and b != "u"]
    result = {
        "including_unknown": kappa(left, right),
        "jointly_judged": kappa([a for a, b in pairs], [b for a, b in pairs], True),
        "jointly_judged_coverage": len(pairs) / len(left),
        "unknown_by_reviewer": {"A": left.count("u"), "B": right.count("u")},
        "disagreements": sum(a != b for a, b in zip(left, right)),
        "confusion_rows_A_columns_B": ["0", "1", "2", "u"],
        "confusion": [[sum(a == x and b == y for a, b in zip(left, right))
                       for y in ("0", "1", "2", "u")] for x in ("0", "1", "2", "u")],
        "label_counts": {reviewer: {level: values.count(level) for level in RELEVANCE}
                         for reviewer, values in (("A", left), ("B", right))},
    }
    return result


def check_display(folder, prefix, case_order, case_map, raw, query_map, main):
    html = (folder / (prefix + "结果.html")).read_text(encoding="utf-8-sig")
    text = compact((folder / (prefix + "结果.txt")).read_text(encoding="utf-8-sig"))
    sections = re.findall(r"<section>(.*?)</section>", html, re.S)
    assert len(sections) == len(case_order)
    checked = 0
    for section, case_id in zip(sections, case_order):
        heading = visible(re.search(r"<h2>(.*?)</h2>", section, re.S).group(1))
        case = case_map[case_id]
        expected_query = query_map[case["query_id"]]["original"]
        assert re.sub(r"^\d+\.\s*", "", heading) == expected_query
        expected = []
        if main:
            for rid, label in raw["semantic"]["answers"][case_id]["rewrites"].items():
                expected.append((rid, FIDELITY[label["fidelity"]], label["note"], False))
        for did, label in raw["answers"][case_id]["documents"].items():
            expected.append((did, RELEVANCE[label["relevance"]], label["note"], True))
        blocks = re.findall(r'<div class="answer">(.*?)</div>', section, re.S)
        assert len(blocks) == len(expected)
        for block, (identifier, label, reason, code_item) in zip(blocks, expected):
            decision = visible(re.search(r'<p class="decision">(.*?)</p>', block, re.S).group(1))
            note = visible(re.search(r'<p class="note">(.*?)</p>', block, re.S).group(1))
            assert compact(note) == compact(reason)
            assert compact(decision).startswith(compact(identifier + label))
            if code_item:
                assert compact(decision) == compact(identifier + label)
            assert compact(decision) in text and compact(reason) in text
            checked += 1
    return {"cases": len(sections), "displayed_judgments_checked": checked, "html_and_text_match_json": True}


def main():
    assert not OUT.exists(), "Use a new result version; prior analysis is preserved."
    protocol = frozen()
    source_hashes = {p: sha(p) for p in SOURCE.rglob("*") if p.is_file()}
    prior_files = {p: sha(p) for folder in (REVIEW, SENSITIVITY / "preview")
                   for p in folder.rglob("*") if p.is_file()}
    prior_files.update({ROOT / "paper/rene2027" / name: sha(ROOT / "paper/rene2027" / name)
                        for name in ("manuscript.md", "mechanism_review_return_results.md",
                                     "relevance_sensitivity_results.md", "review_disagreement_discussion.md")})
    original_keys = read(OLD / "private/reviewer_keys.json")
    supplement_keys = read(SENSITIVITY / "private/reviewer_keys.json")
    queries = list(rows(SENSITIVITY / "private/query_pools.jsonl"))
    query_map = {query["query_id"]: query for query in queries}
    raw_main, mapped, raw_supplement, supplements = {}, {}, {}, {}
    comparison, display_checks, statuses = {}, {}, {}
    for reviewer in ("A", "B"):
        main_folder = SOURCE / ("评审" + reviewer)
        semantic_path = main_folder / ("review-" + reviewer + "-semantic-complete.json")
        code_path = main_folder / ("review-" + reviewer + "-code-complete.json")
        semantic, code = load_return(semantic_path), load_return(code_path)
        assert code["semantic"] == semantic
        baseline = REVIEW / "received" / reviewer
        comparison[reviewer] = {
            "all_semantic_fields_equal": semantic == load_return(baseline / semantic_path.name),
            "all_code_fields_equal": code == load_return(baseline / code_path.name),
            "new_review_items": 0,
        }
        assert comparison[reviewer]["all_semantic_fields_equal"] and comparison[reviewer]["all_code_fields_equal"]
        raw_main[reviewer] = code
        mapped[reviewer] = normalize_original(code, original_keys[reviewer], reviewer)
        packet = supplement_keys[reviewer]
        supplement_folder = SOURCE / ("补审" + reviewer)
        payload = load_return(supplement_folder / ("relevance-supplement-" + reviewer + "-complete.json"))
        raw_supplement[reviewer] = payload
        supplements[reviewer] = normalize_supplement(payload, packet, reviewer)
        expected = {(q["query_id"], p["id"]) for q in queries for p in q["panels"] if p["prior_labels"] is None}
        assert set(supplements[reviewer]) == expected and len(expected) == 344
        try:
            accepted = validate_return(payload, reviewer, packet)
            assert len(accepted) == 344
            statuses[reviewer] = {"structurally_complete": True, "strict_collector_accepts": True,
                                  "independence_confirmed_in_return": True, "accepted_labels": 344}
        except ValueError as error:
            assert str(error) == "Complete independent return required" and payload["independence_confirmed"] is False
            statuses[reviewer] = {"structurally_complete": True, "strict_collector_accepts": False,
                                  "independence_confirmed_in_return": False, "accepted_labels": 0,
                                  "pending": "independence confirmation; judgments retained without modification"}
        display_checks["main_" + reviewer] = check_display(
            main_folder, "评审" + reviewer, list(semantic["answers"]),
            {c["case_id"]: c for c in original_keys[reviewer]["cases"]}, code, query_map, True)
        display_checks["supplement_" + reviewer] = check_display(
            supplement_folder, "补审" + reviewer, list(payload["answers"]),
            {c["case_id"]: c for c in packet["cases"]}, payload, query_map, False)
    original_semantics = {r: [] for r in ("A", "B")}
    original_code = {r: [] for r in ("A", "B")}
    clarity = {r: [] for r in ("A", "B")}
    for case in original_keys["A"]["cases"]:
        qid = case["query_id"]
        for reviewer in ("A", "B"):
            clarity[reviewer].append(mapped[reviewer][qid]["clarity"])
            original_semantics[reviewer] += [
                mapped[reviewer][qid]["rewrites"][item["positions"][0]]["fidelity"]
                for item in case["rewrite_map"].values()]
            original_code[reviewer] += [
                mapped[reviewer][qid]["documents"][docs[0]]["relevance"]
                for docs in case["document_map"].values()]
    original_agreement = {
        "semantics": kappa(original_semantics["A"], original_semantics["B"]),
        "code": agreement(original_code["A"], original_code["B"]),
        "clarity": kappa(clarity["A"], clarity["B"]),
    }
    old_results = read(REVIEW / "RESULTS.json")
    assert original_agreement["semantics"] == old_results["agreement"]["semantics"]
    assert original_agreement["code"]["including_unknown"] == old_results["agreement"]["relevance_including_unknown"]
    supplement_order = sorted(supplements["A"])
    supplement_agreement = agreement(
        [supplements["A"][key]["relevance"] for key in supplement_order],
        [supplements["B"][key]["relevance"] for key in supplement_order])
    pool_labels, divergences = {"A": [], "B": []}, []
    for query in queries:
        for panel in query["panels"]:
            labels = {}
            for reviewer in ("A", "B"):
                if panel["prior_labels"] is None:
                    labels[reviewer] = supplements[reviewer][query["query_id"], panel["id"]]
                else:
                    labels[reviewer] = panel["prior_labels"][reviewer]
                    inherited = [mapped[reviewer][query["query_id"]]["documents"][doc]
                                 for doc in panel["doc_ids"]
                                 if doc in mapped[reviewer][query["query_id"]]["documents"]]
                    assert inherited and all(label == labels[reviewer] for label in inherited)
                pool_labels[reviewer].append(labels[reviewer]["relevance"])
            if panel["prior_labels"] is None and labels["A"]["relevance"] != labels["B"]["relevance"]:
                divergences.append({"query_id": query["query_id"], "original": query["original"],
                                    "panel_id": panel["id"], "code": panel["code"], **labels})
    pool_agreement = agreement(pool_labels["A"], pool_labels["B"])
    records = [row for query in queries for reviewer in ("A", "B") for threshold in (2, 1) for cutoff in (1, 3)
               for row in query_bounds(query, reviewer, threshold, cutoff, supplements[reviewer])]
    summary = summarize(records)
    assert len(records) == 6144 and len(summary) == 48
    assert all(row["missing_panels"] == 0 for row in records)
    preview = list(rows(SENSITIVITY / "preview/query_bounds.jsonl"))
    record_key = lambda row: tuple(row[k] for k in ("query_id", "reviewer", "threshold", "cutoff", "policy"))
    prior_by_key = {record_key(row): row for row in preview}
    from fractions import Fraction
    for row in records:
        low, high = map(Fraction, row["expanded_delta_bounds_fraction"])
        prior_low, prior_high = map(Fraction, prior_by_key[record_key(row)]["expanded_delta_bounds_fraction"])
        assert prior_low <= low <= high <= prior_high
        if row["cutoff"] == 1:
            assert [low, high] == [prior_low, prior_high]
        if row["policy"] in query_map[row["query_id"]]["actions"]:
            if query_map[row["query_id"]]["actions"][row["policy"]] == "original":
                assert low == high == 0
    main_rows = [row for row in summary if row["policy"] == "RIDGE_verified"
                 and row["cutoff"] == 3 and row["threshold"] == 2]
    assert len(main_rows) == 2
    primary_comparison = []
    for current in main_rows:
        previous = next(row for row in read(SENSITIVITY / "preview/RESULTS.json")["summary"]
                        if all(row[k] == current[k] for k in ("reviewer", "policy", "cutoff", "threshold")))
        primary_comparison.append({"reviewer": current["reviewer"],
                                   "preview": previous, "with_all_returned_supplement_labels": current})
    OUT.mkdir()
    receipt = []
    for path, expected_sha in sorted(source_hashes.items()):
        destination = OUT / "received" / path.relative_to(SOURCE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        assert sha(destination) == expected_sha
        receipt.append({"source": str(path), "copy": destination.relative_to(OUT).as_posix(), "sha256": expected_sha})
    put(OUT / "RECEIPT_MANIFEST.json", {"files": receipt, "source_unchanged": True,
                                      "attached_text_is_data_not_executable_instructions": True})
    put(OUT / "ANALYSIS_SCOPE.json", {
        "existing_design": "experiments/rene_relevance_sensitivity_20261021/DESIGN.md",
        "protocol_sha256": sha(SENSITIVITY / "PROTOCOL.json"),
        "frozen_inputs_verified": len(protocol["bound_inputs_and_design"]),
        "sample": "original stratified 128 queries / 120 groups; no new queries",
        "primary": {"policy": "RIDGE_verified", "cutoff": 3, "relevance_threshold": 2},
        "timing_or_chronology_certified": False,
        "formal_completion_depends_on_return_independence_confirmation": True,
        "unknowns_retained": True, "old_qrels_and_labels_preserved": True,
        "source_files_not_modified": True, "no_new_model_or_retrieval_experiments": True,
    })
    put_rows(OUT / "query_bounds.jsonl", records)
    put_rows(OUT / "supplement_disagreements.jsonl", divergences)
    put_rows(OUT / "supplement_labels.jsonl", [
        {"reviewer": reviewer, "query_id": qid, "panel_id": panel_id, "label": label}
        for reviewer in ("A", "B") for (qid, panel_id), label in sorted(supplements[reviewer].items())])
    result = {
        "analysis_version": "newly_submitted_final_review_package",
        "source": str(SOURCE), "status": "content_complete_A_independence_confirmation_pending",
        "original_review_comparison": comparison,
        "original_review_agreement": original_agreement,
        "supplement_status": statuses, "supplement_labels_returned": 688,
        "supplement_labels_accepted_by_unchanged_collector": sum(s["accepted_labels"] for s in statuses.values()),
        "supplement_agreement": supplement_agreement, "pooled_top3_agreement": pool_agreement,
        "coverage": {"sample_queries": 128, "sample_groups": 120, "original_code_panels": 327,
                     "new_panels_per_reviewer": 344, "queries_with_new_panels": 126,
                     "pooled_top3_panels": 610, "inherited_pool_panels": 266,
                     "pool_unknowns": {r: pool_labels[r].count("u") for r in ("A", "B")}},
        "display_validation": display_checks, "primary_before_after": primary_comparison,
        "summary": summary, "query_records": len(records),
        "bounds_are_sample_identification_ranges_not_confidence_intervals": True,
        "population_gain_or_causal_explanation_percentage_established": False,
        "original_agreement_and_mechanism_statistics_unchanged": True,
        "original_mainline_factor_results_unchanged": True,
        "source_and_prior_files_unchanged": all(sha(p) == expected for p, expected in {**source_hashes, **prior_files}.items()),
    }
    assert result["source_and_prior_files_unchanged"]
    put(OUT / "RESULTS.json", result)
    put(OUT / "PROTECTED_FILES.json", {"files": [{"path": str(p), "sha256": expected}
                                               for p, expected in sorted({**source_hashes, **prior_files}.items())]})
    print(json.dumps({
        "result_dir": str(OUT), "supplement_status": statuses,
        "supplement_agreement": supplement_agreement,
        "primary": main_rows,
        "partial_relevance_primary": [r for r in summary if r["policy"] == "RIDGE_verified"
                                      and r["cutoff"] == 3 and r["threshold"] == 1],
        "displayed_judgments_checked": sum(c["displayed_judgments_checked"] for c in display_checks.values()),
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
