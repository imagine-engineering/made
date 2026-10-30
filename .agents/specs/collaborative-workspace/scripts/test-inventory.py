from pathlib import Path
import csv
import importlib.util
import json
import sys
import tempfile
import unittest


root = Path(__file__).resolve().parents[4]
specification = importlib.util.spec_from_file_location(
    "collaborative_inventory",
    root / ".agents/specs/collaborative-workspace/scripts/check-inventory.py",
)
module = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = module
specification.loader.exec_module(module)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checker = module.InventoryChecker(Path(self.directory.name), "all")
        self.path = self.checker.spec_root / "catalogs/final-coverage.csv"
        self.path.parent.mkdir(parents=True)
        self.checker.owners = {
            "CAP-001": module.CapabilityOwner("native", "Merge"),
        }
        self.checker.leaves = {
            "1.1": module.LeafTask("x", frozenset({"1.1"}), frozenset({"CAP-001"})),
            "2.1": module.LeafTask("~", frozenset({"1.2"}), frozenset({"CAP-001"})),
        }
        self.row = dict(
            capability_id="CAP-001",
            implementation_tasks="1.1;2.1",
            requirement_ids="1.1;1.2",
            source_rows="",
            source_row_count="0",
            component_status="partial",
            validation_evidence="Runtime evidence pending",
        )

    def check(self, rows=None):
        with self.path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(self.row))
            writer.writeheader()
            writer.writerows([self.row] if rows is None else rows)
        self.checker._check_final_coverage()
        return self.checker.errors

    def test_honest_partial(self):
        self.assertEqual(self.check(), [])

    def test_omitted_correction(self):
        self.row["implementation_tasks"] = "1.1"
        self.assertTrue(any("2.1" in error for error in self.check()))

    def test_false_completion(self):
        self.row["component_status"] = "complete"
        self.assertTrue(any("unfinished tasks" in error for error in self.check()))

    def test_requirement_gap(self):
        self.row["requirement_ids"] = "1.1"
        self.assertTrue(any("1.2" in error for error in self.check()))

    def test_invented_source(self):
        self.row["source_rows"] = "data:invented"
        self.assertTrue(any("invented" in error for error in self.check()))

    def test_duplicate_capability(self):
        self.assertTrue(
            any("duplicate capability" in error for error in self.check([self.row, self.row]))
        )

    def test_missing_evidence(self):
        self.row["validation_evidence"] = ""
        self.assertTrue(any("missing validation evidence" in error for error in self.check()))

    def test_checked_tasks_allow_complete(self):
        self.checker.leaves["2.1"] = module.LeafTask(
            "x", frozenset({"1.2"}), frozenset({"CAP-001"})
        )
        self.row["component_status"] = "complete"
        self.assertEqual(self.check(), [])


class AdrReferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checker = module.InventoryChecker(Path(self.directory.name), "all")
        self.path = self.checker.spec_root / "decisions/adr-test.md"
        self.path.parent.mkdir(parents=True)
        self.checker.leaves = {
            task: module.LeafTask("~", frozenset(), frozenset())
            for task in ["1.1", "1.2", "1.3", "2.1"]
        }

    def check(self, trace):
        self.path.write_text("# ADR\n\n## Implementation and validation trace\n\n" + trace)
        self.checker._check_adr_task_references()
        return self.checker.errors

    def test_valid_partial_tasks_are_allowed(self):
        self.assertEqual(self.check("Tasks 1.1–1.3 and 2.1 own implementation."), [])

    def test_unknown_task_reports_source_and_reference(self):
        errors = self.check("Task 47.7 owns retirement.")
        self.assertTrue(any("adr-test.md:5" in error and "47.7" in error for error in errors))

    def test_range_checks_interior_leaves(self):
        del self.checker.leaves["1.2"]
        self.assertTrue(any("1.2" in error for error in self.check("Tasks 1.1–1.3 own implementation.")))

    def test_invalid_ranges_are_rejected(self):
        for reference in ["1.1–1.8", "1.3–1.1", "1.1–2.1"]:
            with self.subTest(reference=reference):
                self.checker.errors.clear()
                self.assertTrue(self.check(f"Tasks {reference} own implementation."))

    def test_missing_trace_is_rejected(self):
        self.path.write_text("# ADR\nNo implementation trace.\n")
        self.checker._check_adr_task_references()
        self.assertTrue(any("missing ADR" in error for error in self.checker.errors))


class RemediationChecklistTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checker = module.InventoryChecker(Path(self.directory.name), "all")
        self.checker.leaves = {"1.1": module.LeafTask("~", frozenset({"1.1"}), frozenset())}
        self.path = self.checker.spec_root / "remediation-checklist.json"
        self.path.parent.mkdir(parents=True)
        (self.checker.spec_root / "requirements.md").write_text("1. **1.1** Criterion\n")
        self.row = dict(task_id="1.1", spec_state="~", acceptance_criteria={"1.1": "Criterion"},
                        affected_files="owner.rs", required_validation="owner test", findings=[dict(
                            id="finding", historical_finding="Gap", source="audit", current_status="open")])

    def check(self, rows=None):
        self.path.write_text(json.dumps({"tasks": [self.row] if rows is None else rows}))
        self.checker._check_remediation_checklist()
        return self.checker.errors

    def test_open_finding_preserves_pending_work(self):
        self.assertEqual(self.check(), [])

    def test_affected_file_path_list_is_accepted(self):
        self.row["affected_files"] = ["crates/editor/src/input.rs", "crates/project/src/project.rs"]
        self.assertEqual(self.check(), [])

    def test_malformed_affected_file_mapping_is_rejected(self):
        for mapping in [list("owner.rs"), ["owner.rs", " "], ["owner.rs", 42], 42, {"owner.rs": True}]:
            with self.subTest(mapping=mapping):
                self.checker.errors.clear()
                self.row["affected_files"] = mapping
                self.assertTrue(any("invalid affected files" in error for error in self.check()))

    def test_omitted_canonical_task(self):
        self.assertTrue(any("missing canonical task 1.1" in error for error in self.check([])))

    def test_duplicate_task(self):
        self.assertTrue(any("duplicate task" in error for error in self.check([self.row, self.row])))

    def test_missing_criterion(self):
        self.row["acceptance_criteria"] = {}
        self.assertTrue(any("acceptance criteria" in error for error in self.check()))

    def test_stale_specification_state(self):
        self.row["spec_state"] = "x"
        self.assertTrue(any("stale specification state" in error for error in self.check()))

    def test_changed_criterion_text(self):
        self.row["acceptance_criteria"]["1.1"] = "Weaker criterion"
        self.assertTrue(any("acceptance criteria" in error for error in self.check()))

    def test_resolution_without_evidence(self):
        self.row["findings"][0]["current_status"] = "resolved"
        self.assertTrue(any("claims resolution without" in error for error in self.check()))

    def test_blocker_without_dependency(self):
        self.row["findings"][0]["current_status"] = "externally_blocked"
        self.assertTrue(any("specific external dependency" in error for error in self.check()))

    def test_duplicate_finding(self):
        self.row["findings"] *= 2
        self.assertTrue(any("duplicate finding ID" in error for error in self.check()))


if __name__ == "__main__":
    unittest.main()
