"""Budget accounting and process-lock regressions; run on nano4 with unittest."""

import contextlib
import hashlib
import io
import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path

from exp.research_study import (
    STAGE_GPU_HOURS,
    attach_run,
    initialize_study,
    main,
    reconcile_run,
    release_run,
    reserve_run,
    show_study,
)


def _reserve_worker(study, config, run_id, seconds, start, results):
    start.wait(timeout=20)
    try:
        reserve_run(Path(study), run_id, "infrastructure", 1, seconds, Path(config), 42)
        results.put((run_id, "accepted"))
    except ValueError:
        results.put((run_id, "rejected"))
    except Exception as error:  # noqa: BLE001 -- report child-process failures to the parent test
        results.put((run_id, repr(error)))


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.study = self.root / "study.json"
        self.protocol = self.root / "protocol.json"
        self.protocol.write_text(json.dumps({
            "budget": {"total_gpu_hours": 64, "stages": STAGE_GPU_HOURS},
            "training": {"steps": 3250},
        }))
        self.config = self.root / "config.json"
        self.config.write_text('{"method": "muon", "lr": 0.025}\n')
        initialize_study(self.study, self.protocol)

    def reserve(self, run_id="run", stage="infrastructure", gpus=2, seconds=3600):
        return reserve_run(self.study, run_id, stage, gpus, seconds, self.config, 42)

    def test_protocol_caps_must_match_authorized_plan(self):
        protocol = json.loads(self.protocol.read_text())
        protocol["budget"]["stages"]["confirmation"] = 19
        self.protocol.write_text(json.dumps(protocol))
        with self.assertRaisesRegex(ValueError, "confirmation"):
            initialize_study(self.root / "bad.json", self.protocol)
        self.assertFalse((self.root / "bad.json").exists())
        with self.assertRaisesRegex(ValueError, "already exists"):
            initialize_study(self.study, self.protocol)

    def test_stage_and_total_caps_charge_full_requested_allocation(self):
        first = self.reserve(gpus=2, seconds=7200)
        self.assertEqual(first["usage"]["reserved_gpu_hours"], 4)
        before = self.study.read_bytes()
        with self.assertRaisesRegex(ValueError, "infrastructure stage"):
            self.reserve("extra", gpus=1, seconds=1)
        self.assertEqual(self.study.read_bytes(), before)
        for stage, cap in STAGE_GPU_HOURS.items():
            if stage != "infrastructure":
                self.reserve(stage, stage, 1, cap * 3600)
        self.assertEqual(show_study(self.study)["usage"]["remaining_gpu_hours"], 0)
        with self.assertRaisesRegex(ValueError, "total GPU-hour"):
            self.reserve("over-total", stage="recovery", gpus=1, seconds=1)

    def test_duplicate_run_id_cannot_replace_or_reuse_reservation(self):
        self.reserve()
        before = self.study.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.reserve()
        self.assertEqual(self.study.read_bytes(), before)
        release_run(self.study, "run")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.reserve()

    def test_snapshots_pin_exact_bytes_and_original_files_can_change(self):
        original = self.config.read_bytes()
        result = self.reserve()
        record = result["record"]
        self.assertEqual(record["config_sha256"], hashlib.sha256(original).hexdigest())
        snapshot = Path(record["config_snapshot_path"])
        self.assertEqual(snapshot.read_bytes(), original)
        self.assertEqual(snapshot.stat().st_mode & 0o222, 0)
        self.config.write_text('{"method": "adamw"}')
        self.protocol.write_text('{}')
        self.assertEqual(show_study(self.study)["ledger"]["runs"]["run"]["config_sha256"],
                         record["config_sha256"])
        snapshot.chmod(0o644)
        snapshot.write_text('{}')
        with self.assertRaisesRegex(ValueError, "integrity failure"):
            show_study(self.study)

    def test_completed_allocation_replaces_reservation_with_actual_cost(self):
        self.reserve()
        attach_run(self.study, "run", "123")
        result = reconcile_run(self.study, "run", "COMPLETED", 1800)
        self.assertEqual(result["usage"]["charged_gpu_hours"], 1)
        self.assertEqual(result["usage"]["reserved_gpu_hours"], 0)
        self.assertEqual(result["record"]["actual_gpu_seconds"], 3600)
        self.assertFalse(result["record"]["wall_limit_exceeded"])

    def test_failed_allocations_are_charged_and_retry_is_idempotent(self):
        self.reserve()
        attach_run(self.study, "run", "123")
        first = reconcile_run(self.study, "run", "OUT_OF_MEMORY", 900)
        before = self.study.read_bytes()
        second = reconcile_run(self.study, "run", "OUT_OF_MEMORY", 900)
        self.assertEqual(first, second)
        self.assertEqual(second["usage"]["charged_gpu_hours"], 0.5)
        self.assertEqual(self.study.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "Conflicting reconciliation"):
            reconcile_run(self.study, "run", "FAILED", 900)
        with self.assertRaisesRegex(ValueError, "Conflicting reconciliation"):
            reconcile_run(self.study, "run", "OUT_OF_MEMORY", 1800)

    def test_only_terminal_states_and_attached_jobs_can_be_reconciled(self):
        self.reserve()
        with self.assertRaisesRegex(ValueError, "attached Slurm job"):
            reconcile_run(self.study, "run", "FAILED", 1)
        attach_run(self.study, "run", "123")
        for state in ("RUNNING", "PENDING", "COMPLETING", "COMPLETED+", "CANCELLED by 1000"):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "terminal"):
                reconcile_run(self.study, "run", state, 10)
        for elapsed in (-1, float("nan"), float("inf"), True):
            with self.subTest(elapsed=elapsed), self.assertRaisesRegex(ValueError, "finite"):
                reconcile_run(self.study, "run", "FAILED", elapsed)

    def test_overruns_persist_full_cost_and_block_reservations_when_cap_exceeded(self):
        self.reserve(gpus=2, seconds=7200)
        attach_run(self.study, "run", "123")
        result = reconcile_run(self.study, "run", "TIMEOUT", 7300)
        self.assertEqual(result["record"]["actual_gpu_seconds"], 14600)
        self.assertTrue(result["record"]["wall_limit_exceeded"])
        self.assertLess(result["usage"]["stages"]["infrastructure"]["remaining_gpu_hours"], 0)
        self.assertEqual(result["usage"]["violations"], ["infrastructure stage cap exceeded"])
        with self.assertRaisesRegex(ValueError, "blocked"):
            self.reserve("other", "screening")
        self.assertEqual(show_study(self.study)["ledger"]["runs"]["run"]["elapsed_seconds"], 7300)

    def test_cli_overrun_is_nonzero_after_persisting_usage(self):
        self.reserve(gpus=1, seconds=60)
        attach_run(self.study, "run", "123")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["reconcile", "--study", str(self.study), "--run-id", "run",
                         "--state", "TIMEOUT", "--elapsed-seconds", "65"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["record"]["actual_gpu_seconds"], 65)

    def test_actual_gpu_count_is_charged_and_mismatch_blocks_new_submissions(self):
        self.reserve(gpus=2, seconds=1800)
        attach_run(self.study, "run", "123")
        result = reconcile_run(self.study, "run", "FAILED", 900, allocated_gpus=4)
        self.assertEqual(result["record"]["gpus"], 2)
        self.assertEqual(result["record"]["allocated_gpus"], 4)
        self.assertEqual(result["record"]["actual_gpu_seconds"], 3600)
        self.assertTrue(result["record"]["allocation_mismatch"])
        self.assertEqual(result["usage"]["charged_gpu_hours"], 1)
        with self.assertRaisesRegex(ValueError, "blocked"):
            self.reserve("other", "screening")
        self.assertEqual(reconcile_run(self.study, "run", "FAILED", 900, allocated_gpus=4), result)
        with self.assertRaisesRegex(ValueError, "Conflicting reconciliation"):
            reconcile_run(self.study, "run", "FAILED", 900, allocated_gpus=2)

    def test_attached_job_cannot_release_or_charge_other_run(self):
        self.reserve()
        self.reserve("other")
        first = attach_run(self.study, "run", "123")
        self.assertEqual(attach_run(self.study, "run", "123"), first)
        with self.assertRaisesRegex(ValueError, "already attached"):
            attach_run(self.study, "other", "123")
        with self.assertRaisesRegex(ValueError, "unsubmitted"):
            release_run(self.study, "run")
        result = release_run(self.study, "other")
        self.assertEqual(result["usage"]["reserved_gpu_hours"], 2)
        self.assertEqual(release_run(self.study, "other"), result)

    def test_corrupt_usage_is_rejected_before_any_new_write(self):
        self.reserve()
        ledger = json.loads(self.study.read_text())
        ledger["runs"]["run"]["actual_gpu_seconds"] = 12
        self.study.write_text(json.dumps(ledger))
        before = self.study.read_bytes()
        with self.assertRaisesRegex(ValueError, "Unreconciled run"):
            attach_run(self.study, "run", "123")
        self.assertEqual(self.study.read_bytes(), before)

    def test_corrupt_reconciled_usage_is_rejected(self):
        self.reserve()
        attach_run(self.study, "run", "123")
        reconcile_run(self.study, "run", "FAILED", 12)
        ledger = json.loads(self.study.read_text())
        ledger["runs"]["run"]["actual_gpu_seconds"] = 12
        self.study.write_text(json.dumps(ledger))
        with self.assertRaisesRegex(ValueError, "Inconsistent actual usage"):
            show_study(self.study)

    def test_invalid_reservations_do_not_mutate_ledger(self):
        before = self.study.read_bytes()
        for gpus, seconds, seed in [(0, 1, 42), (1, 0, 42), (True, 1, 42),
                                    (1, 1.5, 42), (1, 1, -1)]:
            with self.subTest(gpus=gpus, seconds=seconds, seed=seed), self.assertRaises(ValueError):
                reserve_run(self.study, "bad", "screening", gpus, seconds, self.config, seed)
        self.assertEqual(self.study.read_bytes(), before)

    def test_concurrent_reservations_serialize_without_lost_updates_or_overspend(self):
        context = multiprocessing.get_context("spawn")
        start, results = context.Event(), context.Queue()
        workers = [context.Process(target=_reserve_worker,
                   args=(str(self.study), str(self.config), f"parallel-{i}", 10800, start, results))
                   for i in range(4)]
        try:
            for worker in workers:
                worker.start()
            start.set()
            rows = [results.get(timeout=30) for _ in workers]
            for worker in workers:
                worker.join(timeout=30)
                self.assertEqual(worker.exitcode, 0)
            self.assertEqual(sum(state == "accepted" for _, state in rows), 1, rows)
            self.assertEqual(sum(state == "rejected" for _, state in rows), 3, rows)
            result = show_study(self.study)
            self.assertEqual(len(result["ledger"]["runs"]), 1)
            self.assertEqual(result["usage"]["reserved_gpu_hours"], 3)
            self.assertEqual(result["ledger"]["revision"], 1)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=10)
            results.close()
            results.join_thread()


if __name__ == "__main__":
    unittest.main()
