"""Submission/accounting regressions; run on nano4, never contact real Slurm.

Run with ``PYTHONPATH=src python -m unittest exp.test_research_submit``.
All subprocess and ledger calls are mocked; temporary artifacts exercise the
recovery receipts and frozen configuration paths used by the real launcher.
"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from exp import research_submit


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.study = self.root / "study with spaces"
        self.source = self.root / "source with spaces"
        self.study.mkdir()
        self.launcher = self.source / "src/exp/research_run.sbatch"
        self.launcher.parent.mkdir(parents=True)
        self.launcher.write_text("#!/bin/bash\n")
        self.config = self.root / "configuration.json"
        self.config.write_text('{"method": "muon"}\n')
        self.protocol = self.study / "protocol.json"
        self.protocol.write_text('{"budget": {"total_gpu_hours": 64}}\n')
        snapshots = self.study / "budget.json.snapshots"
        snapshots.mkdir()
        self.frozen_protocol = snapshots / "frozen-protocol.json"
        self.frozen_protocol.write_bytes(self.protocol.read_bytes())
        self.frozen_config = snapshots / "frozen-config.json"
        self.frozen_config.write_bytes(self.config.read_bytes())
        self.run_id = "smoke-42"
        self.record = {
            "run_id": self.run_id,
            "gpus": 8,
            "protocol_snapshot_path": str(self.frozen_protocol),
            "config_snapshot_path": str(self.frozen_config),
        }

    def submit(self):
        return research_submit.submit(
            self.study, self.source, self.run_id, "infrastructure",
            self.config, 42, 7, mode="smoke",
        )

    def test_reservation_precedes_sbatch_and_receipt_precedes_attachment(self):
        events = []
        receipt_path = self.study / "logs" / f"{self.run_id}-submission.json"

        def ledger(study, action, *arguments):
            self.assertEqual(study, self.study)
            events.append(action)
            if action == "reserve":
                options = dict(zip(arguments[::2], arguments[1::2], strict=True))
                self.assertEqual(options["--gpus"], 8)
                self.assertEqual(options["--seconds"], 7 * 60)
                self.assertEqual(options["--stage"], "infrastructure")
                self.assertEqual(options["--seed"], 42)
                return {"record": self.record}
            self.assertEqual(action, "attach")
            self.assertTrue(receipt_path.is_file())
            self.assertEqual(json.loads(receipt_path.read_text())["job_id"], "12345")
            self.assertEqual(arguments, ("--run-id", self.run_id, "--job-id", "12345"))
            return {"record": self.record | {"job_id": "12345"}}

        def sbatch(command, **kwargs):
            self.assertEqual(events, ["reserve"])
            events.append("sbatch")
            self.assertEqual(command[0], "sbatch")
            for option in ("--parsable", "--no-requeue", "--nodes=1", "--ntasks=1",
                           "--gpus=8", "--gpus-per-node=8", "--time=7"):
                self.assertIn(option, command)
            self.assertEqual(command[-7:], [
                str(self.launcher), str(self.study), self.run_id,
                str(self.frozen_config), "42", "smoke", str(self.frozen_protocol),
            ])
            self.assertNotIn(str(self.protocol), command)
            self.assertNotIn(str(self.config), command)
            self.assertIn(f"--chdir={self.source}", command)
            self.assertTrue(kwargs.get("check"))
            self.assertTrue(kwargs.get("capture_output"))
            self.assertTrue(kwargs.get("text"))
            self.assertFalse(kwargs.get("shell", False))
            return subprocess.CompletedProcess(command, 0, "12345;cluster\n", "")

        with mock.patch.object(research_submit, "ledger", side_effect=ledger), \
                mock.patch.object(research_submit.subprocess, "run", side_effect=sbatch):
            receipt = self.submit()
        self.assertEqual(events, ["reserve", "sbatch", "attach"])
        self.assertEqual(receipt, json.loads(receipt_path.read_text()))

    def test_rejected_budget_does_not_contact_scheduler(self):
        with (
            mock.patch.object(research_submit, "ledger", side_effect=ValueError("cap exceeded")),
            mock.patch.object(research_submit.subprocess, "run") as run,
            self.assertRaisesRegex(ValueError, "cap exceeded"),
        ):
            self.submit()
        run.assert_not_called()

    def test_changed_protocol_releases_only_before_scheduler_submission(self):
        self.protocol.write_text('{"changed": true}\n')
        with (
            mock.patch.object(research_submit, "ledger", return_value={"record": self.record}) as ledger,
            mock.patch.object(research_submit.subprocess, "run") as run,
            self.assertRaisesRegex(ValueError, "protocol changed"),
        ):
            self.submit()
        run.assert_not_called()
        self.assertEqual([call.args[1] for call in ledger.call_args_list], ["reserve", "release"])

    def test_ambiguous_rpc_failure_retains_budget_and_records_diagnostics(self):
        # A failed client RPC may follow scheduler acceptance; even a numeric
        # stdout must not cause automatic release or a second submission.
        error = subprocess.CalledProcessError(
            1, ["sbatch"], output="67890\n",
            stderr="Socket timed out on send/recv operation",
        )
        with (
            mock.patch.object(research_submit, "ledger", return_value={"record": self.record}) as ledger,
            mock.patch.object(research_submit.subprocess, "run", side_effect=error) as run,
            self.assertRaisesRegex(RuntimeError, "reservation retained.*Socket timed out"),
        ):
            self.submit()
        run.assert_called_once()
        self.assertEqual([call.args[1] for call in ledger.call_args_list], ["reserve"])
        receipt_path = self.study / "logs" / f"{self.run_id}-submission-error.json"
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt["run_id"], self.run_id)
        self.assertEqual(receipt["returncode"], 1)
        self.assertEqual(receipt["stdout"], "67890\n")
        self.assertEqual(receipt["stderr"], error.stderr)
        self.assertIs(receipt["reservation_retained"], True)
        self.assertEqual(receipt["command"], run.call_args.args[0])

    def test_attachment_failure_preserves_job_id_receipt_and_reservation(self):
        def ledger(study, action, *arguments):
            if action == "reserve":
                return {"record": self.record}
            self.assertEqual(action, "attach")
            raise RuntimeError("ledger unavailable after scheduler acceptance")

        with (
            mock.patch.object(research_submit, "ledger", side_effect=ledger) as calls,
            mock.patch.object(research_submit.subprocess, "run", return_value=
                              subprocess.CompletedProcess(["sbatch"], 0, "12345\n", "")),
            self.assertRaisesRegex(RuntimeError, "ledger unavailable"),
        ):
            self.submit()
        self.assertEqual([call.args[1] for call in calls.call_args_list], ["reserve", "attach"])
        receipt = json.loads((self.study / "logs" / f"{self.run_id}-submission.json").read_text())
        self.assertEqual(receipt["job_id"], "12345")
        self.assertEqual(receipt["reservation"], self.record)

    def reconciliation(self, output, records=None):
        if records is None:
            records = {"run": {"run_id": "run", "job_id": "12345", "status": "submitted", "gpus": 8}}
        usage = {"charged_gpu_hours": 0.25}

        def ledger(study, action, *arguments):
            if action == "show":
                return {"ledger": {"runs": records}, "usage": usage}
            self.assertEqual(action, "reconcile")
            return {"record": {"status": "reconciled"}, "usage": usage}

        ledger_mock = mock.patch.object(research_submit, "ledger", side_effect=ledger)
        subprocess_mock = mock.patch.object(
            research_submit.subprocess, "run",
            return_value=subprocess.CompletedProcess(["sacct"], 0, output, ""),
        )
        return ledger_mock, subprocess_mock

    def test_reconcile_charges_actual_gpu_count_without_double_counting_typed_tres(self):
        output = (
            "12345.batch|COMPLETED|225|cpu=32,gres/gpu=8|\n"
            "99999|COMPLETED|1000|gres/gpu=8|\n"
            "12345|COMPLETED|225|cpu=32,gres/gpu=4,gres/gpu:h200=4|\n"
        )
        ledger_patch, subprocess_patch = self.reconciliation(output)
        with ledger_patch as ledger, subprocess_patch as run:
            result = research_submit.reconcile(self.study)
        self.assertEqual(len(result["updates"]), 1)
        self.assertIn(mock.call(self.study, "reconcile", "--run-id", "run", "--state", "COMPLETED",
                               "--elapsed-seconds", 225, "--allocated-gpus", 4), ledger.call_args_list)
        command = run.call_args.args[0]
        self.assertEqual(command[0], "sacct")
        self.assertIn("-X", command)
        self.assertIn("--format=JobIDRaw,State%40,ElapsedRaw,AllocTRES", command)

    def test_reconcile_normalizes_cancelled_state_and_charges_failed_allocation(self):
        ledger_patch, subprocess_patch = self.reconciliation(
            "12345|CANCELLED by 1000|90|gres/gpu=8|\n"
        )
        with ledger_patch as ledger, subprocess_patch:
            research_submit.reconcile(self.study)
        self.assertIn(mock.call(self.study, "reconcile", "--run-id", "run", "--state", "CANCELLED",
                               "--elapsed-seconds", 90, "--allocated-gpus", 8), ledger.call_args_list)

    def test_active_and_missing_accounting_records_keep_reservation(self):
        for output in ("", "12345|RUNNING|60|gres/gpu=8|\n", "12345|COMPLETING|60|gres/gpu=8|\n"):
            with self.subTest(output=output):
                ledger_patch, subprocess_patch = self.reconciliation(output)
                with ledger_patch as ledger, subprocess_patch:
                    result = research_submit.reconcile(self.study)
                self.assertEqual(result["updates"], [])
                self.assertTrue(all(call.args[1] == "show" for call in ledger.call_args_list))

    def test_missing_gpu_allocation_cannot_silently_use_requested_count(self):
        ledger_patch, subprocess_patch = self.reconciliation("12345|FAILED|90|cpu=32|\n")
        with ledger_patch as ledger, subprocess_patch, self.assertRaisesRegex(ValueError, "actual GPU allocation"):
            research_submit.reconcile(self.study)
        self.assertTrue(all(call.args[1] == "show" for call in ledger.call_args_list))

    def test_unsubmitted_and_already_settled_runs_do_not_query_scheduler(self):
        records = {
            "new": {"run_id": "new", "job_id": None, "status": "reserved", "gpus": 8},
            "done": {"run_id": "done", "job_id": "12345", "status": "reconciled", "gpus": 8},
            "released": {"run_id": "released", "job_id": None, "status": "released", "gpus": 8},
        }
        ledger_patch, subprocess_patch = self.reconciliation("", records)
        with ledger_patch, subprocess_patch as run:
            result = research_submit.reconcile(self.study)
        run.assert_not_called()
        self.assertEqual(result["updates"], [])


if __name__ == "__main__":
    unittest.main()
