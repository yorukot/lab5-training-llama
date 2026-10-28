"""Submit one budget-reserved Slurm run from an isolated research snapshot.

Run on nano4, not on the workstation. The reservation is made BEFORE sbatch;
ambiguous submission errors retain the reservation. Scheduler terminal
status and allocation seconds, not rank-zero result.json, settle the budget.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def ledger(study, action, *arguments):
    result = subprocess.run(
        [sys.executable, "-m", "exp.research_study", action,
         "--study", str(study / "budget.json"), *map(str, arguments)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def submit(study, source, run_id, stage, config, seed, minutes, mode="train"):
    if type(minutes) is not int or not 1 <= minutes <= 120:
        raise ValueError("Wall limit must be 1..120 whole minutes")
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("Run ID must contain only letters, digits, hyphens, underscores")
    study, source, config = study.resolve(), source.resolve(), config.resolve()
    launcher = source / "src/exp/research_run.sbatch"
    if not launcher.is_file() or not config.is_file():
        raise ValueError("Missing source snapshot or experiment configuration")
    if (study / "runs" / run_id).exists():
        raise ValueError("Use a fresh run ID; existing runs are never overwritten")
    reservation = ledger(study, "reserve", "--run-id", run_id, "--stage", stage,
                         "--gpus", 8, "--seconds", minutes * 60,
                         "--config", config, "--seed", seed)
    record = reservation["record"]
    (study / "logs").mkdir(exist_ok=True)
    # Match the protocol actually executed to the immutable ledger snapshot.
    if Path(record["protocol_snapshot_path"]).read_bytes() != (study / "protocol.json").read_bytes():
        ledger(study, "release", "--run-id", run_id)
        raise ValueError("Study protocol changed after budget initialization")
    command = ["sbatch", "--parsable", "--no-requeue", "--nodes=1", "--ntasks=1", "--gpus=8", "--gpus-per-node=8", f"--time={minutes}",
               f"--job-name=opt-{run_id}", f"--chdir={source}",
               f"--output={study / 'logs' / (run_id + '-%j.log')}",
               str(launcher), str(study), run_id,
               record["config_snapshot_path"], str(seed), mode, record["protocol_snapshot_path"]]
    try:
        job_id = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]
    except subprocess.CalledProcessError as error:
        failure = {"run_id": run_id, "command": command,
                   "returncode": error.returncode, "stdout": error.stdout,
                   "stderr": error.stderr, "reservation_retained": True}
        (study / "logs" / f"{run_id}-submission-error.json").write_text(json.dumps(failure, indent=2) + "\n")
        # A controller RPC timeout may occur after accepting the job. Keep the
        # reservation until scheduler accounting proves no allocation exists.
        raise RuntimeError(f"sbatch failed; reservation retained. {error.stderr}") from error
    # A failure after sbatch deliberately retains the reservation: the job may
    # already be live. Recover its ID from this durable receipt and attach it.
    receipt = {"job_id": job_id, "run_id": run_id, "source": str(source),
               "command": command, "reservation": record}
    (study / "logs" / f"{run_id}-submission.json").write_text(json.dumps(receipt, indent=2) + "\n")
    ledger(study, "attach", "--run-id", run_id, "--job-id", job_id)
    return receipt


def reconcile(study):
    snapshot = ledger(study, "show")
    records = snapshot["ledger"]["runs"]
    if isinstance(records, list):
        records = {record["run_id"]: record for record in records}
    updates = []
    for run_id, record in records.items():
        job_id = record.get("job_id")
        if not job_id or record.get("status") in ("reconciled", "released"):
            continue
        query = subprocess.run(
            ["sacct", "-n", "-X", "-P", "-j", str(job_id),
             "--format=JobIDRaw,State%40,ElapsedRaw,AllocTRES"],
            check=True, capture_output=True, text=True,
        )
        for line in query.stdout.splitlines():
            fields = line.split("|")
            if len(fields) < 4 or fields[0] != str(job_id):
                continue
            state = fields[1].split()[0].rstrip("+")
            if state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING"}:
                continue
            resources = dict(item.split("=", 1) for item in fields[3].split(",") if "=" in item)
            allocated_gpus = resources.get("gres/gpu")
            if allocated_gpus is None:
                if int(fields[2]) == 0 and state in {"CANCELLED", "FAILED"}:
                    allocated_gpus = str(record["gpus"])
                else:
                    raise ValueError(f"Cannot establish actual GPU allocation for {job_id}")
            updates.append(ledger(study, "reconcile", "--run-id", run_id,
                                  "--state", state, "--elapsed-seconds", int(fields[2]),
                                  "--allocated-gpus", int(allocated_gpus)))
    return {"updates": updates, "budget": ledger(study, "show")["usage"]}


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("submit")
    launch.add_argument("--study", required=True, type=Path)
    launch.add_argument("--source", required=True, type=Path)
    launch.add_argument("--run-id", required=True)
    launch.add_argument("--stage", required=True)
    launch.add_argument("--config", required=True, type=Path)
    launch.add_argument("--seed", type=int, default=42)
    launch.add_argument("--minutes", type=int, default=10)
    launch.add_argument("--mode", choices=("verify", "smoke", "train"), default="train")
    settle = sub.add_parser("reconcile")
    settle.add_argument("--study", required=True, type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    print(json.dumps(submit(**args) if command == "submit" else reconcile(**args), indent=2))


if __name__ == "__main__":
    main()
