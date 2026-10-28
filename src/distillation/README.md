# Offline teacher logits

`generate_logits.py` reads the packed training data from `src/prepare.py` and saves
Llama 3.1 8B **Base** predictions. Change the constants at the top of the file;
there are no command-line options. All paths default to `/work/<user>/lab5-training-llama`.

Rows are cached in the order training visits them (`training_order`: `torch.randperm` with
seed 42, as accelerate's sampler does), so a run over the first N tokens finds every row.
`MAX_TOKENS = None` (the default) caches the entire train split, 6B non-padding input tokens,
about **4.65 TB** at Top-128 (2.33 TB at Top-64, 1.18 TB at Top-32). A budget such as `MAX_TOKENS = 1_000_000_000`
caches only the first rows of that order.

To expand **1B → 6B**, set `MAX_TOKENS = 6_000_000_000` and submit again with the
**same `OUTPUT_DIR`**. Completed rows are reused, including rows in the old final
partial shard; only missing rows are evaluated. The resulting cache is about
4.65 TB at Top-128. A smaller budget also reuses the cache and does not delete existing rows.
Use a new directory only when changing the teacher/revision, source dataset,
Top-K or shard format. GPU count and inference batch size can change on restart.

From the repository root on nano4, with access to the teacher on Hugging Face:

```bash
mkdir -p /work/$USER/lab5-training-llama/logs
sbatch scripts/distillation.sbatch
```

For a small first check, set `MAX_TOKENS = 1_000_000` and a separate output directory.
The script also runs on a single allocated GPU with
`.venv/bin/python src/distillation/generate_logits.py`.

Each GPU loads one BF16 teacher and processes different shards independently.
Documents retain their original boundaries and reset positions, matching the
student's attention. The vocabulary projection is chunked so the full
`[batch tokens, vocabulary]` logits tensor is never materialized at once.
Completed `.pt` shards are skipped on restart; unfinished `.tmp` files are rewritten.
Extending a partial shard writes a replacement first, so an interruption leaves
its old completed rows intact. A worker with no missing rows does not load weights.
Only run one job against an output directory at a time.

## Saved tensors

Each file covers positions `[metadata.row_start, metadata.row_stop)` of the training order;
`rows` holds the dataset row of each entry.
It contains all non-padding positions, including each document's last position:

| Key | Shape / dtype | Meaning |
| --- | --- | --- |
| `topk_ids` | `[tokens, 128]`, int32 | Top-128 vocabulary token IDs |
| `topk_logits` | `[tokens, 128]`, float16 | Corresponding raw teacher logits, descending |
| `logsumexp` | `[tokens]`, float32 | Log normalizer over the **entire** vocabulary, at temperature 1 |
| `row_offsets` | `[rows + 1]`, int64 | Token offsets for retrieving each cached row |
| `rows` | `[rows]`, int64 | The dataset row of each cached row |
| `metadata` | dict | Teacher revision, source dataset fingerprint, settings and row range |

```python
import torch

cache = torch.load("shard-0000000.pt", mmap=True, weights_only=True)
row = 0  # Local row index; its dataset row is cache["rows"][row].
start, stop = cache["row_offsets"][row : row + 2].tolist()
ids = cache["topk_ids"][start:stop].long()
logits = cache["topk_logits"][start:stop].float()
log_probs = logits - cache["logsumexp"][start:stop, None]
topk_probs = log_probs.exp()
tail_probability = (1 - topk_probs.sum(-1)).clamp(0, 1)
```

**Alignment:** cache position `t` predicts input token `t + 1`. For a row, pair
`cache[:-1]` with `input_ids[1:]` and mask targets at document starts (BOS), just
like `training.py`. Never train on a prediction crossing a document boundary.
Use the original dataset's `input_ids` and `doc_lens`; they are not duplicated here.
`manifest.json` records the latest **requested** row/token count and the shared
cache settings; it is not a completion marker. Before training, check that the
shards cover that prefix without gaps, using each shard's `row_start`/`row_stop`.
A future student run must sample covered rows, rather than shuffle uncached data.
After requesting a smaller budget, some shards can extend beyond that prefix.

The retained probabilities do **not** sum to one. Preserve the residual mass when
using them for distillation; normalizing only the Top-128 changes the target.
The tail is still an approximation, and the saved normalizer supports temperature
1 only. Other temperatures require regenerating the normalizer from full logits.

Local numerical checks (tiny random Llama on CPU; no downloads):

```bash
.venv/bin/python -m unittest discover -s src/distillation -p 'test_*.py'
```
