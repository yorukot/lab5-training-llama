"""Numerical checks using a tiny CPU Llama; no weights or data are downloaded."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import generate_logits as cache
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM


class LogitCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = LlamaConfig(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=2,
        )
        config._attn_implementation = "sdpa"
        cls.teacher = LlamaForCausalLM(config).eval()
        cls.docs = [[1, 5, 6, 7], [1, 8, 9], [1, 10, 11, 12, 13]]
        cls.rows = [
            {"input_ids": cls.docs[0] + cls.docs[1], "doc_lens": [4, 3], "n_tokens": 7},
            {"input_ids": cls.docs[2], "doc_lens": [5], "n_tokens": 5},
        ]

    def test_packed_cache_matches_documents_scored_separately(self):
        inputs = cache.pack_rows(self.rows, "cpu")
        self.assertEqual(inputs["cu_seq_lens_q"].tolist(), [0, 4, 7, 12])
        self.assertEqual(
            inputs["position_ids"].tolist(), [[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4]]
        )
        # Both batching cases and multiple projection chunks must preserve the same targets.
        for batch_rows in (1, 2):
            with self.subTest(batch_rows=batch_rows):
                with (
                    patch.object(cache, "BATCH_ROWS", batch_rows),
                    patch.object(cache, "LOGIT_CHUNK_TOKENS", 3),
                ):
                    shard = cache.generate_shard(self.teacher, self.rows, [3, 9])
                with torch.inference_mode():
                    reference = torch.cat(
                        [
                            self.teacher(torch.tensor([doc])).logits[0]
                            for doc in self.docs
                        ]
                    )
                expected_logits, expected_ids = reference.topk(cache.TOP_K, dim=-1)
                self.assertEqual(shard["row_offsets"].tolist(), [0, 7, 12])
                self.assertEqual(shard["rows"].tolist(), [3, 9])
                self.assertEqual(shard["topk_ids"].dtype, torch.int32)
                self.assertEqual(shard["topk_logits"].dtype, torch.float16)
                self.assertEqual(shard["logsumexp"].dtype, torch.float32)
                torch.testing.assert_close(shard["topk_ids"].long(), expected_ids)
                torch.testing.assert_close(
                    shard["topk_logits"].float(), expected_logits, atol=2e-4, rtol=1e-3
                )
                torch.testing.assert_close(
                    shard["logsumexp"], reference.logsumexp(-1), atol=1e-6, rtol=1e-6
                )
                log_probs = shard["topk_logits"].float() - shard["logsumexp"][:, None]
                torch.testing.assert_close(
                    log_probs,
                    reference.log_softmax(-1).gather(-1, expected_ids),
                    atol=2e-4,
                    rtol=1e-4,
                )
                # Top-128 of this 256-token vocabulary has a substantial residual tail.
                self.assertTrue(torch.all(log_probs.exp().sum(-1) < 0.9))

    def test_saved_rows_can_be_read_with_mmap_and_resumed(self):
        shard = cache.generate_shard(self.teacher, self.rows, [3, 9])
        metadata = {
            "teacher_id": "tiny-test",
            "dataset_fingerprint": "test",
        }
        shard["metadata"] = {**metadata, "row_start": 5, "row_stop": 7}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard-0000005.pt"
            self.assertIsNone(cache.load_shard(path, metadata, 5))
            cache.save_shard(path, shard)
            self.assertFalse(path.with_suffix(".tmp").exists())
            self.assertIsNotNone(cache.load_shard(path, metadata, 5))
            loaded = torch.load(path, mmap=True, weights_only=True)
            start, stop = loaded["row_offsets"][1:3].tolist()
            separate = cache.generate_batch(self.teacher, self.rows[1:])
            for key, expected in separate.items():
                torch.testing.assert_close(loaded[key][start:stop], expected)
            with self.assertRaisesRegex(ValueError, "choose a new OUTPUT_DIR"):
                cache.load_shard(
                    path, {**metadata, "dataset_fingerprint": "changed"}, 5
                )

            # An interrupted replacement must leave the old completed file readable.
            def interrupted_save(payload, temporary):
                temporary.write_bytes(b"incomplete")
                raise OSError("Interrupted write")

            with (
                patch.object(torch, "save", side_effect=interrupted_save),
                self.assertRaises(OSError),
            ):
                cache.save_shard(path, shard)
            self.assertIsNotNone(cache.load_shard(path, metadata, 5))
            cache.save_shard(path, shard)
            self.assertFalse(path.with_suffix(".tmp").exists())

    def test_training_order_is_the_samplers(self):
        order = cache.training_order(1000)
        generator = torch.Generator()
        generator.manual_seed(42)
        self.assertEqual(
            order.tolist(), torch.randperm(1000, generator=generator).tolist()
        )

    def test_budget_never_splits_or_exceeds_a_row(self):
        counts = [7, 5, 8]
        self.assertEqual(cache.rows_within_budget(counts, 6), 0)
        self.assertEqual(cache.rows_within_budget(counts, 7), 1)
        self.assertEqual(cache.rows_within_budget(counts, 11), 1)
        self.assertEqual(cache.rows_within_budget(counts, 12), 2)
        self.assertEqual(cache.rows_within_budget(counts, 20), 3)
        self.assertEqual(cache.rows_within_budget(counts, None), 3)

    def test_main_extends_partial_shards_without_recomputing_existing_rows(self):
        tokenizer = SimpleNamespace(get_vocab=lambda: {"BOS": 1}, bos_token_id=1)
        with tempfile.TemporaryDirectory() as directory:
            data_dir, output_dir = Path(directory) / "data", Path(directory) / "cache"
            cache.Dataset.from_list(self.rows * 3).save_to_disk(data_dir)
            with (
                patch.multiple(
                    cache,
                    DATA_DIR=data_dir,
                    OUTPUT_DIR=output_dir,
                    MAX_TOKENS=19,  # Three rows: one full shard, then one partial shard.
                    ROWS_PER_SHARD=2,
                    training_order=lambda n: np.arange(n),
                ),
                patch.object(
                    cache,
                    "model_info",
                    return_value=SimpleNamespace(sha="test-revision"),
                ),
                patch.object(
                    cache.PreTrainedTokenizerFast,
                    "from_pretrained",
                    return_value=tokenizer,
                ),
                patch.object(
                    cache.AutoModelForCausalLM,
                    "from_pretrained",
                    return_value=self.teacher,
                ) as load_teacher,
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "set_device"),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                for rank in (0, 1):
                    with patch.dict(
                        os.environ, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2"
                    ):
                        cache.main()
                paths = sorted(output_dir.glob("*.pt"))
                self.assertEqual(len(paths), 2)
                originals = []
                for (start, stop), path in zip(((0, 2), (2, 3)), paths, strict=True):
                    saved = torch.load(path, mmap=True, weights_only=True)
                    originals.append(saved)
                    self.assertEqual(saved["metadata"]["row_start"], start)
                    self.assertEqual(saved["metadata"]["row_stop"], stop)
                    self.assertEqual(
                        saved["metadata"]["teacher_revision"], "test-revision"
                    )
                # A restart can change the worker count without recomputing any shards.
                load_teacher.reset_mock()
                with (
                    patch.dict(os.environ, RANK="0", LOCAL_RANK="0", WORLD_SIZE="1"),
                    patch.object(
                        cache,
                        "generate_shard",
                        side_effect=AssertionError("Already cached"),
                    ),
                ):
                    cache.main()
                load_teacher.assert_not_called()

                full_shard_mtime = paths[0].stat().st_mtime_ns
                with (
                    patch.dict(os.environ, RANK="0", LOCAL_RANK="0", WORLD_SIZE="1"),
                    patch.object(cache, "MAX_TOKENS", 36),  # Expand to all six rows.
                    patch.object(
                        cache, "generate_shard", wraps=cache.generate_shard
                    ) as generate,
                ):
                    cache.main()
                generated_rows = [
                    row for call in generate.call_args_list for row in call.args[1]
                ]
                self.assertEqual(len(generated_rows), 3)
                self.assertEqual(sum(row["n_tokens"] for row in generated_rows), 17)
                self.assertEqual(paths[0].stat().st_mtime_ns, full_shard_mtime)
                extended = torch.load(paths[1], mmap=True, weights_only=True)
                self.assertEqual(extended["metadata"]["row_stop"], 4)
                self.assertEqual(extended["row_offsets"].tolist(), [0, 7, 12])
                for key in ("topk_ids", "topk_logits", "logsumexp"):
                    torch.testing.assert_close(
                        extended[key][:7], originals[1][key], rtol=0, atol=0
                    )
                manifest = json.loads((output_dir / "manifest.json").read_text())
                self.assertEqual(manifest["requested_rows"], 6)
                self.assertEqual(manifest["requested_tokens"], 36)
                self.assertEqual(len(list(output_dir.glob("*.pt"))), 3)

                # A smaller request also leaves the larger cache intact.
                with (
                    patch.dict(os.environ, RANK="0", LOCAL_RANK="0", WORLD_SIZE="1"),
                    patch.object(cache, "MAX_TOKENS", 7),
                    patch.object(
                        cache,
                        "generate_shard",
                        side_effect=AssertionError("Already cached"),
                    ),
                ):
                    cache.main()
                self.assertEqual(paths[0].stat().st_mtime_ns, full_shard_mtime)
                self.assertEqual(len(list(output_dir.glob("*.pt"))), 3)


if __name__ == "__main__":
    unittest.main()
