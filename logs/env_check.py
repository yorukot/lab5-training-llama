import torch, transformers, datasets, accelerate, wandb, liger_kernel, flash_attn_interface
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("Muon in torch.optim:", hasattr(torch.optim, "Muon"))
print("transformers", transformers.__version__, "| datasets", datasets.__version__, "| accelerate", accelerate.__version__, "| wandb", wandb.__version__)
q = torch.randn(1, 256, 8, 64, device="cuda", dtype=torch.bfloat16)
out = flash_attn_interface.flash_attn_func(q, q, q, causal=True)
out = out[0] if isinstance(out, tuple) else out
print("FA3 forward OK:", tuple(out.shape), out.dtype)
