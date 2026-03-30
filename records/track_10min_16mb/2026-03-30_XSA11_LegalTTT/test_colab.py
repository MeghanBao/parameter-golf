"""
Colab smoke test for train_gpt.py
Validates: ResidLambdas + SLOT + TTT logic on a single GPU with synthetic data.

Usage in Colab:
  !pip install sentencepiece torch --quiet
  !python test_colab.py
"""
import sys, types, os

# ── 1. Patch flash_attn_interface before train_gpt imports it ──────────────
def _fake_flash_attn(q, k, v, causal=True):
    """Standard scaled dot-product attention as a drop-in replacement."""
    import torch, torch.nn.functional as F
    # q/k/v: (bsz, seqlen, nheads, head_dim)
    bsz, seqlen, nh, hd = q.shape
    q = q.transpose(1, 2)   # (bsz, nh, seqlen, hd)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    # kv heads may differ from q heads (GQA) — repeat_interleave
    nkv = k.shape[1]
    if nh != nkv:
        k = k.repeat_interleave(nh // nkv, dim=1)
        v = v.repeat_interleave(nh // nkv, dim=1)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return y.transpose(1, 2).contiguous()   # (bsz, seqlen, nh, hd)

fa_mod = types.ModuleType("flash_attn_interface")
fa_mod.flash_attn_func = _fake_flash_attn
sys.modules["flash_attn_interface"] = fa_mod

# ── 2. Fake sentencepiece (only used for tokenizer path, not needed here) ──
if "sentencepiece" not in sys.modules:
    sp_mod = types.ModuleType("sentencepiece")
    class FakeSPP:
        def Load(self, path): pass
        def EncodeAsIds(self, text): return [1]
        def piece_to_id(self, p): return 0
    sp_mod.SentencePieceProcessor = FakeSPP
    sys.modules["sentencepiece"] = sp_mod

# ── 3. Env vars: tiny model, synthetic data, SLOT + ResidLambdas enabled ──
os.environ.update({
    # Model
    "NUM_LAYERS": "4",
    "MODEL_DIM": "64",
    "NUM_HEADS": "4",
    "NUM_KV_HEADS": "2",
    "BIGRAM_VOCAB_SIZE": "64",
    "BIGRAM_DIM": "16",
    "XSA_LAST_N": "4",
    "VE_ENABLED": "1",
    "VE_DIM": "16",
    "VE_LAYERS": "2,3",
    # Training (tiny)
    "ITERATIONS": "8",
    "WARMUP_STEPS": "2",
    "WARMDOWN_ITERS": "2",
    "TRAIN_BATCH_TOKENS": "512",
    "TRAIN_SEQ_LEN": "64",
    "MAX_WALLCLOCK_SECONDS": "300",
    "EVAL_STRIDE": "32",
    "SEED": "42",
    # Features under test
    "SLOT_ENABLED": "1",
    "SLOT_LR": "0.003",
    "SLOT_STEPS": "2",       # reduce steps for speed
    "TTT_ENABLED": "1",
    "TTT_LR": "0.002",
    "TTT_EPOCHS": "1",
    "TTT_CHUNK_TOKENS": "256",
    "TTT_FREEZE_BLOCKS": "0",
    "TTT_MOMENTUM": "0.9",
    "TTT_BATCH_SEQS": "2",
    "TTT_GRAD_CLIP": "1.0",
    # Optimizers
    "MATRIX_LR": "0.025",
    "SCALAR_LR": "0.025",
    "TIED_EMBED_LR": "0.035",
    "MUON_WD": "0.04",
    "ADAM_WD": "0.04",
    # Disable heavy features for speed
    "QAT_ENABLED": "0",
    "SWA_ENABLED": "0",
    "LAWA_ENABLED": "0",
    "GATED_ATTENTION": "0",
    "DTG_ENABLED": "0",
    "VALUE_RESIDUAL": "0",
    "MTP_NUM_HEADS": "0",
})

# ── 4. Synthetic data files ────────────────────────────────────────────────
import numpy as np, struct, tempfile, pathlib

VOCAB = 1024
SEQ   = 2048   # must match default train_gpt.py expectations for file format
N_TRAIN = 8
N_VAL   = 4

data_dir = pathlib.Path(tempfile.mkdtemp()) / "fineweb10B_sp1024"
data_dir.mkdir(parents=True)
os.environ["DATA_PATH"] = str(data_dir.parent)

rng = np.random.default_rng(0)

def write_bin(path, n_seqs):
    """Write a .bin file in the format expected by train_gpt.py (uint16 tokens)."""
    tokens = rng.integers(1, VOCAB, size=(n_seqs * SEQ + 1,), dtype=np.uint16)
    tokens.tofile(path)

for i in range(2):
    write_bin(data_dir / f"fineweb_train_{i:06d}.bin", N_TRAIN)
write_bin(data_dir / "fineweb_val_000000.bin", N_VAL)

# Also write a fake tokenizer model (sentencepiece won't be called with real data)
fake_tok_dir = data_dir.parent / "tokenizers"
fake_tok_dir.mkdir(exist_ok=True)
(fake_tok_dir / "fineweb_1024_bpe.model").write_bytes(b"fake")

os.environ["TOKENIZER_PATH"] = str(fake_tok_dir / "fineweb_1024_bpe.model")

# ── 5. Patch tokenizer loading to skip sentencepiece ──────────────────────
# train_gpt reads tokenizer only for val_bpb byte counting;
# we stub out the byte-level LUT building to return dummy tensors.

import unittest.mock as mock
import importlib, builtins

_orig_open = builtins.open
def _patched_open(path, *a, **kw):
    if "fineweb_1024_bpe.model" in str(path):
        import io
        return io.BytesIO(b"fake")
    return _orig_open(path, *a, **kw)

# ── 6. Run main() ─────────────────────────────────────────────────────────
print("=" * 60)
print("Smoke test: ResidLambdas + SLOT + TTT on synthetic data")
print("=" * 60)

# torchrun is not available; simulate single-process distributed
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Patch spm loading inside train_gpt
with mock.patch("builtins.open", _patched_open):
    import train_gpt as tg

# Patch val_bpb tokenizer init to avoid spm.Load crash
_orig_init = tg.Hyperparameters.__init__ if hasattr(tg.Hyperparameters, "__init__") else None

# The tokenizer LUT is built in main(); patch spm.SentencePieceProcessor.Load
original_load = None
try:
    import sentencepiece as spm_real
    original_load = spm_real.SentencePieceProcessor.Load
    def _noop_load(self, path): pass
    spm_real.SentencePieceProcessor.Load = _noop_load
except Exception:
    pass

# Patch the piece/byte LUT building to return dummy tensors
_orig_main = tg.main

def _patched_main():
    """Wrap main() to intercept tokenizer-dependent LUT building."""
    import torch
    # Monkey-patch build_byte_luts if it exists
    if hasattr(tg, "build_byte_luts"):
        _orig_luts = tg.build_byte_luts
        def _fake_luts(*a, **kw):
            vocab = VOCAB
            return (
                torch.ones(vocab, dtype=torch.float32),   # base_bytes_lut
                torch.zeros(vocab, dtype=torch.bool),      # has_leading_space_lut
                torch.zeros(vocab, dtype=torch.bool),      # is_boundary_token_lut
            )
        tg.build_byte_luts = _fake_luts

    _orig_main()

    if hasattr(tg, "build_byte_luts"):
        tg.build_byte_luts = _orig_luts

try:
    tg.main()
    print("\n✓ Smoke test PASSED — ResidLambdas + SLOT + TTT ran without errors.")
except SystemExit as e:
    if e.code == 0:
        print("\n✓ Smoke test PASSED")
    else:
        print(f"\n✗ Smoke test FAILED (exit {e.code})")
        raise
except Exception as e:
    print(f"\n✗ Smoke test FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)
