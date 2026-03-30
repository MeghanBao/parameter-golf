# XSA11 + Legal TTT + SLOT + ResidLambdas

**val_bpb: TBD** (pending 3-seed run) | ~15.x MB | 8×H100 SXM

Built on PR #549 (abaybektursun) + PR #414 (signalrush) stack.

## Key Changes Over PR #549

### 1. XSA on All 11 Layers (`xsa_last_n`: 4 → 11)
Exclusive Self-Attention extended to every transformer block, not just the last 4.
Rascal (PR #1120) demonstrated this as the single largest architectural gain:
pre-TTT baseline 1.1099 vs PR #549's 1.1218, a ~0.012 BPB improvement without TTT.

### 2. Warmdown Shortened (3500 → 2000 steps)
Rascal showed shorter warmdown works better with XSA-all.
More training steps at full LR before the final decay phase.

### 3. QAT Global Flag Fix
`CastedLinear._qat_enabled` was a class attribute — `torch.compile(fullgraph=True)`
constant-folds it to `False` at trace time and never recompiles when the flag flips.
Fix: module-level `_qat_enabled: bool` variable, which torch.compile guards and
recompiles on when it changes to `True` at `scale < LATE_QAT_THRESHOLD`.

### 4. SWA Actually Applied Fix
SWA checkpoints were collected during warmdown (`swa_state`, `swa_count`) but the
weight-averaging block only branched on LAWA or EMA — SWA was never applied.
Fix: added `elif swa_state is not None and swa_count > 0` branch to correctly
average and load SWA weights before quantization and eval.

### 5. SLOT (Sample-specific LM Optimization at Test-time)
Per-batch δ ∈ ℝ^512 vector optimized at the last hidden layer during TTT scoring
(Phase 1). 5 AdamW steps (lr=0.003) per batch, zero cost to artifact size.
SLOT and TTT address complementary bottlenecks: TTT adapts all weights to local
data distribution (chunk-level), SLOT fine-tunes the final hidden→logit mapping
per-batch. The two stack because they operate at different granularities.

### 6. Residual Lambdas
Learnable per-sublayer residual scaling: `x_out = λ_attn * x_in + attn_scale * attn_out`
and `x_out = λ_mlp * x_out + mlp_scale * mlp_out`. Init = √1.1 ≈ 1.049 (near-identity).
Optimized via AdamW at 5× scalar_lr with no weight decay. From PR #1130: ~−0.005 BPB.

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with LeakyReLU(0.5)² |
| **XSA** | **All 11 layers** (was: last 4) |
| BigramHash | 1536 |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9,10 |
| **ResidLambdas** | **√1.1 init, 5× scalar_lr, no WD** |
| Weight avg | EMA(0.997) + **SWA(every 50, now actually applied)** |
| Quantization | GPTQ-lite int6 + lzma |
| Optimizer | Parallel Muon (Parameter Banking) |
| **Warmdown** | **2000 iters** (was: 3500) |
| Late QAT | STE at lr_scale < 0.15 (now correctly triggers) |
| **SLOT** | **δ∈ℝ⁵¹², 5 steps, lr=0.003** |

## Run Command

```bash
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
SLOT_ENABLED=1 SLOT_LR=0.003 SLOT_STEPS=5 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected Improvement

| Change | Estimated BPB delta |
|--------|---------------------|
| XSA-all (11L vs 4L) | ~−0.012 (from Rascal PR #1120) |
| SWA bug fix | ~−0.001 to −0.003 |
| QAT bug fix | ~−0.001 to −0.002 |
| Legal TTT | ~−0.002 (from PR #549 ablation) |
| ResidLambdas | ~−0.005 (from PR #1130) |
| SLOT (on top of TTT) | ~−0.006 to −0.007 (from PR #1128) |
| **Total vs PR #549** | **~−0.027 to −0.031** |
| **Expected BPB** | **~1.088 to 1.092** |

Current SOTA: PR #1089 = 1.1091

## Credits

- **XSA-all**: PR #1120 (Rascal) by @newjordan
- **Legal TTT**: PR #461 by @Christopher-Lee-McClendon
- **SLOT**: PR #1128 by @AnubhavBharadwaaj (Hu et al., arXiv:2505.12392v2)
- **ResidLambdas**: PR #1130 by @alejandro-co
- **LeakyReLU² activation**: PR #493 by @parinzee, PR #518 by @sofiabod
- **Parallel Muon / Parameter Banking**: PR #399 by @abaybektursun
- **Base model (11L XSA4 EMA)**: PR #414 by @signalrush, PR #549 by @abaybektursun
