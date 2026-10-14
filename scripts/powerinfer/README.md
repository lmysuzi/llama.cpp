# ReluLLaMA common-model comparison

Protocol: [POWERINFER_COMPARISON_PLAN.md](../../POWERINFER_COMPARISON_PLAN.md).
Progress and results: [POWERINFER_COMPARISON_RESULTS.md](../../POWERINFER_COMPARISON_RESULTS.md).

The current first delivery is the smaller two-engine pilot (12 measured requests):

```bash
/home/lush/miniconda3/bin/python scripts/powerinfer/run_pilot.py
```

It uses the validated fixed layout and held-out prompts. The larger suite below
is retained for a later paper-quality evaluation and is not the current default.

Run from the repository root. Python examples use `/home/lush/miniconda3/bin/python`,
which contains PyTorch, NumPy, PyArrow and NVML bindings. GPU commands require host
GPU access. Runs are serial and bind inference to physical cores 0–5; telemetry
uses cores 6–7. Do not run other GPU experiments concurrently.

```bash
bash scripts/powerinfer/build_comparison.sh
/home/lush/miniconda3/bin/python scripts/powerinfer/prepare_comparison.py fetch
/home/lush/miniconda3/bin/python scripts/powerinfer/prepare_comparison.py prepare
/home/lush/miniconda3/bin/python scripts/powerinfer/verify_inputs.py
```

Fetch stores immutable revisions and file hashes. Prepare reuses the pinned files;
do not regenerate inputs after freezing. Both native runners read exactly the same
IDs. Legacy tokenization has different leading-space normalization, so tokenizer
equality is diagnosed separately from the required vocabulary/ID equality.

The independent PyTorch reference and C++ scalar operator tests are diagnostic
tools. The reference streams original F16 weights one layer at a time and performs
FP32 math with F16 KV. It does not use llama/ggml execution code.

```bash
build-release/bin/test-relullama-outprod
/home/lush/miniconda3/bin/python scripts/powerinfer/reference_relullama.py \
  --input logs/powerinfer-comparison/inputs/calibration.jsonl \
  --out logs/powerinfer-comparison/reference
taskset -c 0-5 build-release/bin/llama-pipo-perf \
  -m /home/lush/Models/ReluLLaMA-13B-PowerInfer-GGUF/llama-13b-relu.powerinfer.gguf \
  -c logs/powerinfer-comparison/profile-f32.json -t 6
build-release/bin/llama-pipo-override \
  -p logs/powerinfer-comparison/profile-f32.json --mem-mib 16384 \
  -c logs/powerinfer-comparison/layout16384-f32
```

Use the explicit `-p` option: the existing solver ignores positional profile paths.
Save a copy of the generated `perf.json` as `static.json` with `offloads: []`.
This removes the generator's unrelated decode heuristic; prefill still uses
dynamic slots and transfers. The checked-in execution record identifies the
calibrated layout; do not regenerate it between formal rounds.

An individual run preserves its full command, binary/library hashes, stdout,
stderr, per-token timestamps, telemetry and exit status. Existing output
directories cause an error. PowerInfer must use `ngl=0` (default); sparse mode in
this commit rejects nonzero values. Its workspace GPU index is independent of the
original model directory and is regenerated when the native budget check requires it.

```bash
/home/lush/miniconda3/bin/python scripts/powerinfer/experiment.py \
  --tag example-layer --engine layer --ngl 29 \
  --input logs/powerinfer-comparison/inputs/calibration.jsonl
/home/lush/miniconda3/bin/python scripts/powerinfer/calibrate_remaining.py
/home/lush/miniconda3/bin/python scripts/powerinfer/freeze_comparison.py
/home/lush/miniconda3/bin/python scripts/powerinfer/run_suite.py \
  --freeze logs/powerinfer-comparison/protocol.json --prefix formal
```

`calibrate_remaining.py` resumes successful calibration directories and refuses
failed/incomplete ones; it requires the earlier PowerInfer and ATS logits checks
recorded in the execution report. The freeze script refuses missing gates or a
changed index. `run_suite.py` checks frozen hashes, runs three rotated system orders,
then the fixed quality subset. Use `--resume` only to skip already completed runs;
failed runs require inspection and an explicit rerun record, never silent deletion.

```bash
/home/lush/miniconda3/bin/python scripts/powerinfer/test_summary.py
/home/lush/miniconda3/bin/python scripts/powerinfer/summarize_comparison.py --prefix formal
```

The summary refuses incomplete matrices, wrong token counts, invalid timestamps,
failed runs and invalid NLL. It averages each prompt across rounds before reporting
the mean and standard deviation across ten prompts. Speedups are paired geometric
means. Subset PPL is `exp(total NLL / total scored tokens)`, not mean window PPL.
`run_baseline.sh` is the historical bring-up script and uses a different protocol.
