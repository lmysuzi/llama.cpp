#!/usr/bin/env python3
"""动态提升(dynamic promotion)收益理论分析 — 基于本机实测 profile 数据。

数据来源:
  - examples/pipo-alg/static_cfg/4090/pipo_profile.json  (本机 4090 逐张量实测, batch-1 decode)
  - examples/pipo-alg/alg_cfg/perf.json                  (当前生效的 overrides/offloads)
  - logs/run-2026-09-13/*.md                             (llama-bench 实测吞吐)

模型:
  T_step(-do 0) = A_cpu + B_gpu + gap            (串行执行)
  提升 k 个张量: ΔT = Σ(t_c - t_g) - Σ f*w_i - k*overhead
    w_i = 传输时间 = bytes(激活专家)/BW, BW=15.6 GB/s (实测)
    f = 传输暴露比例; 盈亏平衡要求 f < (t_c-t_g)/w_i
  带宽约束: Σ w_i <= T_step (DMA 引擎占用不能超过步长)
"""
import json, re, sys

PROF = 'examples/pipo-alg/static_cfg/4090/pipo_profile.json'
CFG  = 'examples/pipo-alg/alg_cfg/perf.json'
BW   = 15.6          # GB/s, pinned H2D 实测
ACTIVE = 10 / 512    # decode 每个专家张量实际传输比例 (top-10 of 512)
T_DO0  = 28.2        # ms, 实测 pipo -do 0 (35.4 tok/s)
T_DO1  = 31.0        # ms, 实测 pipo -do 1 (32.3 tok/s)

prof = {e['weight_name']: e for e in json.load(open(PROF))}
cfg  = json.load(open(CFG))

def match(names, pats):
    return set(n for n in names if any(re.search(p, n) for p in pats))

ov = match(prof, cfg['overrides'])
of = match(prof, cfg['offloads']) & ov
rest = set(prof) - ov

A  = sum(prof[n]['CPU_time'] for n in ov)
B  = sum(prof[n]['GPU_time'] for n in rest)
gain_now = sum(prof[n]['CPU_time'] - prof[n]['GPU_time'] for n in of)
tr_now   = sum(prof[n]['size'] * ACTIVE / 1e6 / BW for n in of)

print(f"== 单步分解 (-do 0, 实测 {T_DO0} ms) ==")
print(f"  A. CPU 常驻 {len(ov)} 张量计算: {A:.2f} ms")
print(f"  B. GPU 权重算子计算:          {B:.2f} ms  (不含 attention 等非权重算子)")
print(f"  gap = 实测 - A - B:           {T_DO0-A-B:.2f} ms  (split边界/同步/激活拷贝)")

print(f"\n== 当前提升集 (-do 1, {len(of)} 个张量) ==")
print(f"  理想节省: {gain_now:.2f} ms | 需 DMA: {tr_now:.2f} ms | 实测回归: +{T_DO1-T_DO0:.1f} ms")

print("\n== 盈亏平衡暴露比例 f* (传输暴露低于此才不亏) ==")
for n in ('blk.30.ffn_gate_exps.weight', 'blk.30.ffn_down_exps.weight'):
    e = prof[n]
    saved = e['CPU_time'] - e['GPU_time']
    w = e['size'] * ACTIVE / 1e6 / BW
    print(f"  {n:32s} saved={saved:.3f} w={w:.3f} -> f*={saved/w*100:.0f}%  (需BW>={e['size']*ACTIVE/1e6/saved:.0f} GB/s)")

print("\n== 情景推演 ==")
downs  = sorted(n for n in ov if 'ffn_down_exps' in n)
gateup = sorted(n for n in ov if 'ffn_gate_exps' in n or 'ffn_up_exps' in n)
gd = sum(prof[n]['CPU_time']-prof[n]['GPU_time'] for n in downs)  # down 张量收益
gu = 0.086   # 单个 gate/up 张量收益 (ms, 实测均值)
w  = 0.378   # 单张量 DMA 时间 (ms: 5.9MB @ 15.6GB/s)

def scenario(name, T, note=''):
    print(f"  {name:46s} T={T:5.1f} ms  {1000/T:5.1f} tok/s  {note}")

# S2: 只修机制(双缓冲/全隐藏), 保持当前5张量集
scenario("S2 修机制, 保持当前5张量集", T_DO0 - gain_now + 0.5, "≈现状, 无意义")
# S3: 修机制 + 带宽最优集合 (5 down + k gate/up), 不修 gap
k3 = (T_DO0 - gd - w*len(downs)) / (w + gu)
T3 = T_DO0 - gd - gu*k3
scenario(f"S3 修机制+最优集合(5down+{k3:.0f}gate/up)", T3, f"DMA={w*(5+k3):.0f}ms/步, 带宽饱和")
# S4: 只修 gap (attention~3ms + 残余~3ms 保留)
T4 = 3 + A + B + 3
scenario("S4 只修 gap (14.2ms -> ~6ms)", T4, "B未含attention已单列")
# S5: 双修, 带宽自洽
k5 = (T4 - gd - w*5) / (w + gu)
T5 = T4 - gd - gu*k5
scenario(f"S5 双修(5down+{k5:.0f}gate/up)", T5, f"DMA={w*(5+k5):.0f}ms/步, 上限")
