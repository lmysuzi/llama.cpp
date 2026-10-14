#!/usr/bin/env python3
"""Render the validated complete summary as paper-friendly tables and figures."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'logs/powerinfer-comparison'
NAMES={'layer':'llama.cpp-layer','atsinfer':'ATSInfer-static+async','powerinfer':'PowerInfer-native'}

def render(summary,out):
    if summary['formal_requests']!=180: raise ValueError('Only render the complete formal matrix')
    lines=['# 共同模型正式实验结果','',
        'ReluLLaMA-13B F16；RTX 4090；同一进程 GPU 上限 20 GiB；6 CPU 线程；context 2560；chunk 512；固定生成 128 token。',
        '', '性能数字为 10 条 prompt 各自三轮平均后的均值 ± prompt 间样本标准差。预热请求不计入统计；共 180 次正式请求。','',
        '| 输入/输出 | 系统 | TTFT (s) | TPOT (ms) | Decode (token/s) | 请求延迟 (s) | GPU 峰值 (GiB) | RSS 峰值 (GiB) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    def metric(r,k,factor=1): return f"{r[k]['mean']*factor:.3f} ± {r[k]['prompt_sd']*factor:.3f}"
    for scenario in ('S','L'):
        for r in summary['statistics']:
            if r['scenario']!=scenario: continue
            resources=[x for x in summary['resources'] if x['system']==r['system']]
            gpu=max(x['gpu_observed_peak_bytes'] for x in resources)/(1<<30)
            rss=max(x['rss_observed_peak_kib'] for x in resources)/(1<<20)
            lines.append(f"| {'128' if scenario=='S' else '2048'}/128 | {NAMES[r['system']]} | {metric(r,'ttft_s')} | {metric(r,'tpot_s',1000)} | {metric(r,'decode_tps')} | {metric(r,'request_s')} | {gpu:.3f} | {rss:.3f} |")
    lines+=['','GPU/RSS 为三轮性能进程的观测峰值，包含加载与预热；GPU 每约 20 ms 采样，不能解释为瞬时硬限额保证。',
        '', '## 配对延迟比','',
        '下面比值是同一 prompt 的三轮均值之比，再对十个 prompt 取几何平均。分子/分母顺序写明；大于 1 表示分母系统更快。','',
        '| 场景 | 分子 / 分母 | TTFT 比 | TPOT 比 | 请求延迟比 |','|---|---|---:|---:|---:|']
    for r in summary['paired_latency_speedups']:
        lines.append(f"| {r['scenario']} | {NAMES[r['numerator_latency_system']]} / {NAMES[r['denominator_latency_system']]} | {r['ttft_s']:.3f} | {r['tpot_s']:.3f} | {r['request_s']:.3f} |")
    lines+=['','## WikiText-2 固定子集','',
        'raw-v1 test；固定 32 个非重叠 1024-token 窗口；每窗口 BOS 不计分；共 32,736 token。PPL 按累计 NLL 计算，并非全量标准测试分数。','',
        '| 系统 | 计分 token | PPL | ΔPPL vs layer | 质量运行 GPU 峰值 (GiB) |','|---|---:|---:|---:|---:|']
    for system,q in summary['quality'].items():
        lines.append(f"| {NAMES[system]} | {q['scored_tokens']} | {q['ppl']:.5f} | {q['delta_ppl_vs_layer']:+.5f} | {q['gpu_peak_bytes']/(1<<30):.3f} |")
    lines+=['','## 适用范围','',
        '这是共同 checkpoint、共同显存上限下的完整系统比较。dense 两条路径共用模型适配和 FP32 累加策略；PowerInfer 保留预测器与原生稀疏计算。',
        'ATSInfer 结果仅代表 static+async 配置。不能据此孤立证明调度粒度、完整在线负载感知机制或所有模型/任务的优劣。',
        '三系统自由生成的 continuation 可以不同；子集 PPL 不能替代聊天、代码或下游任务准确率评估。','']
    out.write_text('\n'.join(lines))

def plot(summary):
    sys.path.insert(0,str(BASE/'plot-deps'))
    os.environ.setdefault('MPLCONFIGDIR',str(BASE/'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=ROOT/'figures/powerinfer-comparison'; dest.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,scenario in zip(axes,('S','L')):
        rows={r['system']:r for r in summary['statistics'] if r['scenario']==scenario}
        systems=list(NAMES); ttft=[rows[s]['ttft_s']['mean'] for s in systems]
        decode=[127*rows[s]['tpot_s']['mean'] for s in systems]
        ax.bar(range(3),ttft,color='#93b9d5',label='Time to first token')
        ax.bar(range(3),decode,bottom=ttft,color='#244b70',label='127 further tokens')
        total=[rows[s]['request_s']['mean'] for s in systems]
        sd=[rows[s]['request_s']['prompt_sd'] for s in systems]
        ax.errorbar(range(3),total,yerr=sd,fmt='none',color='#222222',capsize=3,linewidth=1)
        ax.set_xticks(range(3),['Layer\noffload','ATSInfer\nstatic+async','PowerInfer\nnative'])
        ax.set_title(('128' if scenario=='S' else '2048')+' input / 128 output')
        ax.set_ylabel('Warmed request latency (s)'); ax.set_ylim(bottom=0)
        ax.spines[['right','top']].set_visible(False)
        ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    axes[0].legend(frameon=False,fontsize=9,loc='upper left')
    for suffix in ('png','pdf','svg'): fig.savefig(dest/('latency-breakdown.'+suffix),dpi=220)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--prefix',default='formal'); p.add_argument('--no-plot',action='store_true')
    a=p.parse_args(); summary=json.loads((BASE/(a.prefix+'-summary.json')).read_text())
    out=BASE/(a.prefix+'-summary.md'); render(summary,out)
    if not a.no_plot: plot(summary)
    print(out)
