#!/usr/bin/env python3
"""Independent PyTorch FP32 dense reference, streaming native GGUF layer weights.

Uses stored (already permuted) Q/K weights and adjacent-pair RoPE. No llama/ggml
execution code is reused. Holds only one block on GPU; outputs are teacher-forced.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'gguf-py'))
import gguf.gguf_reader as reader_module

@torch.inference_mode()
def run(a):
    # PWRI has the same GGUF metadata/tensor layout. Read-only acceptance of its magic.
    reader_module.GGUF_MAGIC=int.from_bytes(b'PWRI','little')
    reader=reader_module.GGUFReader(a.model)
    tensors={t.name:t.data for t in reader.tensors}
    meta=lambda k:reader.fields[k].contents()
    hidden=meta('llama.embedding_length'); heads=meta('llama.attention.head_count')
    layers=meta('llama.block_count'); hd=hidden//heads
    eps=meta('llama.attention.layer_norm_rms_epsilon'); base=meta('llama.rope.freq_base')
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda')
    def weight(name): return torch.from_numpy(np.array(tensors[name],copy=True)).to(device=device,dtype=torch.float32)
    def norm(x,w): return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*w
    rows=[json.loads(s) for s in Path(a.input).open()]
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    metadata=[]
    for row in rows:
        start=time.monotonic()
        prompt=row['tokens']; continuation=row['continuation']; ids=prompt+continuation
        seq=len(ids)
        # Embedding lookup on CPU avoids transferring the full embedding table.
        x=torch.from_numpy(np.array(tensors['token_embd.weight'][ids],copy=True)).to(device=device,dtype=torch.float32)
        positions=torch.arange(seq,device=device,dtype=torch.float32)
        angles=positions[:,None]*base**(-torch.arange(0,hd,2,device=device,dtype=torch.float32)/hd)[None,:]
        cos,sin=angles.cos()[:,None,:],angles.sin()[:,None,:]
        def rope(q):
            result=torch.empty_like(q)
            result[...,0::2]=q[...,0::2]*cos-q[...,1::2]*sin
            result[...,1::2]=q[...,0::2]*sin+q[...,1::2]*cos
            return result
        for layer in range(layers):
            prefix=f'blk.{layer}.'
            h=norm(x,weight(prefix+'attn_norm.weight'))
            q=rope(F.linear(h,weight(prefix+'attn_q.weight')).reshape(seq,heads,hd))
            k=rope(F.linear(h,weight(prefix+'attn_k.weight')).reshape(seq,heads,hd))
            v=F.linear(h,weight(prefix+'attn_v.weight')).reshape(seq,heads,hd)
            # Match the declared F16 KV storage while retaining independent FP32 math.
            k=k.half().float(); v=v.half().float()
            attention=F.scaled_dot_product_attention(q.transpose(0,1).unsqueeze(0),k.transpose(0,1).unsqueeze(0),v.transpose(0,1).unsqueeze(0),is_causal=True)
            attention=attention.squeeze(0).transpose(0,1).reshape(seq,hidden)
            x=x+F.linear(attention,weight(prefix+'attn_output.weight'))
            h=norm(x,weight(prefix+'ffn_norm.weight'))
            gate=F.relu(F.linear(h,weight(prefix+'ffn_gate.weight')))
            up=F.linear(h,weight(prefix+'ffn_up.weight'))
            x=x+(gate*up)@weight(prefix+'ffn_down_t.weight')
            if layer%10==9: print(row['id'],'layer',layer+1,flush=True)
        x=norm(x[len(prompt)-1:],weight('output_norm.weight'))
        logits=F.linear(x,weight('output.weight')).cpu().numpy()
        path=out/(row['id']+'.f32'); logits.astype('<f4').tofile(path)
        metadata.append({'id':row['id'],'shape':list(logits.shape),'logits_file':str(path.resolve()),'elapsed_s':time.monotonic()-start,
            'reference':'independent torch FP32 dense, adjacent-pair RoPE, F16 KV, TF32 off'})
        (out/'reference.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print('reference completed',row['id'],metadata[-1]['elapsed_s'],flush=True)
        del x,logits,gate,up,h,q,k,v,attention
        torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='/home/lush/Models/ReluLLaMA-13B-PowerInfer-GGUF/llama-13b-relu.powerinfer.gguf')
    p.add_argument('--input',required=True)
    p.add_argument('--out',required=True)
    run(p.parse_args())
