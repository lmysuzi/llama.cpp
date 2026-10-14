#!/usr/bin/env python3
import json
import math
from pathlib import Path
import tempfile
import unittest
from summarize_comparison import summarize, SYSTEMS, read

def jsonl(path,rows): path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.base=Path(self.tmp.name)
        inputs=self.base/'inputs'; inputs.mkdir()
        self.prompts=[{'id':f'{s}-{i}','scenario':s,'split':'test','tokens':[1]*n}
            for s,n in [('S',128),('L',2048)] for i in range(10)]
        # U+2028 is valid inside a JSON string and must not split JSONL records.
        self.prompts[0]['text']='first\u2028second'
        (inputs/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in self.prompts))
        jsonl(inputs/'quality.jsonl',[{'id':f'q{i}'} for i in range(32)])
        for system,factor in zip(SYSTEMS,[1,0.5,0.25]):
            for round_id in range(3):
                dest=self.base/f't-r{round_id}-{system}'; dest.mkdir()
                records=[{'type':'metadata','start_monotonic_ns':50}]
                for i,p in enumerate(self.prompts):
                    # The paired ratio remains two even with prompt/round variation.
                    tpot=factor*(i+1)*(round_id+1)/1000
                    first=100+int(1e9*factor)
                    times=[first+round(k*1e9*tpot) for k in range(128)]
                    records.append({'type':'perf','id':p['id'],'scenario':p['scenario'],'split':'test',
                        'system':system,'round':round_id,'status':'ok',
                        'n_prompt':len(p['tokens']),'n_gen':128,'output_tokens':[2]*128,
                        'token_monotonic_ns':times,'ttft_s':factor,'tpot_s':tpot,
                        'request_s':factor+127*tpot,'decode_tps':1/tpot,'start_monotonic_ns':100})
                records.append({'type':'complete'})
                jsonl(dest/'results.jsonl',records)
                self.run_record(dest)
                (dest/'telemetry.csv').write_text('monotonic_ns,gpu_bytes,other_gpu_bytes\n10,1000,10\n60,1500,20\n101,2000,30\n')
            dest=self.base/f't-quality-{system}'; dest.mkdir(); self.run_record(dest)
            jsonl(dest/'results.jsonl',[{'type':'quality','id':f'q{i}','system':system,'status':'ok','nll_sum':1023*(1+i%2),'scored_tokens':1023} for i in range(32)])

    def tearDown(self): self.tmp.cleanup()

    def run_record(self,dest):
        (dest/'run.json').write_text(json.dumps({'exit_code':0,'complete':True,'within_cap':True,'gpu_observed_peak_bytes':2000}))

    def test_complete_matrix_and_statistics(self):
        result=summarize(self.base,'t')
        self.assertEqual(result['formal_requests'],180)
        self.assertAlmostEqual(result['quality']['layer']['ppl'],math.exp(1.5))
        self.assertAlmostEqual(result['paired_latency_speedups'][0]['request_s'],2)
        self.assertEqual(result['resources'][0]['phase_peak_bytes'],{'initialization':1000,'warmup':1500,'measurement':2000})

    def test_missing_request_is_rejected(self):
        p=self.base/'t-r0-layer/results.jsonl'; rows=read(p)
        jsonl(p,rows[:1]+rows[2:])
        with self.assertRaisesRegex(ValueError,'Incomplete prompt matrix'): summarize(self.base,'t')

    def test_bad_timing_is_rejected(self):
        p=self.base/'t-r0-layer/results.jsonl'; rows=read(p)
        rows[1]['request_s']+=1; jsonl(p,rows)
        with self.assertRaisesRegex(ValueError,'Timing identity'): summarize(self.base,'t')

    def test_failed_run_is_not_silently_dropped(self):
        p=self.base/'t-r1-layer/run.json'; r=json.loads(p.read_text()); r['exit_code']=1; p.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,'Failed run'): summarize(self.base,'t')

if __name__=='__main__': unittest.main()
