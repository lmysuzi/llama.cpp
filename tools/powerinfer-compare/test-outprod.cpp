#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    ggml_backend_load_all();
    auto cpu=ggml_backend_cpu_init();
    std::vector<ggml_backend_t> backends{cpu};
    for(size_t i=0;i<ggml_backend_dev_count();++i) {
        auto d=ggml_backend_dev_get(i);
        if(ggml_backend_dev_type(d)==GGML_BACKEND_DEVICE_TYPE_GPU) backends.push_back(ggml_backend_dev_init(d,nullptr));
    }
    bool ok=true;
    for(auto shape:std::vector<std::vector<int64_t>>{{33,57,1},{1025,67,5},{5120,13824,1},{5120,13824,8}}) {
        auto m=shape[0],k=shape[1],n=shape[2];
        std::mt19937 rng(123); std::uniform_real_distribution<float> dist(-0.05f,0.05f);
        std::vector<ggml_fp16_t> w(m*k); std::vector<float> x(n*k), reference(m*n,0);
        for(auto & v:w) v=ggml_fp32_to_fp16(dist(rng));
        // Exactly representable half activations isolate kernel correctness from rounding.
        for(auto & v:x) v=ggml_fp16_to_fp32(ggml_fp32_to_fp16(dist(rng)));
        for(int64_t j=0;j<n;++j) for(int64_t r=0;r<k;++r) for(int64_t i=0;i<m;++i)
            reference[j*m+i]+=ggml_fp16_to_fp32(w[r*m+i])*x[j*k+r];
        for(bool transposed:{false,true}) for(auto backend:backends) for(int threads:{1,6}) {
            if(backend!=cpu && threads==1) continue;
            ggml_backend_cpu_set_n_threads(cpu,threads);
            auto ctx=ggml_init({ggml_tensor_overhead()*32+ggml_graph_overhead_custom(32,false),nullptr,true});
            auto a=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,m,k);
            auto bbase=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,transposed?k:n,transposed?n:k);
            auto b=transposed?ggml_transpose(ctx,bbase):bbase;
            auto out=ggml_out_prod(ctx,a,b);
            auto gf=ggml_new_graph_custom(ctx,32,false); ggml_build_forward_expand(gf,out);
            auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
            if(!buffer) return 2;
            ggml_backend_tensor_set(a,w.data(),0,w.size()*2);
            std::vector<float> input(x.size());
            for(int64_t j=0;j<n;++j) for(int64_t r=0;r<k;++r) input[transposed?j*k+r:r*n+j]=x[j*k+r];
            ggml_backend_tensor_set(bbase,input.data(),0,input.size()*4);
            if(ggml_backend_graph_compute(backend,gf)!=GGML_STATUS_SUCCESS) return 3;
            auto start=std::chrono::steady_clock::now();
            for(int repeat=0;repeat<3;++repeat) if(ggml_backend_graph_compute(backend,gf)!=GGML_STATUS_SUCCESS) return 4;
            auto end=std::chrono::steady_clock::now();
            std::vector<float> got(m*n); ggml_backend_tensor_get(out,got.data(),0,got.size()*4);
            double err=0,denom=0,maxerr=0;
            for(size_t i=0;i<got.size();++i) {
                double d=got[i]-reference[i]; err+=d*d; denom+=double(reference[i])*reference[i]; maxerr=std::max(maxerr,std::abs(d));
                if(!std::isfinite(got[i])) ok=false;
            }
            double rel=std::sqrt(err/std::max(denom,1e-30));
            ok=ok && rel<1e-4 && maxerr<1e-4;
            printf("backend=%s m=%lld k=%lld n=%lld transpose=%d threads=%d ms=%.3f relative_l2=%.9g max_abs=%.9g\n",
                ggml_backend_name(backend),(long long)m,(long long)k,(long long)n,transposed,threads,
                std::chrono::duration<double,std::milli>(end-start).count()/3,rel,maxerr);
            fflush(stdout); ggml_backend_buffer_free(buffer); ggml_free(ctx);
        }
    }
    for(auto b:backends) ggml_backend_free(b);
    printf("%s\n",ok?"PASS":"FAIL"); return ok?0:1;
}
