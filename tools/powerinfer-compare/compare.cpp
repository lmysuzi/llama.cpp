// One workload/timing implementation, compiled against the two native engines.
#include "llama.h"
#ifndef POWERINFER_LEGACY
#include "ggml-backend.h"
#endif
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>

using json = nlohmann::json;
static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static json read_json(const std::string & path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    json j; f >> j; return j;
}
static void emit(std::ostream & out, const json & j) { out << j.dump() << '\n' << std::flush; }
static std::string hex(const char * s) {
    const char * digits = "0123456789abcdef";
    std::string r;
    for (const unsigned char * p = (const unsigned char *)s; *p; ++p) {
        r += digits[*p >> 4]; r += digits[*p & 15];
    }
    return r;
}

struct Engine {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_batch batch{};
    int nv = 0, bos = -1, last_n = 0;
#ifndef POWERINFER_LEGACY
    const llama_vocab * vocab = nullptr;
#endif
    std::vector<std::string> patterns;
#ifndef POWERINFER_LEGACY
    std::vector<llama_model_tensor_buft_override> overrides;
#endif
    Engine(const json & a) {
        const bool vocabulary = a.at("mode") == "tokenize" || a.at("mode") == "vocab";
#ifdef POWERINFER_LEGACY
        llama_backend_init(false);
#else
        ggml_backend_load_all();
        llama_backend_init();
#endif
        auto mp = llama_model_default_params();
        auto cp = llama_context_default_params();
        mp.n_gpu_layers = a.at("ngl");
        mp.vocab_only = vocabulary;
        cp.n_ctx = a.at("ctx"); cp.n_batch = a.at("batch");
        cp.n_threads = a.at("threads"); cp.n_threads_batch = a.at("threads");
        mp.use_mmap = true;
#ifdef POWERINFER_LEGACY
        mp.vram_budget_gb = a.at("vram");
        mp.reset_gpu_index = a.at("reset_index");
        cp.f16_kv = true;
        cp.seed = 20260916;
        model = llama_load_model_from_file_with_context(a.at("model").get<std::string>().c_str(), mp, vocabulary ? nullptr : &cp);
#else
        cp.n_ubatch = a.at("batch");
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        mp.enable_pipo = a.at("pipo"); cp.enable_pipo = a.at("pipo");
        if (mp.enable_pipo) {
            auto config = read_json(a.at("config"));
            if (!config.at("offloads").empty()) throw std::runtime_error("static protocol requires empty offloads");
            patterns = config.at("overrides").get<std::vector<std::string>>();
            ggml_backend_buffer_type_t gpu = nullptr, host = nullptr;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                    gpu = ggml_backend_dev_buffer_type(dev); host = ggml_backend_dev_host_buffer_type(dev); break;
                }
            }
            if (!gpu || !host) throw std::runtime_error("PIPO requires GPU and pinned host buffers");
            for (const auto & s : patterns) overrides.push_back({s.c_str(), host});
            overrides.push_back({".*", gpu}); overrides.push_back({nullptr, nullptr});
            mp.tensor_buft_overrides = overrides.data();
        }
        model = llama_model_load_from_file(a.at("model").get<std::string>().c_str(), mp);
#endif
        if (!model) throw std::runtime_error("model load failed");
#ifdef POWERINFER_LEGACY
        nv = llama_n_vocab(model); bos = llama_token_bos(model);
#else
        vocab = llama_model_get_vocab(model); nv = llama_vocab_n_tokens(vocab); bos = llama_vocab_bos(vocab);
        if (mp.enable_pipo) {
            std::vector<const char *> prefill;
            for (const auto & s : patterns) if (s.find("blk") != std::string::npos) prefill.push_back(s.c_str());
            llama_model_set_offload(model, prefill.data(), nullptr, prefill.size(), 0);
        }
#endif
        if (vocabulary) return;
#ifdef POWERINFER_LEGACY
        ctx = llama_new_context_with_model(model, cp);
#else
        ctx = llama_init_from_model(model, cp);
#endif
        if (!ctx) throw std::runtime_error("context creation failed");
        if ((int)llama_n_ctx(ctx) != a.at("ctx").get<int>()) throw std::runtime_error("context capacity changed");
        batch = llama_batch_init(a.at("batch"), 0, 1);
    }
    ~Engine() {
        if (ctx) { llama_batch_free(batch); llama_free(ctx); }
#ifdef POWERINFER_LEGACY
        if (model) llama_free_model(model);
#else
        if (model) llama_model_free(model);
#endif
        llama_backend_free();
    }
    std::vector<llama_token> tokenize(const std::string & text) {
#ifdef POWERINFER_LEGACY
        auto * v = model;
#else
        auto * v = vocab;
#endif
        int n = llama_tokenize(v, text.data(), text.size(), nullptr, 0, false, true);
        std::vector<llama_token> tokens(std::abs(n));
        n = llama_tokenize(v, text.data(), text.size(), tokens.data(), tokens.size(), false, true);
        if (n < 0) throw std::runtime_error("tokenization failed");
        tokens.resize(n); return tokens;
    }
    json vocabulary() {
        json texts = json::array();
        for (int i = 0; i < nv; ++i) {
#ifdef POWERINFER_LEGACY
            texts.push_back(hex(llama_token_get_text(model, i)));
#else
            texts.push_back(hex(llama_vocab_get_text(vocab, i)));
#endif
        }
        return {{"n_vocab",nv},{"bos",bos},{"tokens_hex",texts}};
    }
    void clear() {
#ifdef POWERINFER_LEGACY
        llama_kv_cache_clear(ctx);
#else
        llama_synchronize(ctx);
        llama_memory_clear(llama_get_memory(ctx), true);
#endif
    }
    void decode(const llama_token * tokens, int n, int pos, bool all = false) {
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            if (tokens[i] < 0 || tokens[i] >= nv) throw std::runtime_error("token out of vocabulary");
            batch.token[i] = tokens[i]; batch.pos[i] = pos+i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
            batch.logits[i] = all || i == n-1;
        }
        int rc = llama_decode(ctx, batch);
        if (rc != 0) throw std::runtime_error("llama_decode returned " + std::to_string(rc));
#ifndef POWERINFER_LEGACY
        llama_synchronize(ctx);
#endif
        last_n = n;
    }
    float * logits(int i = -1) { return llama_get_logits_ith(ctx, i < 0 ? last_n-1 : i); }
    llama_token greedy() {
        const float * p = logits();
        int best = 0;
        for (int i = 0; i < nv; ++i) {
            if (!std::isfinite(p[i])) throw std::runtime_error("nonfinite logits");
            if (p[i] > p[best]) best = i;
        }
        return best;
    }
};

int main(int argc, char ** argv) {
    llama_log_set([](ggml_log_level level, const char * text, void *) {
#ifndef POWERINFER_LEGACY
        if (level != GGML_LOG_LEVEL_DEBUG) std::fputs(text, stderr);
#else
        (void) level;
        std::fputs(text, stderr);
#endif
    }, nullptr);
    json a = {{"mode","perf"},{"ngl",0},{"ctx",2560},{"batch",512},{"threads",6},
        {"vram",18.0},{"reset_index",false},{"pipo",false},{"config",""},{"round",0},{"gen",128}};
    try {
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--pipo" || key == "--reset-index") {
                a[key == "--pipo" ? "pipo" : "reset_index"] = true; continue;
            }
            if (key.rfind("--",0) != 0 || i+1 == argc) throw std::runtime_error("expected --key value");
            key = key.substr(2); std::string value = argv[++i];
            if (key == "ngl" || key == "ctx" || key == "batch" || key == "threads" || key == "round" || key == "gen") a[key] = std::stoi(value);
            else if (key == "vram") a[key] = std::stod(value);
            else a[key] = value;
        }
        if (a.at("ctx").get<int>() <= 0 || a.at("batch").get<int>() <= 0 || a.at("threads").get<int>() <= 0)
            throw std::runtime_error("context, batch and threads must be positive");
        Engine e(a);
        const std::string mode = a.at("mode");
        if (mode == "vocab") { emit(std::cout,e.vocabulary()); return 0; }
        if (mode == "tokenize") {
            std::string line;
            while (std::getline(std::cin,line)) { auto j=json::parse(line); emit(std::cout,{{"tokens",e.tokenize(j.at("text"))},{"bos",e.bos}}); }
            return 0;
        }
        std::ofstream out(a.at("output").get<std::string>());
        if (!out) throw std::runtime_error("cannot open output file");
        emit(out,{{"type","metadata"},{"arguments",a},{"pid",getpid()},{"n_vocab",e.nv},{"bos",e.bos},{"start_monotonic_ns",now_ns()}});
        std::ifstream in(a.at("input").get<std::string>());
        if (!in) throw std::runtime_error("cannot open input manifest");
        std::string line;
        while (std::getline(in,line)) {
            auto item = json::parse(line);
            auto tokens = item.at("tokens").get<std::vector<llama_token>>();
            if (tokens.empty() || tokens[0] != e.bos) throw std::runtime_error("expected exactly one leading BOS");
            if ((int)tokens.size() + (mode=="perf" ? a.at("gen").get<int>() : 0) > a.at("ctx").get<int>()) throw std::runtime_error("request exceeds context");
            e.clear();
            json result = {{"type",mode},{"system",a.at("system")},{"round",a.at("round")},
                {"id",item.at("id")},{"scenario",item.value("scenario","")},{"split",item.value("split","test")},
                {"n_prompt",tokens.size()},{"status","ok"}};
            const int chunk = a.at("batch");
            const int64_t start = now_ns();
            result["start_monotonic_ns"] = start;
            if (mode == "quality") {
                double nll = 0; int scored = 0;
                for (int pos=0; pos<(int)tokens.size()-1; pos+=chunk) {
                    int n=std::min(chunk,(int)tokens.size()-1-pos);
                    e.decode(tokens.data()+pos,n,pos,true);
                    for (int i=0;i<n;++i) {
                        auto p=e.logits(i); double max=-INFINITY;
                        for(int j=0;j<e.nv;++j) { if(!std::isfinite(p[j])) throw std::runtime_error("nonfinite logits"); max=std::max(max,(double)p[j]); }
                        double sum=0; for(int j=0;j<e.nv;++j) sum+=std::exp(p[j]-max);
                        nll += max+std::log(sum)-p[tokens[pos+i+1]]; ++scored;
                    }
                }
                result["nll_sum"]=nll; result["scored_tokens"]=scored;
            } else {
                for (int pos=0;pos<(int)tokens.size();pos+=chunk) e.decode(tokens.data()+pos,std::min(chunk,(int)tokens.size()-pos),pos);
                result["prefill_s"]=(now_ns()-start)*1e-9;
                if (mode == "logits") {
                    auto cont=item.at("continuation").get<std::vector<llama_token>>();
                    std::string path=a.at("output").get<std::string>()+"."+item.at("id").get<std::string>()+".f32";
                    std::ofstream dump(path,std::ios::binary);
                    if(!dump) throw std::runtime_error("cannot open logits dump");
                    for(size_t i=0;i<=cont.size();++i) {
                        dump.write((const char *)e.logits(),e.nv*sizeof(float));
                        if(i<cont.size()) e.decode(&cont[i],1,tokens.size()+i);
                    }
                    if(!dump) throw std::runtime_error("logits write failed");
                    result["logits_file"]=path; result["shape"]={cont.size()+1,e.nv};
                } else if (mode == "perf") {
                    const int gen=a.at("gen"); if(gen<2) throw std::runtime_error("gen must be >=2");
                    std::vector<llama_token> output; output.reserve(gen);
                    std::vector<int64_t> times; times.reserve(gen);
                    output.push_back(e.greedy()); times.push_back(now_ns());
                    for(int i=1;i<gen;++i) { e.decode(&output.back(),1,tokens.size()+i-1); auto tok=e.greedy(); times.push_back(now_ns()); output.push_back(tok); }
                    const double decode=(times.back()-times.front())*1e-9;
                    result["n_gen"]=gen; result["output_tokens"]=output; result["token_monotonic_ns"]=times;
                    result["ttft_s"]=(times.front()-start)*1e-9; result["request_s"]=(times.back()-start)*1e-9;
                    result["tpot_s"]=decode/(gen-1); result["decode_tps"]=(gen-1)/decode;
                } else throw std::runtime_error("unknown mode " + mode);
            }
            result["end_monotonic_ns"]=now_ns();
            struct rusage ru{}; getrusage(RUSAGE_SELF,&ru); result["rss_peak_kib"]=ru.ru_maxrss;
            emit(out,result);
            std::cerr << "completed " << item.at("id") << " mode=" << mode << '\n';
        }
        emit(out,{{"type","complete"},{"end_monotonic_ns",now_ns()}});
    } catch (const std::exception & ex) { std::cerr << "comparison runner: " << ex.what() << '\n'; return 1; }
    return 0;
}
