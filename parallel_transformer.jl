#!/usr/bin/env julia
# serial_parallel_gpt2.jl
# -------------------------------------------------------------
# Compare serial vs. parallel prompt completion with
# Transformers.jl + Dagger.jl (CPU-only, single machine)

using Dagger, Dates, LinearAlgebra
using Flux, StatsBase
using TextEncodeBase, Transformers
using Transformers.HuggingFace

# ──────────────────── housekeeping ───────────────────────────
BLAS.set_num_threads(1)                 # avoid BLAS × Threads oversubscription

# ──────────────────── load tokenizer + GPT-2 once ────────────
@info "Loading GPT-2 weights (one-off) …"
const textenc = hgf"gpt2:tokenizer"
const model = hgf"gpt2:lmheadmodel"

function temp_softmax(logits; temperature=1.2)
    return softmax(logits ./ temperature)
end

function top_k_sample(probs; k=10)
    sorted = sort(probs, rev = true)
    indexes = partialsortperm(probs, 1:k, rev=true)
    index = sample(indexes, ProbabilityWeights(sorted[1:k]), 1)
    return index
end

function generate_text(context=""; max_length=50, temperature=1.2, k=10)
    encoded = encode(textenc, context).token
    ids = encoded.onehots
    ends_id = lookup(textenc.vocab, textenc.endsym)
    
    for i in 1:max_length
        input = (; token = encoded)
        outputs = model(input)
        logits = @view outputs.logit[:, end, 1]
        probs = temp_softmax(logits; temperature=temperature)
        new_id = top_k_sample(probs; k=k)[1]
        push!(ids, new_id)
        new_id == ends_id && break
    end
    
    return join(decode(textenc, encoded))
end

# ─────────────────── serial completion ────────────────────────
function serial_complete(prompts; max_length=50, temperature=1.2, k=10)
    results = String[]
    for prompt in prompts
        text = generate_text(prompt; max_length=max_length, temperature=temperature, k=k)
        push!(results, text)
    end
    return results
end

# ─────────────────── parallel completion with Dagger ─────────
function parallel_complete(prompts; max_length=50, temperature=1.2, k=10)
    thunks = [
        Dagger.@spawn generate_text(prompt; 
                                   max_length=max_length, 
                                   temperature=temperature, 
                                   k=k)
        for prompt in prompts
    ]
    return fetch.(thunks)
end

# ─────────────────── test prompts ─────────────────────────────
base_prompts = [
    "My name is Thomas and my main",
    "The quick brown fox",
    "Once upon a time in a distant land",
    "Technology has revolutionized the way we",
    "In the depths of the ocean",
    "The scientist looked at the data and",
    "Climate change is affecting",
    "Artificial intelligence will transform",
    "The old library contained secrets",
    "Space exploration has revealed"
]

# Create more prompts by repeating the base set
prompts = repeat(base_prompts, 3)  # 30 total prompts
println("Testing with $(length(prompts)) prompts...")

# ─────────────────── benchmark serial vs parallel ────────────
println("\n=== SERIAL GENERATION ===")
t_serial_start = now()
serial_results = serial_complete(prompts; max_length=40)
t_serial_end = now()
serial_time = Millisecond(t_serial_end - t_serial_start).value / 1000

println("\n=== PARALLEL GENERATION ===")
t_parallel_start = now()
parallel_results = parallel_complete(prompts; max_length=40)
t_parallel_end = now()
parallel_time = Millisecond(t_parallel_end - t_parallel_start).value / 1000

speedup = round(serial_time / parallel_time; digits = 2)

# ─────────────────── results ──────────────────────────────────
println("\n=== BENCHMARK RESULTS ===")
@info "Available threads    : $(Threads.nthreads())"
@info "Prompts processed    : $(length(prompts))"
@info "Serial time (s)      : $serial_time"
@info "Parallel time (s)    : $parallel_time" 
@info "Speedup              : $(speedup)x"

# ─────────────────── sample outputs ───────────────────────────
println("\n=== SAMPLE OUTPUTS ===")
println("Prompt: \"$(prompts[1])\"")
println("Serial  : $(serial_results[1])")
println("Parallel: $(parallel_results[1])")

println("\nPrompt: \"$(prompts[2])\"")
println("Serial  : $(serial_results[2])")
println("Parallel: $(parallel_results[2])")

# ─────────────────── verify results match ────────────────────
# Note: Results will be different due to randomness in sampling
# but both should be valid completions
all_valid = true
for i in 1:length(prompts)
    if length(serial_results[i]) < length(prompts[i]) || 
       length(parallel_results[i]) < length(prompts[i])
        all_valid = false
        break
    end
end

println("\n=== VALIDATION ===")
@info "All results valid    : $all_valid"
@info "Results are different due to random sampling - this is expected"