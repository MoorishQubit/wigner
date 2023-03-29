include("functionals.jl")
using NPZ
using ProgressMeter

BLAS.set_num_threads(1)

function uniform_sampler(d, r)
    ψ = rand(HaarKet(d * r))
    ptrace(ψ, [d, r], 2)
end

function product_sampler(d, r)
    ψ = rand(HaarKet(isqrt(d)))
    ϕ = rand(HaarKet(isqrt(d)))
    proj(kron(ψ, ϕ))
end


function sample(steps, d, r, state_sampler)
    samples = zeros(steps, 3)
    p = Progress(steps, "Calculating r=$(r), sampler=$(state_sampler)")
    Threads.@threads for i = 1:steps
        ρ = state_sampler(d, r)
        w = wooters_wigner(ρ)
        samples[i, 1] = real(sum(w[w .< 0]))
        samples[i, 2] = real(negativity(ρ, [isqrt(d), isqrt(d)], 1))
        samples[i, 3] = real(log_negativity(ρ, [isqrt(d), isqrt(d)], 1))
        next!(p)
    end
    finish!(p)
    return samples
end

function main()
    steps = 1_000_000
    d = 4
    for (r, sampling_function) in [(1, uniform_sampler), (2, uniform_sampler), (3, uniform_sampler), (4, uniform_sampler), (1, product_sampler)]
        samples = sample(steps, d, r, sampling_function)
        npzwrite("samples_rank=$(r)_dist=$(sampling_function).npy", samples)
    end
end

main()