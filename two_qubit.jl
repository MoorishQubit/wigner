include("functionals.jl")
using NPZ
using ProgressMeter

BLAS.set_num_threads(1)

function sample(steps, d, r)
    samples = zeros(steps, 3)
    p = Progress(steps, "Calculating r=$(r)")
    Threads.@threads for i = 1:steps
        ψ = rand(HaarKet(d * r))
        ρ = ptrace(ψ, [d, r], 2)
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
    for r = 1:d
        samples = sample(steps, d, r)
        npzwrite("samples_$(r).npy", samples)
    end
end

main()