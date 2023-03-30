using QuantumInformation
using LinearAlgebra
using Memoize
using Kronecker
import Base.Iterators.partition
using HCubature
using ProgressMeter


@memoize wooters_kernel(x::Int, p::Int) =
    (I + (-1)^x * sz + (-1)^p * sx + (-1)^(x + p + 1) * sy) / 2
@memoize tilma_kernel(θ::Float64, φ::Float64) =
    (I - sqrt(3) * (cos(2 * θ) * sz + sin(2 * θ) * (cos(2 * φ) * sx + sin(2 * φ) * sy))) / 2

function wooters_wigner(ρ::Matrix)
    n = Int(log2(size(ρ, 1)))

    w = zeros(4^n)
    for i = 1:4^n
        bits = digits(i - 1, base = 2, pad = 2n) |> reverse
        kernel = kronecker(Base.splat(wooters_kernel).(partition(bits, 2))...)
        w[i] = real(1 / 2^n * tr(kernel * ρ))
    end
    return w
end

function tilma_wigner(ρ::Matrix, θ1::Float64, φ1::Float64, θ2::Float64, φ2::Float64)
    kernel = kronecker(Base.splat(tilma_kernel).(partition([θ1, φ1, θ2, φ2], 2))...)
    w = real(tr(kernel * ρ))
    return w
end

function concurrence_fix(ρ::AbstractMatrix{<:Number})
    if size(ρ, 1) != 4
        throw(ArgumentError("Concurrence only properly defined for two qubit systems"))
    end

    σ = kron(sy, sy) * conj.(ρ) * kron(sy, sy)
    λ = sort(real(eigvals(ρ * σ)), rev = true)
    λ[λ.<0] .= 0
    λ = sqrt.(λ)
    max(0, λ[1] - λ[2] - λ[3] - λ[4])
end
