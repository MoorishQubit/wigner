using QuantumInformation
using LinearAlgebra
using Memoize
using Kronecker
import Base.Iterators.partition

@memoize wooters_kernel(x::Int, p::Int) = (I + (-1)^x * sz + (-1)^p * sx + (-1)^(x+p+1) * sy)/2

function wooters_wigner(ρ::Matrix)
    n = Int(log2(size(ρ, 1)))
    
    w = zeros(4^n)
    for i=1:4^n
        bits = digits(i-1, base=2, pad=2n) |> reverse
        kernel = kronecker(Base.splat(wooters_kernel).(partition(bits, 2))...)
        w[i] = real(1/2^n * tr(kernel * ρ))
    end
    return w
end