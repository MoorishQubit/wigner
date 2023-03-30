using QuantumInformation, Plots, LinearAlgebra, TensorOperations, HCubature, MultiQuad


function state(r) # randon 2 qubit state with fixed rank r
    h = rand(HaarKet{2}(4 * r))
    ρ = ptrace(h, [4, r], 2)
    return ρ
end

function wootters(r, x1, p1, x2, p2) # discrete Wigner function for 2 qubits: Wootters construction
    kernel1 =
        (1 / 2) *
        (Matrix(I, 2, 2) + ((-1)^x1) * sz + ((-1)^p1) * sx + ((-1)^(x1 + p1)) * sy)
    kernel2 =
        (1 / 2) *
        (Matrix(I, 2, 2) + ((-1)^x2) * sz + ((-1)^p2) * sx + ((-1)^(x2 + p2)) * sy)
    w = (1 / 4) * tr((kernel1 ⊗ kernel2) * state(r))
    return real(w)
end

function tilma(r, t1, f1, t2, f2) # Wigner function for arbitrary quantum systems: Tilma Construction
    kernel1 =
        (1 / 2) * (
            Matrix(I, 2, 2) -
            sqrt(3) *
            (cos(2 * t1) * sz + sin(2 * t1) * (cos(2 * f1) * sx + sin(2 * f1) * sy))
        )
    kernel2 =
        (1 / 2) * (
            Matrix(I, 2, 2) -
            sqrt(3) *
            (cos(2 * t2) * sz + sin(2 * t2) * (cos(2 * f2) * sx + sin(2 * f2) * sy))
        )
    w = tr((kernel1 ⊗ kernel2) * state(r))
    return real(w)
end

function negativity_wootters(r) #Negativity of the Wigner function under Wootters construction
    s = []
    for x1 in [0, 1]
        for p1 in [0, 1]
            for x2 in [0, 1]
                for p2 in [0, 1]
                    push!(s, wootters(r, x1, p1, x2, p2))
                end
            end
        end
    end
    return sum(s)
end

negativity_wootters(4)

function negativity_tilma(r) #Negativity of the Wigner function under Tilma construction
    #intg=dblquad((t,f) -> tilma(r,t,f,t,f)*(sin(2*t)/pi),0,pi/2,0,2*pi)[1]
    intg = hcubature(
        x -> tilma(r, x[1], x[2], x[1], x[2]) * (sin(2 * x[1]) / pi),
        (0, 0),
        (pi / 2, 2 * pi),
    )[1]
    return intg
end




# c=Any[]; neg_w=Any[]; neg_t=Any[]; neg=Any[]
# for x in 1:10000
#     push!(c,concurrence(state(1)))
#     push!(neg,negativity(state(1),[2,2],2))
# end

# scatter(neg, c, label="rank 1")
# ylabel!("Concurrence")
# xlabel!("Negativity Entanglement")
