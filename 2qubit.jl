using QuantumInformation, Plots, LinearAlgebra, TensorOperations

function state(r)
    ρ=rand(HilbertSchmidtStates{2, r}(4))
end


state(1)