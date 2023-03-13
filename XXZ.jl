using Calculus
using QuadGK

#Ground state energy
function e(d)
    if d<=-1
        e=d/4
    elseif d<1 && d>-1
        e=d/4 - sin(acos(d))/(2*pi) *quadgk(w -> (sinh((1-acos(d)/pi)*w)/(sinh(w)*cosh((acos(d)/pi) *w))) ,-Inf,Inf)[1]
    elseif d==1
        e=1/4 -log(2)
    elseif d>1
        e=cosh(acosh(d))/4-sinh(acosh(d))/2.0 * quadgk(k->exp(-acosh(d)*abs(k))/cosh(acosh(d)*k),-Inf,Inf)[1]
    end
end

# Computing ⟨σ^z_i σ^z_j⟩
function zz(d)
    z=4*derivative(e,d)
end

# Computing ⟨σ^x_i σ^x_j⟩
function  xx(d)
    x=2*(e(d)-d*derivative(e,d))
end

e(1)