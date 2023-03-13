using LinearAlgebra
using QuadGK
using ToeplitzMatrices
using MultiQuad
using Plots; pythonplot()
using Printf
using LaTeXStrings
using HCubature


function omega(l,g,fi)
    w=sqrt((g*l*sin(fi))^2+((1+l*cos(fi))^2))/2.0
end

function mag(l,g,T)
    sz=quadgk(fi -> -(1.0+l*cos(fi))*tanh(omega(l,g,fi)/T)/(2.0*pi*omega(l,g,fi)),0,pi)[1]
end

function G(l,n,T,g)
    g=quadgk(fi->(tanh(omega(l,g,fi)/T)/(2.0*pi*omega(l,g,fi)))*(cos(fi*n)*(1.0+l*cos(fi))-g*l*sin(n*fi)*sin(fi)),0,pi)[1]
end

function zz(l,n,T,g)
    z=mag(l,g,T)^2 - G(l,n,T,g)*G(l,-n,T,g)
end

function xx(l,r,T,g)
    x=det(Toeplitz([G(l,r-2,T,g) for r = 1:r],[G(l,-r,T,g) for r = 1:r]))
end

function yy(l,n,T,g)
    y=det(Toeplitz([G(l,n,T,g) for n = 1:n],[G(l,-n+2,T,g) for n=1:n]))
end

function concurrence(l,n,T,g)
    x=(xx(l,n,T,g)+yy(l,n,T,g))/4
    up=1/4 + mag(l,g,T)/2.0 + zz(l,n,T,g)/4.0
    um=1/4 - mag(l,g,T)/2.0 + zz(l,n,T,g)/4.0
    y=(xx(l,n,T,g)-yy(l,n,T,g))/4
    w=(1.0-zz(l,n,T,g))/4.0
    c=2.0*max(0.0,abs(x)-sqrt(up*um),abs(y)-w)
end

function wigxy(t1,t2,f1,f2,l,n,T,g)
    h=(1.0-mag(l,g,T)*sqrt(3.0)*(cos(2.0*t1)+cos(2*t2))
       +3.0*sin(2*t1)*sin(2*t2)*cos(2.0*f1)*cos(2.0*f2)*xx(l,n,T,g)
       +3.0*sin(2*t1)*sin(2*t2)*sin(2.0*f1)*sin(2.0*f2)*yy(l,n,T,g)
       +3.0*zz(l,n,T,g)*cos(2.0*t1)*cos(2.0*t2))/4.0
    abs(h)-h
end

function negativity(l,n,T,g)
   intg=dblquad((t,f) -> wigxy(t,t,f,f,l,n,T,g)*(sin(2*t)/pi),0,pi,0,2*pi)[1]
end


n=1
T=1e-7
g=0.5
lambda = range(0, 2, length=100)
y = concurrence.(lambda,n,T,g)
p=plot(lambda,y,label=L"γ")
xlabel!(L"λ")
ylabel!(L"C_{ρ}")
savefig(p, "./figs/concurrence_XY.pdf")
p