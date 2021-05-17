"""
Run SG-PALM with simulated data

Author: XXXX XXXX
Last modified: 02/09/2021
"""

using Kronecker
using TensorToolbox
using LinearAlgebra
using Plots # for plotting
# using MIRT: jim # for plotting
using BenchmarkTools # for benchmarking code efficiency
using SparseArrays
# using Profile # for analyzing code efficiency
include("sg_palm.jl") # the main PALM algorithm function

####### Only include if have Intel MKL installed #######
using MKL
using MKLSparse
BLAS.set_num_threads(16)
########################################################


######################### Simulation Data ###############################
K = 3
N = 1000
dims = [32,32,32]
dens = 0.05
PsiT = [Unif_model(dims[k],dens) for k=1:K]
cond.([collect(PsiT[k]) for k=1:K]) # compute condition number for each factor
density = X -> (sum(count.(i->(i!=0),X)) - dims[1])/dims[1]^2
density.(PsiT) # compute fraction of non-zeros in Psi's
X = gen_sylvester_data(N,PsiT) # N × prod(dims)
Xt = copy(transpose(X)) # column-major matrix for faster computation: prod(dims) × N
#########################################################


#########################Prepare Inputs to PALM#################################
## compute each mode-k gram matrix
X_kGram = [zeros(dims[k],dims[k]) for k=1:K]
Xk = [zeros(dims[k],Int(prod(dims)/dims[k])) for k=1:K]
for k=1:K
    for i=1:N
        copy!(Xk[k], tenmat(reshape(X[i,:],Tuple(dims)),k))
        X_kGram[k] .+= Xk[k]*copy(transpose(Xk[k]))
    end
    rmul!(X_kGram[k],1.0/N)
end
#########################################################################################


################################ Run PALM ###########################################################
## regularization parameters 
lambda = [5.5*sqrt(dims[k]*log(prod(dims))/N) for k=1:K] 
a = 20 # for SCAD and MCP only

## initial iterates
Psi0 = [sparse(eye(dims[k])) for k=1:K]

## user-defined function to evaluate at each iterates
# fun = (iter,Psi) -> nrmse(Psi,PsiT) # NRMSE 
# fun = (iter,Psi) -> [time(), cost(Psi, lambda, Xt, X_kGram)] # COST + TIME
fun = (iter,Psi) -> [cost(Psi, lambda, a, "L1", Xt, X_kGram), time(), deepcopy(Psi)] # COST + TIME + Psi
# fun = (iter,Psi) -> cost(Psi, lambda, a, "L1", Xt, X_kGram) # COST 
# fun = (iter,Psi) -> 0 # NONE

## max number of iteration
niter = 50

@time PsiH, out = syglasso_palm(Xt, X_kGram, lambda, Psi0, regtype="L1", a=a, 
                            niter=niter, η0=0.01, c=0.1, lsrule="constant", ϵ=1e-6, fun=fun);

# performance profiling
# @profile syglasso_palm(Xt, X_kGram, lambda, Psi0, niter=niter, η0=0.01, c=0.1, lsrule="bb", fun=fun)
# Profile.print()
##################################################################################################


##################################### Plot Convergence ##############################################
## prepare quantities to plot
mylog = x -> log(max(x,1e-1)) # avoid over/underflow
est_cost_palm = out[end][1]
cost_palm = [out[t][1] for t=1:length(out)]
cost_diff_palm = cost_palm .- est_cost_palm

## cost - final cost vs. iteration
plot(2:(length(out)-1), mylog.(cost_diff_palm[2:(length(out)-1)]), label="PALM Cost",
            color=:blue)
scatter!(2:(length(out)-1), mylog.(cost_diff_palm[2:(length(out)-1)]), legend=false,
            color=:blue)
xlabel!("Iteration")
ylabel!("Cost")

## cost - final cost vs. time
times_palm = [out[t][2]-out[1][2] for t=1:length(out)]
plot(times_palm[1:(end-1)], mylog.(cost_diff_palm[1:(end-1)]), label="PALM Cost")
xlabel!("Time in Seconds")
ylabel!("Cost")

## graph recovery vs. time
times_palm = [out[t][2]-out[1][2] for t=1:length(out)]
PsiHs = [out[i][end] for i=1:length(out)]
mcc_palm = mcc_iter(PsiHs,PsiT)
plot(times_palm[2:end], mcc_palm[2:end], label="MCC")
xlabel!("Time in Seconds")
ylabel!("MCC")

## NRMSE vs. time
times_palm = [out[t][2]-out[1][2] for t=1:length(out)]
PsiHs = [out[i][end] for i=1:length(out)]
nrmse = (xhat,xtrue) -> 
        mean([norm(offdiag(xhat[k])-offdiag(xtrue[k]))/norm(offdiag(xtrue[k])) for k=1:K]) # avg. NRMSE for offdiag
nrmse_palm = [nrmse(PsiH,PsiT) for PsiH in PsiHs]
plot(times_palm[2:end], nrmse_palm[2:end], label="PALM NRMSE")
xlabel!("Time in Seconds")
ylabel!("NRMSE")

## NRMSE vs. iteration 
plot(1:length(out), nrmse_palm, yaxis=:log, label="PALM NRMSE")
xlabel!("Iteration")
ylabel!("NRMSE")

## sparsity patterns of the estimates vs. truth
plot(jim(PsiH[2], color=:viridis, title="Psi hat"), jim(PsiT[2], color=:viridis, title="Psi true"))
#######################################################################################################