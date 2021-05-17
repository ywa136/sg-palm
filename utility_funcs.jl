"""
Collections of utility functions for simulation and evaluations

Author: XXXX XXXX
Last modified: 02/09/2021
"""

using SparseArrays
using LinearAlgebra
using SpecialMatrices
using Distributions
using Random


function AR_model(p::Int,ρ::Real)
    """
    Generate sparse inv cov matrix that has auto-regressive structure and
    its corresponding cov matrix
    In
    p::Int: dimension of the cov matrix
    ρ::Real: weight param between 0 and 1
    Out
    cov::SparseMatrixCSC
    invcov::SparseMatrixCSC
    """
    cov = ρ.^(abs.(Toeplitz(collect(-(p-1):(p-1)))))

    invcov = sparse(Diagonal(repeat([1+ρ^2],p)))
    for i=1:(p-1)
        invcov[i, i+1] = -ρ
        invcov[i+1, i] = -ρ
    end

    return invcov, cov
end


function Unif_model(p::Int, dens::Real; min_eigen::Real=1e-3)
    """
    Generate sparse inverse covariance matrix with uniformly distributed weights
    In
    p: the desired dimension of the (inv)covariance matrix
    dens: percentage of non-zeros of the final inverse covariance
    min_eigen: the minimum bound for the eigenvalues of the desired (inv)covariance matrix
    Out
    p x p (inv)covariance matrix with dens non-zeros
    """
    U = sprand(p,p,dens)
    for i=1:p 
        for j=1:p
            if U[i,j] != 0
                U[i,j] = rand([-1.0,1.0],1)[1]
            end
        end
    end
    U .= transpose(U)*U
    Udiag = Diagonal(U) .+ Diagonal(ones(p))
    Uoff = max.(min.(U .- Diagonal(U), 1.0), -1.0)
    inv_cov = Uoff .+ Udiag
    eigs, _ = eigen(collect(inv_cov))
    inv_cov .= inv_cov .+ Diagonal(repeat([max(-1.2*minimum(eigs),min_eigen)],p))

    return inv_cov
end


function SB_model(p::Int,ρ::Real;num_subgraph::Int=4,tol::Real=1e-5)
    """
    Generate sparse inv cov matrix that has star-block structure and
    its corresponding cov matrix
    In
    p::Int: dimension of the cov matrix
    ρ::Real: weight param between 0 and 1
    num_subgraph::Int: number of star structures
    tol:Real: thresholding value for zeros
    Out
    cov::SparseMatrixCSC
    invcov::SparseMatrixCSC
    """
    p_subgraph = Int(floor(p/num_subgraph))
    uneq_graph = (p != p_subgraph*num_subgraph)

    list_subgraph = []
    list_invsubgraph = []

    for i=1:num_subgraph
        if i==num_subgraph && uneq_graph
            p_subgraph = p - p_subgraph*(i-1)
        end
        central_node = Int(rand(Tuple(collect(1:p_subgraph))))
        subgraph = repeat([ρ^2],p_subgraph,p_subgraph)
        subgraph[:,central_node] .= ρ
        subgraph[central_node,:] .= ρ
        subgraph = subgraph - Diagonal(subgraph) + Diagonal(ones(p_subgraph))
        invsubgraph = inv(subgraph)
        push!(list_subgraph,sparse(subgraph))
        push!(list_invsubgraph,sparse(invsubgraph)) 
    end

    cov = SparseArrays.blockdiag(list_subgraph...)
    cov[abs.(cov).<tol] .= 0
    invcov = SparseArrays.blockdiag(list_invsubgraph...)
    invcov[abs.(invcov).<tol] .= 0

    return invcov, cov
end


function metric_edge(true_edge,est_edge)
    """
    Computes FNR, FPR, MCC, etc
    In
    - true_edge::AbstractArrary: true sparse precision matrix
    - est_edge::AbstractArray: estimated sparse precision matrix
    Out
    - metric_res::Array: FP, FN, TP, TN, FPR, FNR, Precision, Recall, MCC
    """
    p = size(true_edge,1)
    nz_pos = (true_edge .!= 0)
    z_pos = (true_edge .== 0)

    FP = (sum((est_edge[z_pos] .!= 0)))/2
    FN = (sum((est_edge[nz_pos] .== 0)))/2
    TP = (sum((est_edge[nz_pos] .!= 0)) - p)/2
    TN = (sum((est_edge[z_pos] .== 0)))/2

    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    MCC = (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    metric_res = [FP, FN, TP, TN, FPR, FNR, Precision, Recall, MCC]

    return metric_res
end


function mcc_iter(PsiHs,PsiT)
    """
    Computes MCC for a sequence of estimated precision factors
    In
    - PsiHs: a sequence of estimated sparse precision factors
    - PsiT: K true sparse precision factors
    Out
    - MCC: a sequene of MCCs (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)),
    where TP,FP,TN,FN are aggregated across all K factors 
    """
    MCC = []
    for i=1:length(PsiHs)
        PsiH = PsiHs[i]
        sum_metric = sum(metric_edge.(PsiT,PsiH))
        FP = sum_metric[1]
        FN = sum_metric[2]
        TP = sum_metric[3]
        TN = sum_metric[4]
        push!(MCC, (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    end
    return MCC
end


function kroneckersum_alt(A::AbstractMatrix, B::AbstractMatrix)
    """
    Alternate Kronecker sum defined as I ⊗ A + B ⊗ I
    """
    return kroneckersum(B,A)
end


function kroneckersum_list(mat_list::AbstractVector{<:Any})
    """
    Kronecker sum for a list of matrices using package Kronecker.jl
    In
    - mat_list::AbstractVector{<:Any}: a list of matrices 
    Out
    - mat::KroneckerSum: mat_list[1] ⊕ ... ⊕ mat_list[end] 
    """
    K = length(mat_list)
    mat = copy(mat_list[1])
    if K == 1
        return mat
    else
        for j=2:K
            mat = kroneckersum_alt(mat,mat_list[j])
        end
    end
    return mat
end


function kronsum(A::AbstractMatrix, B::AbstractMatrix)
    """
    Alternate Kronecker sum defined as I ⊗ A + B ⊗ I
    """
    m = size(A,1)
    n = size(B,1)
    return kron(I(n),A) + kron(B,I(m))
end


function kronsum_list(mat_list::AbstractVector{<:Any})
    """
    Kronecker sum for a list of matrices using package Kronecker.jl
    In
    - mat_list::AbstractVector{<:Any}: a list of matrices 
    Out
    - mat::KroneckerSum: mat_list[1] ⊕ ... ⊕ mat_list[end] 
    """
    K = length(mat_list)
    mat = copy(mat_list[1])
    if K == 1
        return mat
    else
        for j=2:K
            mat = kronsum(mat,mat_list[j])
        end
    end
    return mat
end


function kronecker_list(mat_list::AbstractVector{<:Any})
    """
    Kronecker product for a list of matrices using package Kronecker.jl
    In
    - mat_list::AbstractVector{<:Any}: a list of matrices 
    Out
    - mat::KroneckerProduct: mat_list[1] ⊗ ... ⊗ mat_list[end] 
    """
    K = length(mat_list)
    mat = copy(mat_list[1])
    if K == 1
        return mat
    else
        for j=2:K
            mat = kronecker(mat,mat_list[j])
        end
    end
    return mat
end


function kron_list(mat_list::AbstractVector{<:Any})
    K = length(mat_list)
    mat = deepcopy(mat_list[1])
    if K == 1
        return mat
    else
        for j=2:K
            mat = kron(mat,mat_list[j])
        end
    end
    return mat    
end


function mul_alt!(C::StridedMatrix, X::StridedMatrix, A::SparseMatrixCSC)
    mX, nX = size(X)
    nX == A.m || throw(DimensionMismatch())
    fill!(C, zero(eltype(C)))
    rowval = A.rowval
    nzval = A.nzval
    @inbounds for  col = 1:A.n, k=A.colptr[col]:(A.colptr[col+1]-1)
        ki=rowval[k]
        kv=nzval[k]
        for multivec_row=1:mX
            C[multivec_row, col] += X[multivec_row, ki] * kv
        end
    end
    C
end


import Base.*
function *(B::StridedMatrix, A::SparseMatrixCSC)
    mB, nB = size(B)
    nB == A.m || throw(DimensionMismatch())
    C = zeros(mB,A.n)
    rowval = A.rowval
    nzval = A.nzval
    @inbounds for  col = 1:A.n, k=A.colptr[col]:(A.colptr[col+1]-1)
        ki=rowval[k]
        kv=nzval[k]
        for multivec_row=1:mB
            C[multivec_row, col] += B[multivec_row, ki] * kv
        end
    end
    C
end


function offdiag(X::AbstractMatrix)
    """
    Extract the off-diagonal matrix of X
    """
    return X - Diagonal(X)
end


function offdiag!(X::AbstractMatrix)
    """
    In placce version of offdiag
    """
    X .= X .- Diagonal(X)
    return X
end


function gen_sylvester_data(N::Int,Ψ::AbstractVector{<:Any})
    """
    Generate multivariate data satisfying the Sylvester equation:
        X ×_1 Ψ_1 + ... + X ×_K Ψ_K = N(0,I)
    """
    K = length(Ψ)
    d = size.(Ψ,1)
    p = prod(d)
    U = []; Λ = []
    for k=1:K
        Λk, Uk = eigen(collect(Ψ[k])) 
        push!(Λ,Diagonal(Λk))
        push!(U,Uk)
    end
    eigsSigmaSqrt = inv(Diagonal(kroneckersum_list(Λ)))
    reverse!(U)
    eigvecsSigmaSqrt = kronecker_list(U)

    X = zeros(N,p)
    z = zeros(p)
    xtilde = zeros(p)
    x = zeros(p)
    for i=1:N
        randn!(z)
        mul!(xtilde,eigsSigmaSqrt,z)
        mul!(x,eigvecsSigmaSqrt,xtilde)
        X[i,:] .= x
    end

    return X
end


function ksum_gauss_loglik(Ψ::AbstractVector{<:AbstractArray}, X::AbstractArray{<:Real}, 
    X_kGram::AbstractVector{<:AbstractArray})
    """
    Compute Gaussian log-likelihood function with KroneckerSum-structured precision matrix
    *** This currently only works for K=3 ***
    """
    d = size.(Ψ,1)
    K = length(X_kGram)
    N = size(X,2)
    # logdet term
    Λ = [zeros(d[k]) for k=1:K]
    for k=1:K
        Λk, _ = eigen(collect(Ψ[k])) # assuming Ψ is converted to dense
        copyto!(Λ[k], Λk)
    end
    # logDet = logdet(Diagonal(kroneckersum_list(Λ))) # Kronecker sum stracutred
    logDet = sum(reshape([log(sum([x,y,z])) for x=Λ[1], y=Λ[2], z=Λ[3]], prod(d))) ### Only works for K=3 ###
    # trace term
    trTerm = sum([tr(Ψ[k]*X_kGram[k]) for k=1:K])
    # logdet - trace 
    return logDet - trTerm
end


function sylvester_gauss_loglik(Ψ::AbstractVector{<:AbstractArray}, μ::AbstractArray{<:Real},
    X::AbstractArray{<:Real})
    """
    Compute Gaussian log-likelihood function with Sylvester-structured precision matrix
    *** This currently only works for K=3 ***
    """
    X .-= μ
    d = size.(Ψ,1)
    K = length(Ψ)
    N = size(X,2)
    # Compute mode-k Gram matrices
    X_kGram = [zeros(d[k],d[k]) for k=1:K]
    Xk = [zeros(d[k],Int(prod(d)/d[k])) for k=1:K]
    for k=1:K
        for i=1:N
            copy!(Xk[k], tenmat(reshape(view(X,:,i),Tuple(d)),k))
            mul!(X_kGram[k], Xk[k], copy(transpose(Xk[k])), 1.0/N, 1.0)
        end
    end
    # logdet term
    Λ = [zeros(d[k]) for k=1:K]
    for k=1:K
        Λk, _ = eigen(collect(Ψ[k])) # assuming Ψ is converted to dense
        copyto!(Λ[k], Λk)
    end
    # logDet = logdet(Diagonal(kroneckersum_list(Λ))^2) # SyGlasso strucutred
    logDet = sum(reshape([log(sum([x,y,z])^2) for x=Λ[1], y=Λ[2], z=Λ[3]], prod(d))) ### Only works for K=3 ###
    # trace term
    crossTerm = 0
    sqrTerm = sum([tr(Ψ[k]^2*X_kGram[k]) for k=1:K])
    @inbounds for k=1:(K-1)
        crossTermMat = zeros(d[k],d[k]) # pre-allocate
        Xk = zeros(d[k],Int(prod(d)/d[k])) # pre-allocate
        XkT = zeros(Int(prod(d)/d[k]),d[k]) # pre-allocate
        kp_k = spzeros(Int(prod(d)/d[k]),Int(prod(d)/d[k])) # pre-allocate
        @inbounds for l=(k+1):K
            sumKGramTilde = zeros(size(Ψ[k])) # pre-allocate
            Ψ_k = [(i==l) ? Ψ[l] : Diagonal(ones(d[i])) for i=1:K]
            deleteat!(Ψ_k,k)
            reverse!(Ψ_k)
            copy!(kp_k, (length(Ψ_k)==1) ? Ψ_k[1] : kron(Ψ_k...)) # handles case K=2
            @inbounds for i=1:N
                copy!(Xk, tenmat(reshape(view(X,:,i),Tuple(d)),k))
                mul!(XkT, kp_k, copy(transpose(Xk)))
                mul!(sumKGramTilde, Xk, XkT, 1.0/N, 1.0)
            end
            mul!(crossTermMat,Ψ[k],sumKGramTilde)
            crossTerm += 2*tr(crossTermMat)
        end
    end
    # logdet - trace 
    return logDet - sqrTerm - crossTerm
end


function sylvester_pseudo_loglik(Ψ::AbstractVector{<:AbstractArray}, μ::AbstractArray{<:Real},
    X::AbstractArray{<:Real})
    """
    Compute the SyGlasso cost function
    """
    X .-= μ
    d = size.(Ψ,1)
    K = length(Ψ)
    N = size(X,2)
    # Compute mode-k Gram matrices
    X_kGram = [zeros(d[k],d[k]) for k=1:K]
    Xk = [zeros(d[k],Int(prod(d)/d[k])) for k=1:K]
    for k=1:K
        for i=1:N
            copy!(Xk[k], tenmat(reshape(view(X,:,i),Tuple(d)),k))
            mul!(X_kGram[k], Xk[k], copy(transpose(Xk[k])), 1.0/N, 1.0)
        end
    end
    # logdet term
    logDet = logdet(Diagonal(kroneckersum_list([Diagonal(Ψ[k]) for k=1:K]))^2)
    # trace term
    crossTerm = 0
    sqrTerm = sum([tr(Ψ[k]^2*X_kGram[k]) for k=1:K])
    @inbounds for k=1:(K-1)
        crossTermMat = zeros(d[k],d[k]) # pre-allocate
        Xk = zeros(d[k],Int(prod(d)/d[k])) # pre-allocate
        XkT = zeros(Int(prod(d)/d[k]),d[k]) # pre-allocate
        kp_k = spzeros(Int(prod(d)/d[k]),Int(prod(d)/d[k])) # pre-allocate
        @inbounds for l=(k+1):K
            sumKGramTilde = zeros(size(Ψ[k])) # pre-allocate
            Ψ_k = [(i==l) ? Ψ[l] : Diagonal(ones(d[i])) for i=1:K]
            deleteat!(Ψ_k,k)
            reverse!(Ψ_k)
            ### TODO: Use LuxurySparse ###
            ### TODO: Use LinearMap!!! ###
            copy!(kp_k, (length(Ψ_k)==1) ? Ψ_k[1] : kron(Ψ_k...)) # handles case K=2
            @inbounds for i=1:N
                copy!(Xk, tenmat(reshape(view(X,:,i),Tuple(d)),k))
                mul!(XkT, kp_k, copy(transpose(Xk)))
                mul!(sumKGramTilde, Xk, XkT, 1.0/N, 1.0)
            end
            mul!(crossTermMat,Ψ[k],sumKGramTilde)
            crossTerm += 2*tr(crossTermMat)
        end
    end
    # logdet - trace 
    return logDet - sqrTerm - crossTerm 
end
