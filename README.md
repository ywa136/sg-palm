# SG-PALM

`Julia` implementation of the Proximal Alternating Linearized Minimization (PALM) method for Sylvester Graphical (SG) model.

Paper: Y. Wang, A. O. Hero. SG-PALM: a Fast Physically Interpretable Tensor Graphical Model. To appear in ICML'21. 

Required: Julia 1.5
Dependencies: Kronecker, LinearAlgebra, TensorToolbox, SparseArrays, Printf, SpecialMatrices, Distributions, Random

The main algorithm is in sg_palm.jl. One can run simulations from the experiment_synthetic.jl script, which calls several helper functions (that gene>

