% TENSORLAB
% Version 1.01, 2013-04-29
%
% BLOCK TERM DECOMPOSITION
% Algorithms
%    btdLx1_als   - (rank-L(r) x rank-1) BTD by alternating least squares.
%    btdLx1_minf  - (rank-L(r) x rank-1) BTD by minimizing an objective function.
%    btdLx1_nls   - (rank-L(r) x rank-1) BTD by nonlinear least squares.
%    btdLx1nn_nls - Non-negative (rank-L(r) x rank-1) BTD by NLS.
% Initialization
%    btdLx1_rnd   - Pseudorandom initialization for (rank-L(r) x rank-1) BTD.
% Utilities
%    btdLx1gen    - Generate full tensor given a (rank-L(r) x rank-1) BTD.
%
% CANONICAL POLYADIC DECOMPOSITION
% Algorithms
%    cpd          - Canonical polyadic decomposition.
%    cpd_als      - CPD by alternating least squares.
%    cpd_minf     - CPD by unconstrained nonlinear optimization.
%    cpd_nls      - CPD by nonlinear least squares.
%    cpd3_sd      - CPD by simultaneous diagonalization.
%    cpd3_sgsd    - CPD by simultaneous generalized Schur decomposition.
%    cpdnn_nlsb   - Non-negative CPD by bound-constrained nonlinear least squares.
%    cpds_minf    - Structured/symmetric CPD by unconstrained nonlinear optimization.
%    cpds_nls     - Structured/symmetric CPD by nonlinear least squares.
% Initialization
%    cpd_gevd     - CPD by a generalized eigenvalue decomposition.
%    cpd_rnd      - Pseudorandom initialization for CPD.
% Line and plane search
%    cpd_aels     - CPD approximate enhanced line search.
%    cpd_els      - CPD exact line search.
%    cpd_eps      - CPD exact plane search.
%    cpd_lsb      - CPD line search by Bro.
% Utilities
%    cpderr       - Errors between factor matrices in a CPD.
%    cpdgen       - Generate full tensor given a polyadic decomposition.
%    rankest      - Estimate rank.
%
% LOW MULTILINEAR RANK APPROXIMATION
% Algorithms
%    lmlra        - Low multilinear rank approximation.
%    lmlra_hooi   - LMLRA by higher-order orthogonal iteration.
%    lmlra3_dgn   - LMLRA by a differential-geometric Newton method.
%    lmlra3_rtr   - LMLRA by a Riemannian trust region method.
% Initialization
%    mlsvd        - (Truncated) multilinear singular value decomposition.
%    lmlra_rnd    - Pseudorandom initialization for LMLRA.
% Utilities
%    lmlraerr     - Errors between factor matrices in a LMLRA.
%    lmlragen     - Generate full tensor given a core tensor and factor matrices.
%    mlrank       - Multilinear rank.
%    mlrankest    - Estimate multilinear rank.
%
% COMPLEX OPTIMIZATION
% Nonlinear least squares
%    nls_gncgs    - Nonlinear least squares by Gauss-Newton with CG-Steihaug.
%    nls_gndl     - Nonlinear least squares by Gauss-Newton with dogleg trust region.
%    nls_lm       - Nonlinear least squares by Levenberg-Marquardt.
%    nlsb_gndl    - Bound-constrained NLS by projected Gauss-Newton dogleg TR.
% Unconstrained nonlinear optimization
%    minf_lbfgs   - Minimize a function by L-BFGS with line search.
%    minf_lbfgsdl - Minimize a function by L-BFGS with dogleg trust region.
%    minf_ncg     - Minimize a function by nonlinear conjugate gradient.
% Utilities
%    deriv        - Approximate gradient and Jacobian.
%    ls_mt        - Strong Wolfe line search by More-Thuente.
%    mpcg         - Modified preconditioned conjugate gradients method.
%
% UTILITIES
% Clustering
%    gap          - Optimal clustering based on the gap statistic.
%    kmeans       - Cluster multivariate data using the k-means++ algorithm.
% Polynomials
%    polymin      - Minimize a polynomial.
%    polymin2     - Minimize bivariate and real polyanalytic polynomials.
%    polyval2     - Evaluate bivariate and univariate polyanalytic polynomials.
%    polysol2     - Solve a system of two bivariate polynomials.
%    ratmin       - Minimize a rational function.
%    ratmin2      - Minimize bivariate and real polyanalytic rational functions.
% Statistics
%    cum4         - Fourth-order cumulant tensor.
%    scov         - Shifted covariance matrices.
% Tensors
%    dotk         - Dot product in K-fold precision.
%    frob         - Frobenius norm.
%    kr           - Khatri-Rao product.
%    kron         - Kronecker product.
%    mat2tens     - Tensorize a matrix.
%    noisy        - Generate a noisy version of a given array.
%    sumk         - Summation in K-fold precision.
%    tens2mat     - Matricize a tensor.
%    tens2vec     - Vectorize a tensor.
%    tmprod       - Mode-n tensor-matrix product.
%    vec2tens     - Tensorize a vector.
% Visualization
%    slice3       - Visualize a third-order tensor with slices.
%    spy3         - Visualize a third-order tensor's sparsity pattern.
%    surf3        - Visualize a third-order tensor with surfaces.
%    voxel3       - Visualize a third-order tensor with voxels.
