function [U,output] = cpdnn_nlsb(T,U0,options)
%CPDNN_NLSB Non-negative CPD by bound-constrained nonlinear least squares.
%   [U,output] = cpdnn_nlsb(T,U0) computes the factor matrices U{1}, ...,
%   U{N}, some of which are constrained to be non-negative, belonging to a
%   canonical polyadic decomposition of the N-th order tensor T by
%   minimizing 0.5*frob(T-cpdgen(U))^2. The algorithm is initialized with
%   the factor matrices U0{n}. The structure output returns additional
%   information:
%
%      output.Name  - The name of the selected algorithm.
%      output.<...> - The output of the selected algorithm.
%
%   cpdnn_nlsb(T,U0,options) may be used to set the following options:
%
%      options.Algorithm =   - The desired optimization method.
%      [{@nlsb_gndl}]
%      options.LargeScale    - If true, the Gauss-Newton or Levenberg-
%      = sum(size(T))*R>1e2    Marquardt steps are computed using a
%                              preconditioned conjugate gradient algorithm.
%                              Otherwise, a direct solver is used.
%      options.M =           - The preconditioner to use when
%      [false|'Jacobi'|...     options.LargeScale is true.
%       {'block-Jacobi'}|...
%       'block-SSOR']
%      options.Nonnegative = - The indices of the factor matrices which are
%      1:ndims(T)              constrained to be non-negative.
%      options.<...>         - Parameters passed to the selected method.

%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)
%
%   References:
%   [1] L. Sorber, M. Van Barel, L. De Lathauwer, "Optimization-based
%       algorithms for tensor decompositions: canonical polyadic
%       decomposition, decomposition in rank-(Lr,Lr,1) terms and a new
%       generalization," ESAT-SISTA Internal Report 12-37, Department of
%       Electrical Engineering (ESAT), KU Leuven, Leuven, Belgium, 2012.
%   [2] L. Sorber, M. Van Barel, L. De Lathauwer, "Unconstrained
%       optimization of real functions in complex variables", SIAM J. Opt.,
%       Vol. 22, No. 3, 2012, pp. 879-898.

% Check the options structure.
if nargin < 3, options = struct; end
if ~isfield(options,'Algorithm'), options.Algorithm = @nlsb_gndl; end
if ~isfield(options,'Nonnegative'), options.Nonnegative = 1:ndims(T); end
nlsb = options.Algorithm;

% Force the initialization to be nonnegative.
U0 = cellfun(@(u)abs(real(u))+abs(imag(u))*1i,U0,'UniformOutput',false);

% Set the bound-constraints and call cpd_nls.
ninf = @(i,j)-inf(i,j);
lb = cpd_rnd(size(T),size(U0{1},2),struct('Real',ninf,'Imag',ninf));
ub = cpd_rnd(size(T),size(U0{1},2),struct('Real',@inf,'Imag',@inf));
for n = options.Nonnegative, lb{n} = zeros(size(lb{n})); end
options.Algorithm = @(F,dF,U0,options)nlsb(F,dF,lb,ub,U0,options);
[U,output] = cpd_nls(T,U0,options);
