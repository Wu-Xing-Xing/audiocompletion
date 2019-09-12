function [U,output] = btdLx1_minf(T,U0,L,options)
%BTDLX1_MINF (rank-L(r) x rank-1) BTD by minimizing an objective function.
%   [U,output] = btdLx1_minf(T,U0,L) computes the factor matrices U{1},
%   ..., U{N} belonging to a (rank-L(r) x rank-1) block term decomposition
%   of the N-th order tensor T. Each term in the decomposition is the outer
%   product of a rank-L(r) tensor and a rank-1 tensor. The vector L has
%   length R and defines the ranks L(r). The algorithm is initialized with
%   the factor matrices U0{n}, which must have either sum(L) or R columns.
%   For example, in a rank-(L(r),L(r),1) BTD the first two factor matrices
%   U0{1} and U0{2} should have sum(L) columns while U{3} should have R.
%   The structure output returns additional information:
%
%      output.Name  - The name of the selected algorithm.
%      output.<...> - The output of the selected algorithm.
%
%   btdLx1_minf(T,U0,L,options) may be used to set the following options:
%
%      options.Algorithm =     - The desired optimization method.
%      [{@minf_lbfgsdl}|...
%       @minf_lbfgs|@minf_ncg]
%      options.<...>           - Parameters passed to the selected method,
%                                e.g., options.TolFun and options.TolX.
%                                See also help [options.Algorithm].
%
%   See also btdLx1_nls.

%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)
%
%   References:
%   [1] L. Sorber, M. Van Barel, L. De Lathauwer, "Optimization-based
%       algorithms for tensor decompositions: canonical polyadic
%       decomposition, decomposition in rank-(Lr,Lr,1) terms and a new
%       generalization," SIAM J. Opt., 2013.
%   [2] L. Sorber, M. Van Barel, L. De Lathauwer, "Unconstrained
%       optimization of real functions in complex variables," SIAM J. Opt.,
%       Vol. 22, No. 3, 2012, pp. 879-898.

% Check the tensor T.
N = ndims(T);
if N < 3, error('btdLx1_minf:T','ndims(T) should be >= 3.'); end

% Check the initial factor matrices U0.
U = U0(:).';
R = sum(L);
P = find(cellfun(@(u)size(u,2),U) == R);
Q = find(cellfun(@(u)size(u,2),U) == length(L));
E = full(sparse(sum(bsxfun(@gt,1:R,cumsum(L)'),1)+1,1:R,1));
if N ~= length(P)+length(Q)
    error('btdLx1_minf:U0', ...
          'size(U0{n},2) should equal either sum(L) or length(L).');
end
if any(cellfun('size',U,1) ~= size(T))
    error('btdLx1_minf:U0','size(T,n) should equal size(U0{n},1).');
end

% Check the options structure.
func = @(f)isa(f,'function_handle')&&exist(func2str(f),'file');
if nargin < 4, options = struct; end
if ~isfield(options,'Algorithm')
    funcs = {@minf_lbfgsdl,@minf_lbfgs,@minf_ncg};
    options.Algorithm = funcs{find(cellfun(func,funcs),1)};
end
if ~isfield(options,'TolFun'), options.TolFun = 1e-12; end

% Cache the tensor matricizations.
M = arrayfun(@(n)tens2mat(T,n),1:N,'UniformOutput',false);

% Call the optimization method.
[U,output] = options.Algorithm(@f,@g,U,options);
output.Name = func2str(options.Algorithm);

function fval = f(U)
    for n = Q, U{n} = U{n}*E; end
    D = M{1}-U{1}*kr(U(end:-1:2)).';
    fval = 0.5*sum(D(:)'*D(:));
end

function grad = g(U)
    grad = cell(1,N);
    UHU = zeros(R,R,N);
    for n = Q, U{n} = U{n}*E; end
    for n = 1:N, UHU(:,:,n) = U{n}'*U{n}; end
    for n = 1:N
        G1 = U{n}*conj(prod(UHU(:,:,[1:n-1 n+1:N]),3));
        G2 = M{n}*conj(kr(U([N:-1:n+1 n-1:-1:1])));
        if any(Q == n), G1 = G1*E.'; G2 = G2*E.'; end
        grad{n} = G1-G2;
    end
end

end
