function [U,output] = btdLx1_nls(T,U0,L,options)
%BTDLX1_NLS (rank-L(r) x rank-1) BTD by nonlinear least squares.
%   [U,output] = btdLx1_nls(T,U0,L) computes the factor matrices U{1}, ...,
%   U{N} belonging to a (rank-L(r) x rank-1) block term decomposition of
%   the N-th order tensor T. Each term in the decomposition is the outer
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
%   btdLx1_nls(T,U0,L,options) may be used to set the following options:
%
%      options.Algorithm =   - The desired optimization method.
%      [@nls_gncgs| ...
%       {@nls_gndl}|@nls_lm]
%      options.LargeScale    - If true, the Gauss-Newton or Levenberg-
%      = sum(size(T))*R>1e2    Marquardt steps are computed using a
%                              preconditioned conjugate gradient algorithm.
%                              Otherwise, a direct solver is used.
%      options.M =           - The preconditioner to use when
%      [false|...              options.LargeScale is true.
%       {'block-Jacobi'}]
%      options.<...>         - Parameters passed to the selected method,
%                              e.g., options.TolFun and options.TolX.
%                              See also help [options.Algorithm].
%
%   See also btdLx1_minf.

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
if N < 3, error('btdLx1_nls:T','ndims(T) should be >= 3.'); end

% Check the initial factor matrices U0.
U = U0(:).';
R = sum(L);
P = find(cellfun(@(u)size(u,2),U) == R);
Q = find(cellfun(@(u)size(u,2),U) == length(L));
E = full(sparse(sum(bsxfun(@gt,1:R,cumsum(L)'),1)+1,1:R,1));
size_tens = size(T);
if N ~= length(P)+length(Q)
    error('btdLx1_nls:U0', ...
          'size(U0{n},2) should equal either sum(L) or length(L).');
end
if any(cellfun('size',U,1) ~= size_tens)
    error('btdLx1_nls:U0','size(T,n) should equal size(U0{n},1).');
end

% Check the options structure.
isfunc = @(f)isa(f,'function_handle');
xsfunc = @(f)isfunc(f)&&exist(func2str(f),'file');
if nargin < 4, options = struct; end
if ~isfield(options,'LargeScale')
    options.LargeScale = sum(size_tens)*R > 1e2;
end
if ~isfield(options,'JHasFullRank'), options.JHasFullRank = false; end
if ~isfield(options,'M'), options.M = 'block-Jacobi'; end
if ~isfield(options,'CGMaxIter'), options.CGMaxIter = 10; end
if ~isfield(options,'Algorithm')
    funcs = {@nls_gndl,@nls_gncgs,@nls_lm};
    options.Algorithm = funcs{find(cellfun(xsfunc,funcs),1)};
end

% Cache some intermediate variables.
M = arrayfun(@(n)tens2mat(T,n),1:N,'UniformOutput',false);
offset = [0 cumsum(size_tens.*cellfun(@(u)size(u,2),U))];
UHU = []; updateUHU(U);

% Call the optimization method.
if options.LargeScale, dF.JHJx = @JHJx; else dF.JHJ = @JHJ; end
dF.JHF = @g;
switch options.M
    case 'block-Jacobi', dF.M = @M_blockJacobi;
    case 'Jacobi', dF.M = @M_Jacobi;
    otherwise, if isa(options.M,'function_handle'), dF.M = options.M; end
end
[U,output] = options.Algorithm(@f,dF,U,options);
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

function JHJ = JHJ(U)
% Compute JHJ.
    updateUHU(U);
    EkI = cell(1,N);
    for n = Q, U{n} = U{n}*E; EkI{n} = kron(E.',eye(size_tens(n))); end
    JHJ = zeros(offset(end));
    for n = 1:N
        idxn = offset(n)+1:offset(n+1);
        Wn = prod(UHU(:,:,[1:n-1 n+1:N]),3);
        if any(Q == n)
            JHJ(idxn,idxn) = kron(E*Wn*E.',eye(size_tens(n)));
        else
            JHJ(idxn,idxn) = kron(Wn,eye(size_tens(n)));
        end
        for m = n+1:N
            idxm = offset(m)+1:offset(m+1);
            Wnm = prod(UHU(:,:,[1:n-1 n+1:m-1 m+1:N]),3);
            JHJnm = bsxfun(@times,reshape(U{n},[size_tens(n) 1 1 R]), ...
                    reshape(conj(U{m}),[1 size_tens(m) R 1]));
            JHJnm = bsxfun(@times,JHJnm,reshape(Wnm,[1 1 R R]));
            JHJnm = permute(JHJnm,[1 3 2 4]);
            JHJnm = reshape(JHJnm,[size_tens(n)*R size_tens(m)*R]);
            if any(Q == n), JHJnm = EkI{n}.'*JHJnm; end
            if any(Q == m), JHJnm = JHJnm*EkI{m}; end
            JHJ(idxn,idxm) = JHJnm;
            JHJ(idxm,idxn) = JHJnm';
        end
    end
end

function y = JHJx(U,x)
% Compute JHJ*x.
    for n = Q, U{n} = U{n}*E; end
    y = zeros(size(x));
    for n = 1:N
        Wn = conj(prod(UHU(:,:,[1:n-1 n+1:N]),3));
        idx = offset(n)+1:offset(n+1);
        if any(Q == n), Wn = E*Wn*E.'; end
        y(idx) = reshape(x(idx),size_tens(n),[])*Wn;
    end
    for n = 1:N-1
        idxn = offset(n)+1:offset(n+1);
        Un = reshape(x(idxn),size_tens(n),[]);
        if any(Q == n), Un = Un*E; end
        for m = n+1:N
            idxm = offset(m)+1:offset(m+1);
            Wnm = conj(prod(UHU(:,:,[1:n-1 n+1:m-1 m+1:N]),3));
            Um = reshape(x(idxm),size_tens(m),[]);
            if any(Q == m), Um = Um*E; end
            JHJnmx = U{n}*(Wnm.*(Um.'*conj(U{m})));
            JHJmnx = U{m}*(Wnm.*(Un.'*conj(U{n})));
            if any(Q == n), JHJnmx = JHJnmx*E.'; end
            if any(Q == m), JHJmnx = JHJmnx*E.'; end
            y(idxn) = y(idxn)+JHJnmx(:);
            y(idxm) = y(idxm)+JHJmnx(:);
        end
    end
end
 
function x = M_blockJacobi(~,b)
% Solve Mx = b, where M is a block-diagonal approximation for JHJ.
% Equivalent to simultaneous ALS updates for each of the factor matrices.
    x = zeros(size(b));
    for n = 1:N
        Wn = conj(prod(UHU(:,:,[1:n-1 n+1:N]),3));
        idx = offset(n)+1:offset(n+1);
        if any(Q == n), Wn = E*Wn*E.'; end
        x(idx) = reshape(b(idx),size_tens(n),[])/Wn;
    end
end

function updateUHU(U)
% Cache the Gramians U{n}'*U{n}.
    newState = isempty(UHU);
    for n = 1:N
        if newState, break; end
        newState = any(U0{n}(:) ~= U{n}(:));
    end
    if newState
        U0 = U;
        if isempty(UHU), UHU = zeros(R,R,N); end
        for n = 1:N
            if any(Q == n)
                UHU(:,:,n) = E.'*U{n}'*U{n}*E;
            else
                UHU(:,:,n) = U{n}'*U{n};
            end
        end
    end
end

end
