function [U,output] = cpd_nls(T,U0,options)
%CPD_NLS CPD by nonlinear least squares.
%   [U,output] = cpd_nls(T,U0) computes the factor matrices U{1}, ..., U{N}
%   belonging to a canonical polyadic decomposition of the N-th order
%   tensor T by minimizing 0.5*frob(T-cpdgen(U))^2. The algorithm is
%   initialized with the factor matrices U0{n}. The structure output
%   returns additional information:
%
%      output.Name  - The name of the selected algorithm.
%      output.<...> - The output of the selected algorithm.
%
%   cpd_nls(T,U0,options) may be used to set the following options:
%
%      options.Algorithm =   - The desired optimization method.
%      [@nls_gncgs| ...
%       {@nls_gndl}|@nls_lm]
%      options.LargeScale    - If true, the Gauss-Newton or Levenberg-
%      = sum(size(T))*R>1e2    Marquardt steps are computed using a
%                              preconditioned conjugate gradient algorithm.
%                              Otherwise, a direct solver is used.
%      options.M =           - The preconditioner to use when
%      [false|'Jacobi'|...     options.LargeScale is true.
%       {'block-Jacobi'}|...
%       'block-SSOR']
%      options.<...>         - Parameters passed to the selected method,
%                              e.g., options.TolFun and options.TolX.
%                              See also help [options.Algorithm].
%
%   See also cpd_minf.

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

% Check the initial factor matrices U0.
N = ndims(T);
U = U0(:).';
R = size(U{1},2);
size_tens = size(T);
if any(cellfun('size',U,2) ~= R)
    error('cpd_nls:U0','size(U0{n},2) should be the same for all n.');
end
if any(cellfun('size',U,1) ~= size_tens)
    error('cpd_nls:U0','size(T,n) should equal size(U0{n},1).');
end

% Check the options structure.
isfunc = @(f)isa(f,'function_handle');
xsfunc = @(f)isfunc(f)&&exist(func2str(f),'file');
if nargin < 3, options = struct; end
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
offset = [0 cumsum(size_tens)]*R;
UHU = []; updateUHU(U);

% Call the optimization method.
if options.LargeScale, dF.JHJx = @JHJx; else dF.JHJ = @JHJ; end
dF.JHF = @g;
switch options.M
    case 'block-SSOR', dF.M = @M_blockSSOR;
    case 'block-Jacobi', dF.M = @M_blockJacobi;
    case 'Jacobi', dF.M = @M_Jacobi;
    otherwise, if isa(options.M,'function_handle'), dF.M = options.M; end
end
[U,output] = options.Algorithm(@f,dF,U,options);
output.Name = func2str(options.Algorithm);

function fval = f(U)
    D = M{1}-U{1}*kr(U(end:-1:2)).';
    fval = 0.5*sum(D(:)'*D(:));
end

function grad = g(U)
    updateUHU(U);
    grad = cell(1,N);
    for n = 1:N
        G1 = U{n}*conj(prod(UHU(:,:,[1:n-1 n+1:N]),3));
        G2 = M{n}*conj(kr(U([N:-1:n+1 n-1:-1:1])));
        grad{n} = G1-G2;
    end
end

function JHJ = JHJ(U)
% Compute JHJ.
    updateUHU(U);
    JHJ = zeros(offset(end));
    for n = 1:N
        idxn = offset(n)+1:offset(n+1);
        Wn = prod(UHU(:,:,[1:n-1 n+1:N]),3);
        JHJ(idxn,idxn) = kron(Wn,eye(size_tens(n)));
        for m = n+1:N
            idxm = offset(m)+1:offset(m+1);
            Wnm = prod(UHU(:,:,[1:n-1 n+1:m-1 m+1:N]),3);
            JHJnm = bsxfun(@times,reshape(U{n},[size_tens(n) 1 1 R]), ...
                    reshape(conj(U{m}),[1 size_tens(m) R 1]));
            JHJnm = bsxfun(@times,JHJnm,reshape(Wnm,[1 1 R R]));
            JHJnm = permute(JHJnm,[1 3 2 4]);
            JHJnm = reshape(JHJnm,[size_tens(n)*R size_tens(m)*R]);
            JHJ(idxn,idxm) = JHJnm;
            JHJ(idxm,idxn) = JHJnm';
        end
    end
end

function y = JHJx(U,x)
% Compute JHJ*x.
    XHU = zeros(size(UHU));
    y = zeros(size(x));
    for n = 1:N
        idx = offset(n)+1:offset(n+1);
        Wn = conj(prod(UHU(:,:,[1:n-1 n+1:N]),3));
        Xn = reshape(x(idx),size_tens(n),R);
        XHU(:,:,n) = Xn'*U{n};
        y(idx) = Xn*Wn;
    end
    for n = 1:N-1
        idxn = offset(n)+1:offset(n+1);
        Wn = zeros(R);
        for m = n+1:N
            idxm = offset(m)+1:offset(m+1);
            Wnm = conj(prod(UHU(:,:,[1:n-1 n+1:m-1 m+1:N]),3));
            Wn = Wn+Wnm.*conj(XHU(:,:,m));
            JHJmnx = U{m}*(Wnm.*conj(XHU(:,:,n)));
            y(idxm) = y(idxm)+JHJmnx(:);
        end
        JHJnx = U{n}*Wn;
        y(idxn) = y(idxn)+JHJnx(:);
    end
end

function x = M_blockSSOR(U,b)
% Solve Mx = b, where M is a block-Symmetric Successive Overrelaxation
% preconditioner.
    % x = inv(D)*(U+D)*b
    B = cell(size(U));
    BHU = zeros(size(UHU));
    for n = 1:N
        B{n} = b(offset(n)+1:offset(n+1));
        BHU(:,:,n) = B{n}'*U{n};
    end
    X = B;
    for n = 1:N-1
        Wsum = zeros(R);
        for m = n+1:N
            Wnm = conj(prod(UHU(:,:,[1:n-1 n+1:m-1 m+1:N]),3));
            Wsum = Wsum+Wnm.*conj(BHU(:,:,m));
        end
        Wn = conj(prod(UHU(:,:,[1:n-1 n+1:N]),3));
        X{n} = X{n}+(U{n}*Wsum)/Wn;
    end
    % x = (L+D)*x
    B = X;
    for n = 1:N
        Wn = conj(prod(UHU(:,:,[1:n-1 n+1:N]),3));
        BHU(:,:,n) = B{n}'*U{n};
        X{n} = B{n}*Wn;
    end
    for n = 2:N
        Wsum = zeros(R);
        for m = 1:n-1
            Wnm = conj(prod(UHU(:,:,[1:m-1 m+1:n-1 n+1:N]),3));
            Wsum = Wsum+Wnm.*conj(BHU(:,:,m));
        end
        X{n} = X{n}+U{n}*Wsum;
    end
    x = serialize(X);
end

function x = M_blockJacobi(~,b)
% Solve Mx = b, where M is a block-diagonal approximation for JHJ.
% Equivalent to simultaneous ALS updates for each of the factor matrices.
    x = zeros(size(b));
    for n = 1:N
        Wn = conj(prod(UHU(:,:,[1:n-1 n+1:N]),3));
        idx = offset(n)+1:offset(n+1);
        x(idx) = reshape(b(idx),size_tens(n),R)/Wn;
    end
end

function x = M_Jacobi(~,b)
% Solve Mx = b, where M is a diagonal approximation for JHJ.
    x = zeros(size(b));
    W = zeros(size(b));
    for n = 1:N
        idx = bsxfun(@plus,R*R*(([1:n-1 n+1:N]).'-1),1+(R+1)*(0:R-1));
        Wn = prod(UHU(idx),1).';
        idx = offset(n)+1:offset(n+1);
        W(idx) = kron(Wn,ones(size_tens(n),1));
    end
    idx = W > 1e-6*max(W);
    x(idx) = b(idx)./W(idx);
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
        for n = 1:N, UHU(:,:,n) = U{n}'*U{n}; end
    end
end

end
