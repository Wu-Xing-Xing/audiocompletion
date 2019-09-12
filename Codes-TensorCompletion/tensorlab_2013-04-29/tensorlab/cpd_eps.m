function [alpha,output] = cpd_eps(T,U,dU1,dU2,state,options)
%CPD_EPS CPD exact plane search.
%   [alpha,output] = cpd_eps(T,U,dU) computes the real plane search
%   parameters alpha that minimize the objective function f(alpha) := ...
%   0.5*frob(T-cpdgen(U+alpha(1)*dU1+alpha(2)*dU2))^2, in which T is an
%   N-th order tensor, U is a cell array containing N factor matrices U{n},
%   and dUi are cell arrays containing N step matrices dUi{n}.
%
%   If options.Scale is true, the objective function f(alpha) := 0.5* ...
%   frob(T-alpha(3)*cpdgen(U+alpha(1)*dU1+alpha(2)*dU2))^2 is minimized for
%   all three real parameters alpha(1), alpha(2) and alpha(3).
%
%   The structure output returns additional information:
%
%      output.fval    - The value of the objective function at alpha.
%      output.relgain - The improvement in objective function w.r.t.
%                       alpha = [0 0], relative to the improvement of the
%                       step alpha = [1 0] w.r.t. alpha = [0 0].
%
%   cpd_eps(T,U,dU1,dU2,[],options) may be used to set the following
%   options:
%
%      options.Scale = false - If true, computes an optimal scaling
%                              parameter alpha(3) in conjunction with
%                              optimal plane search parameters alpha(1:2).
%      options.SolverOptions - A struct passed to the the polynomial or
%                              rational minimizer. See polymin2 and ratmin2
%                              for details.
%
%   See also cpd_els, cpd_aels.

%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)
%
%   References:
%   [1] L. Sorber, I. Domanov, M. Van Barel, L. De Lathauwer, "Exact Line
%       and Plane Search for Tensor Optimization", ESAT-SISTA Internal
%       Report 13-02, KU Leuven, 2013.
%   [2] M. Rajih, P. Comon, R. Harshman, "Enhanced line search: A novel
%       method to accelerate PARAFAC," SIAM J. Matrix Anal. Appl., Vol. 30
%       2008, pp. 1148-1171.

% Check the tensor T.
N = ndims(T);
if N < 3, error('cpd_eps:T','ndims(T) should be >= 3.'); end

% Check the factor matrices U and dU.
if length(U) ~= N || length(dU1) ~= N || length(dU2) ~= N
    error('cpd_eps:U','length((d)U) should equal ndims(T).');
end
R = size(U{1},2);
if any(cellfun('size',U,2) ~= R) || ...
   any(cellfun('size',dU1,2) ~= R) || ...
   any(cellfun('size',dU2,2) ~= R)
    error('cpd_eps:U','size((d)U{n},2) should be the same for all n.');
end
if any(cellfun('size',U(:).',1) ~= size(T)) || ...
   any(cellfun('size',dU1(:).',1) ~= size(T)) || ...
   any(cellfun('size',dU2(:).',1) ~= size(T))
    error('cpd_eps:U','size(T,n) should equal size((d)U{n},1).');
end

% Check the options structure.
if nargin < 6, options = struct; end
if ~isfield(options,'Scale'), options.Scale = false; end
if ~isfield(options,'SolverOptions'), options.SolverOptions = struct; end

% Check the state structure.
if nargin < 5 || ~isstruct(state), state = struct; end
if ~isfield(state,'fval') || ~isfield(state,'K') || ~isfield(state,'first')
    K1 = kr(U(N:-1:2));
    if ~isfield(state,'fval')
        D = U{1}*K1.';
        D = D(:)-T(:);
        state.fval = 0.5*(D'*D);
    end
    if ~isfield(state,'K') || ~isfield(state,'first')
        state.first = 1;
        state.K{1} = reshape(T,size(T,1),[])*conj(K1);        
    end
end
if ~isfield(state,'T2')
    state.T2 = T(:)'*T(:);
end

% Generate indices of combinations of terms.
Uall = [U(:).'; dU1(:).'; dU2(:).'];
I = cell(1,N);
[I{:}] = ndgrid(1:3);
I = cell2mat(cellfun(@(x)x(:),I,'UniformOutput',false));
degx = sum(I == 2,2);
degy = sum(I == 3,2);

% Compute the polynomial p = <cpdgen(U+x*dU1+y*dU2),T>.
% If state.K is supplied, this costs the equivalent of N*(N+1)/2-1 function
% evaluations. Otherwise, the cost is N*(N+1)/2 function evaluations.
p = zeros(N+1);
p(1) = conj(sum(dot(state.K{state.first},U{state.first})));
for j = 1:N+1
    for i = 1:N+2-j
        next = true;
        if i == 1 && j == 1, continue; end
        J = I(degy == i-1 & degx == j-1,:);
        for k = 1:size(J,1)
            if next
                dT = Uall{J(k)}*kr(Uall(J(k,end:-1:2)+(N-1:-1:1)*3)).';
                next = false;
            else
                dT = dT+Uall{J(k)}*kr(Uall(J(k,end:-1:2)+(N-1:-1:1)*3)).';
            end
        end
        p(i,j) = dT(:)'*T(:);
    end
end

% Build a cell array of all products (d)U{n}'*(d)U{n}.
UHU = cell(1,N);
for n = 1:N
    tmp = cat(2,U{n},dU1{n},dU2{n});
    UHU{n} = tmp'*tmp;
end

% Compute the polynomial q = ||cpdgen(U+x*dU1+y*dU2)||^2.
q = zeros(2*N+1,2*N+1);
for i = 1:size(I,1)
    for j = 1:size(I,1)
        for n = 1:N
            UHUn = UHU{n};
            if n == 1
                W = UHUn((1:R)+R*(I(i,n)-1),(1:R)+R*(I(j,n)-1));
            else
                W = W.*UHUn((1:R)+R*(I(i,n)-1),(1:R)+R*(I(j,n)-1));
            end
        end
        m = degy(i)+degy(j)+1;
        n = degx(i)+degx(j)+1;
        q(m,n) = q(m,n)+real(sum(W(:)));
    end
end

% Compute objective function f = 0.5*[s^2*q - 2*s*real(p) + ||T||^2].
f = q;
f(1:N+1,1:N+1) = f(1:N+1,1:N+1)-2*real(p);
f = 0.5*f;
f(1) = state.fval(end);

if options.Scale
    
    % Compute alpha = argmin -real(p)^2/q.
    [xy,v] = ratmin2(-conv2(real(p),real(p)),q,options.SolverOptions);
    
    % Compute optimal scaling parameter s = real(p)/q.
    if ~isempty(xy)
        [~,idx] = min(v);
        alpha = xy(idx,:);
        pa = real(polyval2(p,alpha));
        qa = polyval2(q,alpha);
        s = pa/qa;
        output.fval = 0.5*(state.T2-(pa*pa)/qa);
        alpha = [alpha s];
    else
        options.Scale = false;
    end
    
end

if ~options.Scale || output.fval > state.fval(end)
    
    % Compute alpha = argmin f.
    [xy,v] = polymin2(f,options.SolverOptions);
    if ~isempty(xy)
        [output.fval,idx] = min(v);
        alpha = xy(idx,:);
    end
    
end

% Compute improvement compared to a step of alpha = 1.
if ~isempty(xy)
    fval0 = state.fval(end);
    fval1 = sum(f(1,:));
    output.relgain = (fval0-output.fval)/max(0,fval0-fval1);
else
    alpha = [];
    output = struct;
end
