function [alpha,output] = cpd_els(T,U,dU,state,options)
%CPD_ELS CPD exact line search.
%   [alpha,output] = cpd_els(T,U,dU) computes the real or complex line
%   search parameter alpha that minimizes the objective function f(alpha)
%   := 0.5*frob(T-cpdgen(U+alpha*dU))^2, in which T is an N-th order
%   tensor, U is a cell array containing N factor matrices U{n}, and dU is
%   a cell array containing N step matrices dU{n}.
%
%   If options.Scale is true, the objective function f(alpha) := 0.5* ...
%   frob(T-alpha(2)*cpdgen(U+alpha(1)*dU))^2 is minimized for both
%   parameters alpha(1) and alpha(2).
%
%   The structure output returns additional information:
%
%      output.fval    - The value of the objective function f at alpha.
%      output.relgain - The improvement in objective function w.r.t.
%                       alpha = 0, relative to the improvement of the step
%                       alpha = 1 w.r.t. alpha = 0.
%
%   cpd_els(T,U,dU,[],options) may be used to set the following options:
%
%      options.IsReal =      - If true, alpha is restricted to the real
%      isreal(T)               domain. Otherwise, alpha may be complex
%                              (e.g., in case of a complex CPD).
%      options.Scale = false - If true, computes an optimal scaling
%                              parameter alpha(2) in conjunction with an
%                              optimal line search parameter alpha(1).
%      options.SolverOptions - A struct passed to the the polynomial or
%                              rational minimizer. See polymin, polymin2, 
%                              ratmin and ratmin2 for details.
%
%   See also cpd_aels, cpd_eps.

%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)
%
%   References:
%   [1] L. Sorber, I. Domanov, M. Van Barel, L. De Lathauwer, "Exact Line
%       and Plane Search for Tensor Optimization", ESAT-SISTA Internal
%       Report 13-02, KU Leuven, 2013.
%   [2] M. Rajih, P. Comon, R. Harshman, "Enhanced line search: A novel
%       method to accelerate PARAFAC," SIAM J. Matrix Anal. Appl., Vol. 30,
%       2008, pp. 1148-1171.

% Check the tensor T.
N = ndims(T);
if N < 3, error('cpd_els:T','ndims(T) should be >= 3.'); end

% Check the factor matrices U and dU.
if length(U) ~= N || length(dU) ~= N
    error('cpd_els:U','length((d)U) should equal ndims(T).');
end
R = size(U{1},2);
if any(cellfun('size',U,2) ~= R) || any(cellfun('size',dU,2) ~= R)
    error('cpd_els:U','size((d)U{n},2) should be the same for all n.');
end
if any(cellfun('size',U(:).',1) ~= size(T)) || ...
   any(cellfun('size',dU(:).',1) ~= size(T))
    error('cpd_els:U','size(T,n) should equal size((d)U{n},1).');
end

% Check the options structure.
if nargin < 5, options = struct; end
if ~isfield(options,'IsReal'), options.IsReal = isreal(T); end
if ~isfield(options,'Scale'), options.Scale = false; end
if ~isfield(options,'SolverOptions'), options.SolverOptions = struct; end

% Check the state structure.
if nargin < 4 || ~isstruct(state), state = struct; end
if ~isfield(state,'fval') || ...
   (options.Scale && (~isfield(state,'K') || ~isfield(state,'first')))
    K1 = kr(U(N:-1:2));
    if ~isfield(state,'fval')
        D = U{1}*K1.';
        D = D(:)-T(:);
        state.fval = 0.5*(D'*D);
    end
    if options.Scale && (~isfield(state,'K') || ~isfield(state,'first'))
        state.first = 1;
        state.K{1} = reshape(T,size(T,1),[])*conj(K1);
    end
end
if ~isfield(state,'T2')
    state.T2 = T(:)'*T(:);
end

% Generate indices of combinations of terms.
Uall = [U(:).'; dU(:).'];
I = cell2mat(arrayfun(@(i)bitget(i,1:N),(0:2^N-1)','UniformOutput',false));
[deg,idx] = sort(sum(I,2));
I = I(idx,:)+1;

% Compute the polynomial p = <cpdgen(U+z*dU),T> if options.Scale is true.
% Otherwise compute p = <cpdgen(U+z*dU),T-cpdgen(U)> so that the objective
% function f is more accurate later on. If state.K is supplied, this costs
% the equivalent of N function evaluations. Otherwise, the cost is N+1
% function evaluations.
p = zeros(N+1,1);
if options.Scale
    p(1) = conj(sum(dot(state.K{state.first},U{state.first})));
else
    T = reshape(T,size(T,1),[])-U{1}*kr(U(N:-1:2)).';
    p(1) = -0.5*(T(:)'*T(:));
end
for d = 1:N
    next = true;
    for i = find(deg.' == d)
        if next
            dT = Uall{I(i,1)}*kr(Uall(I(i,end:-1:2)+(N-1:-1:1)*2)).';
            next = false;
        else
            dT = dT+Uall{I(i,1)}*kr(Uall(I(i,end:-1:2)+(N-1:-1:1)*2)).';
        end
    end
    p(d+1) = dT(:)'*T(:);
end

% Build a cell array of all products (d)U{n}'*(d)U{n}.
UHU = cell(1,N);
for n = 1:N
    tmp = cat(2,U{n},dU{n});
    UHU{n} = tmp'*tmp;
end

% Compute the polynomial q = ||cpdgen(U+z*dU)||^2.
q = zeros(N+1,N+1);
for j = 1+~options.Scale:N+1
    for i = 1+~options.Scale:j
        for n = 1:N
            col = bsxfun(@plus,1:R,R*(I(deg == j-1,n)-1)).';
            row = bsxfun(@plus,1:R,R*(I(deg == i-1,n)-1)).';
            UHUn = UHU{n};
            if n == 1, W = UHUn(row,col); else W = W.*UHUn(row,col); end
        end
        q(i,j) = sum(W(:));
    end
end
q = triu(q,1)+diag(real(diag(q)))+triu(q,1)';

% Compute objective function f = 0.5*[s*s'*q - s'*p - s*conj(p) + ||T||^2].
f = q;
if options.Scale
    f(:,1) = f(:,1)-p;
    f(1,:) = f(:,1)';
    f(1) = state.fval(end);
else
	f(:,1) = -p;
    f(1,:) = -p';
end
f(2:end) = 0.5*f(2:end);

if options.Scale
    
    % Compute alpha = argmin -real(p)^2/q or -p*conj(p)/q.
    pr = real(p');
    pr = conv(pr,pr);
    qr = zeros(1,N+1);
    for d = -N:N, qr(d+N+1) = sum(real(diag(fliplr(q),d))); end
    [z,v] = ratmin(fliplr(-pr),qr,options.SolverOptions);
    if ~options.IsReal
        options.SolverOptions.Univariate = true;
        [zc,vc] = ratmin2(-p*p',q,options.SolverOptions);
        if min(vc) < min(v), z = zc; v = vc; end
    end
    
    % Compute optimal scaling parameter s = real(p)/q or p/q.
    if ~isempty(z)
        [~,idx] = min(v);
        alpha = z(idx);
        if options.IsReal
            pa = real(polyval(fliplr(p'),alpha));
            qa = polyval(qr,alpha);
        else
            pa = polyval(fliplr(p.'),alpha');
            qa = polyval2(q,alpha);
        end
        s = pa/qa;
        output.fval = 0.5*(state.T2-(pa*pa')/qa);
        alpha = [alpha s];
    end
    
else
    
    % Compute alpha = argmin f.
    fr = zeros(1,N+1);
    for d = -N:N, fr(d+N+1) = sum(real(diag(fliplr(f),d))); end
    [z,v] = polymin(fr,options.SolverOptions);
    if ~options.IsReal
        options.SolverOptions.Univariate = true;
        [zc,vc] = polymin2(f,options.SolverOptions);
        if min(vc) < min(v), z = zc; v = vc; end
    end
    if ~isempty(z)
        [output.fval,idx] = min(v);
        alpha = z(idx);
    end
    
end

% Compute improvement compared to a step of alpha = 1.
if ~isempty(z) && output.fval <= state.fval(end)
    fval0 = state.fval(end);
    fval1 = real(sum(f(:)));
    if log10(state.fval(end)) <= log10(state.T2)-16+2.5
        % Remove output.fval if it is likely to be inaccurate.
        output = rmfield(output,'fval');
        output.relgain = nan;
    else
        % If the fval estimator is accurate and the fval is worse than a
        % step alpha = 1, use alpha = 1.
        if output.fval > fval1
            if options.Scale, alpha = [1 1]; else alpha = 1; end
            output.fval = fval1;
        end
        output.relgain = (fval0-output.fval)/max(0,fval0-fval1);
    end
else
    alpha = [];
    output = struct;
end
