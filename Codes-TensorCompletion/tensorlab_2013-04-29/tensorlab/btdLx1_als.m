function [U,output] = btdLx1_als(T,U0,L,options)
%BTDLX1_ALS (rank-L(r) x rank-1) BTD by alternating least squares.
%   [U,output] = btdLx1_als(T,U0,L) computes the factor matrices U{1}, ...,
%   U{N} belonging to a (rank-L(r) x rank-1) block term decomposition of
%   the N-th order tensor T. Each term in the decomposition is the outer
%   product of a rank-L(r) tensor and a rank-1 tensor. The vector L has
%   length R and defines the ranks L(r). The algorithm is initialized with
%   the factor matrices U0{n}, which must have either sum(L) or R columns.
%   For example, in a rank-(L(r),L(r),1) BTD the first two factor matrices
%   U0{1} and U0{2} should have sum(L) columns while U{3} should have R.
%   The structure output returns additional information:
%
%      output.alpha      - The value of the line search step length in
%                          every iteration.
%      output.fval       - The value of the objective function
%                          0.5*frob(T-cpdgen(U))^2 in every iteration.
%      output.info       - The circumstances under which the procedure
%                          terminated:
%                             1: Objective function tolerance reached.
%                             2: Step size tolerance reached.
%                             3: Maximum number of iterations reached.
%      output.iterations - The number of iterations.
%
%   btdLx1_als(T,U0,L,options) may be used to set the following options:
%
%      options.FastUpdate = true - If true, the normal equations are solved
%                                  explicitly. Otherwise, a numerically
%                                  more stable, but slower, method is used.
%                                  Additionally, this enables computing
%                                  the objective function value fval with a
%                                  cheap expression (when it is accurate).
%      options.LineSearch =      - A function handle to the desired line
%      [{false}|@cpd_aels|...      search algorithm.
%       @cpd_els|@cpd_lsb]
%      options.LineSearchOptions - An options structure passed to the
%                                  selected line search algorithm.
%      options.MaxIter = 500     - The maximum number of iterations.
%      options.Order = 1:N       - The order in which to update the factor
%                                  matrices. Must be a permutation of 1:N.
%      options.TolFun = 1e-8     - The tolerance for the objective function
%                                  between two successive iterates,
%                                  relative to its initial value. Note that
%                                  because the objective function is a
%                                  squared norm, TolFun can be as small as
%                                  eps^2 = 1e-32.
%      options.TolX = 1e-6       - The tolerance for the step size relative
%                                  to the norm of the current iterate.

%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)
%
%   References:
%   [1] L. Sorber, M. Van Barel, L. De Lathauwer, "Optimization-based
%       algorithms for tensor decompositions: canonical polyadic
%       decomposition, decomposition in rank-(Lr,Lr,1) terms and a new
%       generalization," SIAM J. Opt., 2013.

% Check the tensor T.
N = ndims(T);
if N < 3, error('btdLx1_als:T','ndims(T) should be >= 3.'); end

% Check the initial factor matrices U0.
U = U0(:).';
R = sum(L);
P = find(cellfun(@(u)size(u,2),U) == R);
Q = find(cellfun(@(u)size(u,2),U) == length(L));
E = full(sparse(sum(bsxfun(@gt,1:R,cumsum(L)'),1)+1,1:R,1));
if N ~= length(P)+length(Q)
    error('btdLx1_als:U0', ...
          'size(U0{n},2) should equal either sum(L) or length(L).');
end
if any(cellfun('size',U,1) ~= size(T))
    error('btdLx1_als:U0','size(T,n) should equal size(U0{n},1).');
end

% Check the options structure.
isfunc = @(f)isa(f,'function_handle');
xsfunc = @(f)isfunc(f)&&exist(func2str(f),'file');
if nargin < 4, options = struct; end
if ~isfield(options,'FastUpdate'), options.FastUpdate = true; end
if ~isfield(options,'LineSearch'), options.LineSearch = false; end
if ~xsfunc(options.LineSearch) ...
   && (~isa(options.LineSearch,'function_handle') && options.LineSearch)
    error('btdLx1_als:LineSearch','Not a valid line search algorithm.');
end
if ~isfield(options,'LineSearchOptions')
    options.LineSearchOptions = struct;
end
if ~isfield(options,'MaxIter'), options.MaxIter = 500; end
if ~isfield(options,'Order'), options.Order = 1:N; end
if ~isfield(options,'TolFun'), options.TolFun = 1e-8; end
if ~isfield(options,'TolX'), options.TolX = 1e-6; end

% Cache some intermediate variables.
T2 = T(:)'*T(:);
if options.FastUpdate
    K = cell(1,N);
    M = arrayfun(@(n)tens2mat(T,n),1:N,'UniformOutput',false);
    UHU1 = zeros(R,R,N);
    UHU = zeros(R,R,N);
    for n = Q, U{n} = U{n}*E; end
    for n = 1:N, UHU(:,:,n) = U{n}'*U{n}; end
else
    M = arrayfun(@(n)tens2mat(T,n).',1:N,'UniformOutput',false);
end

% Alternating least squares.
first = options.Order(1);
last = options.Order(end);
D = kr(U([N:-1:first+1 first-1:-1:1]));
if options.FastUpdate
    K{first} = M{first}*conj(D);
	D = M{first}-U{first}*D.';
else
	D = M{first}-D*U{first}.';
end
output.alpha = [];
output.fval = 0.5*(D(:)'*D(:));
output.info = false;
output.iterations = 0;
output.relgain = [];
while ~output.info
    
    % Save current state.
    U1 = U;
    
    % Update factor matrices.
    for n = options.Order
        if options.FastUpdate
            W = prod(UHU(:,:,[1:n-1 n+1:N]),3);
            if n ~= first
                K{n} = M{n}*conj(kr(U([N:-1:n+1 n-1:-1:1])));
            end
            if any(Q == n)
                U{n} = ((K{n}*E.')/(E*conj(W)*E.'))*E;
            else
                U{n} = K{n}/conj(W);
            end
            UHU(:,:,n) = U{n}'*U{n};
        else
            if any(Q == n)
                U{n} = (kr(U([N:-1:n+1 n-1:-1:1]))\M{n}).';
            else
                U{n} = (((kr(U([N:-1:n+1 n-1:-1:1]))*E.')\M{n}).')*E;
            end
        end
    end
    
    % Line search.
    alpha = nan;
    if isfunc(options.LineSearch)
        % The struct state contains the current output state and some other
        % useful information such as T(:)'*T(:), available to the line
        % search procedure.
        dU = cellfun(@(u,v)u-v,U,U1,'UniformOutput',false);
        state = output;
        state.first = first; state.last = last; state.T2 = T2;
        state.UHU = UHU; if options.FastUpdate, state.K = K; end
        [alpha,ls] = options.LineSearch(T,U1,dU, ...
                     state,options.LineSearchOptions);
        % Interpret the line search as a scaled line search.
        if isempty(alpha), alpha = [1 1];
        elseif length(alpha) == 1, alpha(2) = 1; end
        if isreal(alpha(2))
            alpha(2) = nthroot(alpha(2),N);
        else
            alpha(2) = alpha(2)^(1/N);
        end
    end
    % Use line search output if worthwhile.
    isValidLS = ~any(isnan(alpha)) && ~all(alpha == 1) && ...
                (~isfield(ls,'fval') || ls.fval(end) < output.fval(end));
    if isValidLS
        if isfield(ls,'fval')
            % Objective function value available and better than previous,
            % set next iterate.
            U = cellfun(@(u,v)alpha(2)*(u+alpha(1)*v),U1,dU, ...
                        'UniformOutput',false);
            if options.FastUpdate
                K{first} = M{first}* ...
                           conj(kr(U([N:-1:first+1 first-1:-1:1])));
                for n = 1:N, UHU(:,:,n) = U{n}'*U{n}; end
            end
            fval = ls.fval(end);
            output.alpha(:,end+1) = alpha(:).';
        else
            % Line search parameter computed, but no objective function
            % info. Test if better iterate, otherwise use alpha = 1.
            Ualpha = cellfun(@(u,v)alpha(2)*(u+alpha(1)*v),U1,dU, ...
                             'UniformOutput',false);
            if options.FastUpdate
                K1 = M{first}* ...
                     conj(kr(Ualpha([N:-1:first+1 first-1:-1:1])));
                for n = 1:N, UHU1(:,:,n) = Ualpha{n}'*Ualpha{n}; end
            end
            % Compute cheap fval if possible.
            if options.FastUpdate && ...
               log10(output.fval(end)) > log10(T2)-16+2.5
                fvalalpha = abs(.5*(T2+sum(sum(real(prod(UHU1,3)))))- ...
                                real(sum(dot(K1,Ualpha{first}))));
            else
                D = Ualpha{1}*kr(Ualpha(N:-1:2)).';
                D = T(:)-D(:);
                fvalalpha = 0.5*(D'*D);
            end
            if fvalalpha < output.fval(end)
                U = Ualpha;
                if options.FastUpdate, K{first} = K1; UHU = UHU1; end
                fval = fvalalpha;
                output.alpha(:,end+1) = alpha(:).';
            else
                isValidLS = false;
            end
        end
    end
    
    % If there was no line search, or if it resulted in a worse iterate,
    % use alpha = 1 and recompute the objective function value.
    if ~isValidLS
        output.alpha(:,end+1) = [1; 1];
        if options.FastUpdate
            K{first} = M{first}*conj(kr(U([N:-1:first+1 first-1:-1:1])));
        end
        if options.FastUpdate && log10(output.fval(end)) > log10(T2)-16+2.5
            fval = abs(0.5*(T2+sum(sum(real(W.*UHU(:,:,last)))))- ...
                       real(sum(dot(K{last},U{last}))));
        else
            D = U{1}*kr(U(N:-1:2)).';
            D = T(:)-D(:);
            fval = 0.5*(D'*D);
        end
    end
    
    % Update the output structure.
    if isValidLS && isfield(ls,'relgain')
        output.relgain(end+1) = ls.relgain;
    else
        output.relgain(end+1) = 1;
    end
    output.fval(end+1) = fval;
    output.iterations = output.iterations+1;
    output.info = abs(diff(output.fval(end:-1:end-1))) <= ...
                  options.TolFun*output.fval(1);
    if sqrt(sum(cellfun(@(u,v)(u(:)-v(:))'*(u(:)-v(:)),U,U1))) <= ...
       options.TolX*sqrt(sum(cellfun(@(u)u(:)'*u(:),U)))
        output.info = 2;
    end
    if output.iterations >= options.MaxIter, output.info = 3; end
    
end

% Invert the matrix E.
L = cumsum(sum(E,2))+1;
L = [1 L(1:end-1).'];
for n = Q, C = U{n}; U{n} = C(:,L); end
