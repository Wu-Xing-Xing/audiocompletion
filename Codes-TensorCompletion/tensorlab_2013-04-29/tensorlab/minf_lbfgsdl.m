function [z,output] = minf_lbfgsdl(f,g,z0,options)
%MINF_LBFGSDL Minimize a function by L-BFGS with dogleg trust region.
%   [z,output] = minf_lbfgsdl(f,g,z0) starts at z0 and attempts to find a
%   local minimizer of the real-valued function f(z). The input variables z
%   may be a scalar, vector, matrix, tensor or even a cell array of tensors
%   and its contents may be real or complex.
%
%   If f(x) is a function of real variables x, the function g(x) should
%   compute the partial derivatives of f with respect to the real variables
%   x, i.e. g(xk) := df(xk)/dx. If f(z) is a function of complex variables
%   z, the function g(z) should compute two times the partial derivative
%   of f with respect to conj(z) (treating z as constant), i.e. g(zk) :=
%   2*df(zk)/d(conj(z)) = 2*conj(df(zk)/dz). If g is the empty matrix [],
%   the real gradient or scaled conjugate cogradient is approximated with
%   finite differences. The output of the function g(z) may have the same
%   structure as z (although this is not necessary). The structure output
%   returns additional information:
%
%      output.delta      - The trust region radius at every step attempt.
%      output.fevals     - The total number of function calls.
%      output.fval       - The value of the objective function f in every
%                          iteration.
%      output.gevals     - The total number of gradient calls.
%      output.info       - The circumstances under which the procedure
%                          terminated:
%                             1: Objective function tolerance reached.
%                             2: Step size tolerance reached.
%                             3: Maximum number of iterations reached.
%      output.iterations - The number of iterations.
%      output.rho        - The trustworthiness at every step attempt.
%
%   minf_lbfgsdl(f,g,z0,options) may be used to set the following options:
%
%      options.delta =        - The initial trust region radius.
%      norm(g(z0))/numel(z0)
%      options.m =            - The number of updates to store.
%      min(30,length(z0))
%      options.MaxIter = 500  - The maximum number of iterations.
%      options.TolFun = 1e-6  - The tolerance for the objective function
%                               between two successive iterates, relative
%                               to its initial value.
%      options.TolX = 1e-6    - The tolerance for the step size relative to
%                               the norm of the current iterate.

%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)
%
%   References:
%   [1] L. Sorber, M. Van Barel, L. De Lathauwer, "Unconstrained
%       optimization of real functions in complex variables", SIAM J. Opt.,
%       Vol. 22, No. 3, 2012, pp. 879-898.

% Store the structure of the input space and evaluate the gradient.
dim = structure(z0);
fval = f(z0);
if ~isa(g,'function_handle') && isempty(g)
    grad = serialize(deriv(f,z0,fval));
else
    grad = serialize(g(z0));
end
z0 = serialize(z0);

% Check the options structure.
if nargin < 4, options = struct; end
if ~isfield(options,'delta'), options.delta = norm(grad)/numel(z0); end
if ~isfield(options,'m'), options.m = min(30,length(z0)); end
if ~isfield(options,'MaxIter'), options.MaxIter = 500; end
if ~isfield(options,'TolFun'), options.TolFun = 1e-6; end
if ~isfield(options,'TolX'), options.TolX = 1e-6; end

% Initialize the algorithm.
S = zeros(numel(z0),options.m);
Y = zeros(numel(z0),options.m);
a = zeros(1,options.m);
r = zeros(1,options.m);
m = 0;
midx = 0;

% L-BFGS with dogleg trust region.
output.delta = options.delta;
output.fevals = 1;
output.fval = fval;
output.gevals = 1;
output.info = false;
output.iterations = 0;
output.rho = [];
while ~output.info

    % Compute the quasi-Newton step pqn = -H*grad.
    pqn = -grad;
    for i = 1:m
        a(i) = r(midx(i))*real(S(:,midx(i))'*pqn);
        pqn = pqn-a(i)*Y(:,midx(i));
    end
    if m > 0
        y1 = Y(:,midx(1));
        y1y1 = y1'*y1;
        gamma = y1y1*r(midx(1));
        pqn = 1/gamma*pqn;
    end
    for i = m:-1:1
        b = r(midx(i))*real(Y(:,midx(i))'*pqn);
        pqn = pqn+(a(i)-b)*S(:,midx(i));
    end
    
    % Approximate the Cauchy point pcp = -alpha*grad, where alpha is equal
    % to arg min m(-alpha*grad) and m(p) is a second order model of f at z.
    gg = grad'*grad;
    if m == 0
        alpha = 1;
    else
        s1 = S(:,midx(1));
        gBg = gg-real(grad'*s1)^2/(s1'*s1)+real(grad'*y1)^2/y1y1;
        gBg = gamma*gBg;
        alpha = gg/gBg;
    end

    % Dogleg trust region.
    rho = -inf;
    while rho <= 0

        % Compute the dogleg step p.
        delta = output.delta(end);
        if norm(pqn) <= delta
            p = pqn;
            dfval = -0.5*real(grad'*pqn);
        elseif alpha*sqrt(gg) >= delta
            p = (-delta/sqrt(gg))*grad;
            dfval = delta*(sqrt(gg)-0.5*delta/alpha);
        else
            bma = pqn+alpha*grad; bmabma = bma'*bma;
            a = -alpha*grad; aa = alpha^2*gg;
            c = real(a'*bma);
            if c <= 0
                beta = (-c+sqrt(c^2+bmabma*(delta^2-aa)))/bmabma;
            else
                beta = (delta^2-aa)/(c+sqrt(c^2+bmabma*(delta^2-aa)));
            end
            p = a+beta*bma;
            dfval = 0.5*alpha*(1-beta)^2*gg- ...
                    0.5*beta *(2-beta)*real(grad'*pqn);
        end

        % Compute the trustworthiness rho.
        z = deserialize(z0+p,dim);
        fval = f(z);
        rho = (output.fval(end)-fval)/dfval;
        if isnan(rho), rho = -inf; end
        output.rho(end+1) = rho;
        output.fevals = output.fevals+1;

        % Update trust region radius delta.
        if rho > 0.5
            output.delta(end+1) = max(delta,2*norm(p));
        else
            sigma = (1-0.25)/(1+exp(-14*(rho-0.25)))+0.25;
            output.delta(end+1) = sigma*delta;
        end
        
        % Check for convergence.
        if norm(p) <= options.TolX*norm(z0)
            output.info = 2;
            z = deserialize(z0,dim);
            return;
        end

    end
    
    % Save current state.
    z0 = z0+p;
    grad1 = grad;
    
    % Evaluate the gradient and update step information.
    if ~output.info
        if ~isa(g,'function_handle') && isempty(g)
            grad = serialize(deriv(f,z,fval));
        else
            grad = serialize(g(z));
        end
        s = p;
        y = grad-grad1;
        sy = real(y'*s);
        if sy > 0
            m = min(m+1,options.m);
            midx = [midx(1)+1:-1:1,m:-1:midx(1)];
            S(:,midx(1)) = s;
            Y(:,midx(1)) = y;
            r(:,midx(1)) = 1/sy;
        end
    end
    
    % Update the output structure.
    output.fval(end+1) = fval;
    output.gevals = output.gevals+1;
    output.iterations = output.iterations+1;
    output.info = abs(diff(output.fval(end:-1:end-1))) <= ...
                  options.TolFun*output.fval(1);
    if output.iterations >= options.MaxIter, output.info = 3; end

end

end

function z = deserialize(z,dim)
    if iscell(dim)
        v = z; z = cell(size(dim));
        s = cellfun(@(s)prod(s(:)),dim(:)); o = [0; cumsum(s)];
        for i = 1:length(s), z{i} = reshape(v(o(i)+(1:s(i))),dim{i}); end
    elseif ~isempty(dim)
        z = reshape(z,dim);
    end
end

function z = serialize(z)
    if iscell(z)
        s = cellfun(@numel,z(:)); o = [0; cumsum(s)];
        c = z; z = zeros(o(end),1);
        for i = 1:length(s), ci = c{i}; z(o(i)+(1:s(i))) = ci(:); end
    else
        z = z(:);
    end
end

function dim = structure(z)
    if iscell(z)
        dim = cellfun(@size,z,'UniformOutput',false);
    else
        dim = size(z);
        if numel(z) == dim(1), dim = []; end
    end
end
