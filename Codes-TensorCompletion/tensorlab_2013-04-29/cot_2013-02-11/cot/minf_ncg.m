function [z,output] = minf_ncg(f,g,z0,options)
%MINF_NCG Minimize a function by nonlinear conjugate gradient.
%   [z,output] = minf_ncg(f,g,z0) starts at z0 and attempts to find a local
%   minimizer of the real-valued function f(z). The input variables z may
%   be a scalar, vector, matrix, tensor or even a cell array of tensors and
%   its contents may be real or complex.
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
%      output.alpha      - The line search step length in every iteration.
%      output.fevals     - The total number of function/gradient calls.
%      output.fval       - The value of the objective function f in every
%                          iteration.
%      output.info       - The circumstances under which the procedure
%                          terminated:
%                             1: Objective function tolerance reached.
%                             2: Step size tolerance reached.
%                             3: Maximum number of iterations reached.
%      output.infols     - The circumstances under which the line search
%                          terminated in every iteration.
%      output.iterations - The number of iterations.
%
%   minf_ncg(f,g,z0,options) may be used to set the following options:
%
%      options.beta =            - The conjugate gradient update parameter.
%      ['HS',{'HSm'},'PR',...      Options are 'HS' (Hestenes-Stiefel),
%       'PRm','FR,'SD']            'HSm' (modified Hestenes-Stiefel), 'PR'
%                                  (Polak-Ribiere), 'PRm' (modified Polak-
%                                  Ribiere), 'FR' (Fletcher-Reeves) and
%                                  'SD' (steepest descent).
%      options.LineSearch        - The line search used to minimize the
%      = @ls_mt                    objective function in the quasi-Newton
%                                  descent direction.
%      options.LineSearchOptions - The options structure passed to the line
%                                  search routine.
%      options.MaxIter = 500     - The maximum number of iterations.
%      options.TolFun = 1e-6     - The tolerance for the objective function
%                                  between two successive iterates,
%                                  relative to its initial value.
%      options.TolX = 1e-6       - The tolerance for the step size relative
%                                  to the norm of the current iterate.

%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)
%
%   References:
%   [1] L. Sorber, M. Van Barel, L. De Lathauwer, "Unconstrained
%       optimization of real functions in complex variables", SIAM J. Opt.,
%       Vol. 22, No. 3, 2012, pp. 879-898.

% Evaluate the objective function and gradient.
fval = f(z0);
if ~isa(g,'function_handle') && isempty(g)
    grad = serialize(deriv(f,z0,fval));
else
    grad = serialize(g(z0));
end
z = z0;

% Check the options structure.
if nargin < 4, options = struct; end
if ~isfield(options,'alpha')
    if ~isreal(grad), options.LineSearchOptions.alpha = 0.5;
    else options.alpha = 1; end
end
if ~isfield(options,'beta'), options.beta = 'HSm'; end
if ~isfield(options,'LineSearch'), options.LineSearch = @ls_mt; end
if ~isfield(options,'LineSearchOptions')
    options.LineSearchOptions = struct;
end
if ~isfield(options.LineSearchOptions,'c2')
    options.LineSearchOptions.c2 = 0.1;
end
if ~isfield(options,'MaxIter'), options.MaxIter = 500; end
if ~isfield(options,'TolFun'), options.TolFun = 1e-6; end
if ~isfield(options,'TolX'), options.TolX = 1e-6; end

% Select the conjugate gradient update parameter.
switch options.beta
    % Hestenes-Stiefel.
    case 'HS',  beta = @(g,g1,p)real((g-g1)'*g)/real((g-g1)'*p);
    case 'HSm', beta = @(g,g1,p)max(0,real((g-g1)'*g)/real((g-g1)'*p));
    % Polak-Ribiere (equivalent to HS for exact line search).
    case 'PR',  beta = @(g,g1,~)real((g-g1)'*g)/(g1'*g1);
    case 'PRm', beta = @(g,g1,~)max(0,real((g-g1)'*g)/(g1'*g1));
    % Fletcher-Reeves (equivalent to PR for quadratic objective functions).
    case 'FR',  beta = @(g,g1,~)g'*g/(g1'*g1);
    % Steepest descent.
    case 'SD',  beta = @(~,~,~)0;
    otherwise,  beta = options.beta;
end

% Nonlinear conjugate gradient.
output.alpha = [];
output.fevals = 1;
output.fval = fval;
output.info = false;
output.infols = [];
output.iterations = 0;
while ~output.info

    % Compute the descent direction pncg.
    if output.iterations == 0
        pncg = -grad;
    else
        pncg = -grad+beta(grad,grad1,pncg)*pncg;
    end
    
    % Scale the initial step length.
    gp = real(grad'*pncg);
    if output.iterations > 0
        options.LineSearchOptions.alpha = output.alpha(end)*min(2,gp/g1p1);
    end
    g1p1 = gp;
    
    % Minimize f along z+alpha*pncg.
    [output.alpha(end+1),outputls] = options.LineSearch( ...
    f,g,z,pncg,output.fval(end),grad,options.LineSearchOptions);
    
    % Save current state and update step information.
    z = outputls.zkp1;
    grad1 = grad;
    grad = outputls.gkp1;
    
    % Update the output structure.
    output.fevals = output.fevals+outputls.fevals;
    output.fval(end+1) = outputls.fkp1;
    output.infols(end+1) = outputls.info;
    output.iterations = output.iterations+1;
    output.info = abs(diff(output.fval(end:-1:end-1))) <= ...
                  options.TolFun*output.fval(1);
    if norm(output.alpha(end)*pncg) <= options.TolX*norm(serialize(z))
        output.info = 2;
    end
    if output.iterations >= options.MaxIter, output.info = 3; end

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
