function [U,output] = cpds_minf(T,U0,options)
%CPDS_MINF Structured/symmetric CPD by unconstrained nonlinear optimization.
%   [U,output] = cpds_minf(T,U0) computes the factor matrices U{1}, ...,
%   U{N} belonging to a structured canonical polyadic decomposition of the
%   N-th order tensor T by minimizing 0.5*frob(T-cpdgen(U))^2. The tensor T
%   may contain missing elements, stored as NaN or Inf. The algorithm is
%   initialized with the factor matrices U0{n}. This method can also
%   compute other types of decompositions such as rank-(Lr,Lr,1)
%   block term decompositions by imposing structure on U{n}, see below.
%
%   The cell array options.Symmetry should contain the integers 1:N,
%   partitioned into vectors. The partitioning defines which factor
%   matrices are imposed to be equal to each other. For example,
%   options.Symmetry = {[1 2],3} imposes that the first factor matrix is
%   equal to the second factor matrix, and options.Symmetry = {[1 2 3]}
%   implies a fully symmetric CPD. When symmetry is imposed, the factor
%   matrix U0{n} corresponds to the factor matrices belonging to the modes
%   options.Symmetry{n}. In other words, the number of initial factor
%   matrices length(U0) should equal length(options.Symmetry).
%
%   The cell array options.Structure has length N and decouples the
%   structure in U{n} from the variables that generate U{n}. If there is no
%   structure in U{n}, set options.Structure{n} = []. Structure may be
%   imposed on U using (a combination of) two ways:
%
%   1. Structured variables (symmetric, Toeplitz, Hankel, ...).
%
%      For structured matrices U{n} in which each entry is either a
%      variable or a constant, collect the variables into a vector z. For
%      example, U{n} may have the Toeplitz structure [a b c; 1 a b; 2 1 a],
%      which is generated by the variables z = [a; b; c]. To impose such a
%      structure on U{n}, set options.Structure{n} = [1 2 3; 0 1 2; 0 0 1],
%      where 1, 2 and 3 refer to the first, second and third element in z
%      and a 0 at position (i,j) indicates that U{n}(i,j) is constant and
%      equal to U0{n}(i,j). If there are no constants, U0{n} may also be
%      given as a generator vector z0, in this case of the form [a; b; c].
%
%   2. Structured expressions (Hermitian, Vandermonde, Cauchy, ...).
%
%      Method 1: structured analytic expressions (Vandermonde, Cauchy).
%      If the structure in U{n} does not depend on complex conjugates of
%      its variables z, set options.Structure{n} = {S,J}, where S(z) builds
%      the matrix U{n} given the data z, and J(z) returns the Jacobian
%      d(vec(U{n}))/dz^T at z. For example, if U{n} has a Vandermonde
%      structure [1 a a^2; 1 b b^2] defined by the two variables
%      z = [a; b], then set options.Structure{n} = {@(z)[[1; 1] z z.^2],
%      @(z)[zeros(2); eye(2); 2*diag(z)]}. Or set options.Structure{n} =
%      {S,'Jacobian'} to automatically approximate the Jacobian with finite
%      differences using deriv. The initialization U0{n} must be supplied
%      as a generator vector z0, in this case of the form [a; b].
%
%      Method 2: structured nonanalytic expressions (Hermitian).
%      If the structure depends on the complex conjugates of its variables,
%      set options.Structure{n} = {S,J}, where S(z) builds the matrix U{n}
%      given z, and J(z) returns the complex Jacobian
%      [d(vec(U{n}))/dz^T d(vec(U{n}))/d(conj(z))^T] at z. In the complex
%      Jacobian, the partial derivative w.r.t. z (conj(z)) treats conj(z)
%      (z) as constant. For example, if U{n} has the Hermitian structure
%      [0.5*(a+conj(a)) conj(b); b 1], then set options.Structure{n} = ...
%      {@(z)[real(z(1)) conj(z(2)); z(2) 1], @(z)[0.5 0 0.5 0; 0 1 0 0; ...
%      0 0 0 1; 0 0 0 0]}. Or set options.Structure{n} = {S,'Jacobian-C'}
%      to automatically approximate the complex Jacobian with finite
%      differences using deriv. The initialization U0{n} must be supplied
%      as a generator vector z0, in this case of the form [a; b].
%
%   The structure output returns additional information:
%
%      output.Name  - The name of the selected algorithm.
%      output.<...> - The output of the selected algorithm.
%
%   cpds_minf(T,U0,options) may be used to set the following options:
%
%      options.Algorithm =     - The desired optimization method.
%      [{@minf_lbfgsdl}|...
%       @minf_lbfgs|@minf_ncg]
%      options.<...>           - Parameters passed to the selected method,
%                                e.g., options.TolFun and options.TolX.
%                                See also help [options.Algorithm].
%   See also cpds_nls.

%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)
%
%   References:
%   [1] L. Sorber, M. Van Barel, L. De Lathauwer, "A framework for
%       decoupling structure from parameters in tensor decompositions,"
%       ESAT-SISTA Internal Report 13-12, KU Leuven, 2013.
%   [2] L. Sorber, M. Van Barel, L. De Lathauwer, "Optimization-based
%       algorithms for tensor decompositions: canonical polyadic
%       decomposition, decomposition in rank-(Lr,Lr,1) terms and a new
%       generalization," SIAM J. Opt., 2013.
%   [3] L. Sorber, M. Van Barel, L. De Lathauwer, "Unconstrained
%       optimization of real functions in complex variables," SIAM J. Opt.,
%       Vol. 22, No. 3, 2012, pp. 879-898.

% Check the tensor T.
N = ndims(T);
if N < 3, error('cpds_minf:T','ndims(T) should be >= 3.'); end
size_tens = size(T);
W = ~isfinite(T);
hasMissingEntries = any(W(:));
if hasMissingEntries, T(W) = 0; end;

% Check the options structure.
xsfunc = @(f)isa(f,'function_handle')&&exist(func2str(f),'file');
if nargin < 3, options = struct; end
if ~isfield(options,'Algorithm')
    funcs = {@minf_lbfgsdl,@minf_lbfgs,@minf_ncg};
    options.Algorithm = funcs{find(cellfun(xsfunc,funcs),1)};
end
if ~isfield(options,'Symmetry') || isempty(options.Symmetry)
    options.Symmetry = num2cell(1:N);
end
if ~isfield(options,'Structure') || isempty(options.Structure)
    options.Structure = cell(1,length(options.Symmetry));
end
if ~isfield(options,'TolFun'), options.TolFun = 1e-12; end

% Check options.Symmetry and options.Structure.
sym = options.Symmetry(:).';
str = options.Structure(:).';
isStructVar = cellfun(@(s)isnumeric(s)&&~isempty(s),str);
isStructExp = cellfun(@(s)iscell(s)&&~isempty(s),str);
isStructVarWithConst = false(1,length(sym));
if length(cell2mat(sym)) ~= N || ~all(sort(cell2mat(sym)) == 1:N)
    error('cpds_minf:Symmetry',['options.Symmetry should be a cell ' ...
                                'array containing the integers 1:N.']);
end
if length(str) ~= length(sym)
    error('cpds_minf:Structure',['options.Structure should have length '...
                                 'N or length(options.Symmetry).']);
end
if ~all(cellfun(@(i)all(size_tens(i) == size_tens(i(1))),sym))
    error('cpds_minf:Symmetry',['options.Symmetry is not compatible ' ...
                                'with size(T).']);
end

% Check the initial factor matrices U0.
U = U0(:).';
if length(U) ~= length(sym)
   error('cpds_minf:U0',['U0 should have length N or ' ...
                         'length(options.Symmetry).']);
end

% Extract factor matrices' constant Jacobians and data vectors from U0.
Jconst = cell(1,length(sym));
for i = 1:length(str)
    stri = str{i};
    if isStructVar(i) && ~all(stri(:) == floor(stri(:)))
        error('cpds_minf:Structure', ...
              ['option.Structure{%i} may only contain nonnegative ' ...
               'integers.'],i);
    end
    if isStructExp(i) && (length(stri) ~= 2 || ...
       ~any(cellfun(@(f)isa(f,'function_handle'),stri)))
        error('cpds_minf:Structure', ...
              'option.Structure{%i} must contain function handles.',i);
    end
    if isStructVar(i)
        nz = stri(:) ~= 0;
        Jconst{i} = sparse(find(nz),stri(nz),ones(sum(nz),1), ...
            numel(stri),max(stri(:)));
        isStructVarWithConst(i) = ~all(nz);
        if size(Jconst{i},1) == numel(U0{i})
            Ui = zeros(size(Jconst{i},2),1);
            Ui(stri(nz)) = reshape(U0{i}(nz),[],1);
            U{i} = Ui;
        elseif isStructVarWithConst(i)
            error('cpds_minf:U0', ...
                  ['option.Structure{%i} refers to constants in ' ...
                   'U0{%i}, however U0{%i} is of the wrong size.'],i,i,i);
        elseif min(stri(:)) ~= 1 || max(stri(:)) ~= numel(U0{i})
            error('cpds_minf:U0', ...
                  ['option.Structure{%i} refers to erroneous indices ' ...
                   'in U0{%i}.'],i,i);
        end
    elseif isStructExp(i) && isnumeric(stri{2})
        Jconst{i} = stri{2};
    elseif isStructExp(i) && ischar(stri{2})
        str{i}{2} = @(z)deriv(stri{1},z,[],stri{2});
    end
end

% Cache some intermediate variables.
size_U = cellfun(@numel,U);
sym2full = zeros(1,N);
for i = 1:length(sym), sym2full(sym{i}) = i; end
M = arrayfun(@(n)tens2mat(T,n),1:N,'UniformOutput',false);

% Call the optimization method.
[U,output] = options.Algorithm(@f,@g,U,options);
output.Name = func2str(options.Algorithm);

function U = full(U)
% Convert structured CPD to full CPD.
    idx = 1:length(sym);
    for n = idx(isStructVar)
        if isStructVarWithConst(n)
            Un = U0{n};
            Un(str{n} ~= 0) = U{n}(str{n}(str{n} ~= 0));
            U{n} = Un;
        else
            U{n} = U{n}(str{n});
        end
    end
    for n = idx(isStructExp)
        U{n} = str{n}{1}(U{n});
    end
    U = U(sym2full);
end

function JU = J(U)
% Retrieve (complex) Jacobians of structured factor matrices.
    JU = Jconst;
    idx = 1:length(sym);
    for n = idx(isStructExp & cellfun(@isempty,JU))
        JUn = str{n}{2};
        JU{n} = JUn(U{n});
    end
end

function fval = f(U)
% Objective function.
    U = full(U);
    D = M{1}-U{1}*kr(U(end:-1:2)).';
    if hasMissingEntries, D(W(:)) = 0; end
    fval = 0.5*(D(:)'*D(:));
end

function grad = g(U)
% Objective function's gradient.

    % Compute each factor matrix' Jacobian and convert U to a full CPD.
    JU = J(U);
    U = full(U);
    
    % Precompute some intermediate results.
    grad = cell(1,length(sym));
    if hasMissingEntries
        D = reshape(U{1}*kr(U(end:-1:2)).',size_tens)-T;
        D(W) = 0;
    else
        UHU = zeros(size(U{1},2),size(U{1},2),N);
        for n = 1:N, UHU(:,:,n) = U{n}'*U{n}; end
    end
    
    % Compute the gradient.
    for m = 1:length(sym)
        for n = sym{m}
            
            % Compute the full CPD gradient.
            if hasMissingEntries
                gradn = tens2mat(D,n)*conj(kr(U([N:-1:n+1 n-1:-1:1])));
            else
                G1 = U{n}*conj(prod(UHU(:,:,[1:n-1 n+1:N]),3));
                G2 = M{n}*conj(kr(U([N:-1:n+1 n-1:-1:1])));
                gradn = G1-G2;
            end
            
            % Take into account the factor matrix' structure.
            if size(JU{m},2) == size_U(m)
                gradn = JU{m}'*gradn(:);
            elseif size(JU{m},2) == 2*size_U(m)
                Jm = JU{m};
                gradn = Jm(:,1:end/2)'*gradn(:)+ ...
                        conj(Jm(:,end/2+1:end)'*gradn(:));
            end
            if isempty(grad{m}), grad{m} = gradn;
            else grad{m} = grad{m}+gradn;
            end
            
        end
    end
    
end

end
