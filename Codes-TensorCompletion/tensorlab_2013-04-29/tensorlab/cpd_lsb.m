function [alpha,output] = cpd_lsb(T,U,dU,state,~)
%CPD_LSB CPD line search by Bro.
%   [alpha,output] = cpd_lsb(T,U,dU,state) computes a line search parameter
%   alpha as i^(1/3), where i is the current iterate in the parent
%   algorithm. Requires state.iterations.

%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)
%
%   References:
%   [1] R. Bro, "Multi-way analysis in the food industry: models,
%       algorithms, and applications," PhD thesis, University of
%       Amsterdam and Royal Veterinary and Agricultural University, 1998.

% Check the tensor T.
N = ndims(T);
if N < 3, error('cpd_lsb:T','ndims(T) should be >= 3.'); end

% Check the factor matrices U and dU.
if length(U) ~= N || length(dU) ~= N
    error('cpd_lsb:U','length((d)U) should equal ndims(T).');
end
R = size(U{1},2);
if any(cellfun('size',U,2) ~= R) || any(cellfun('size',dU,2) ~= R)
    error('cpd_lsb:U','size((d)U{n},2) should be the same for all n.');
end
if any(cellfun('size',U(:).',1) ~= size(T)) || ...
   any(cellfun('size',dU(:).',1) ~= size(T))
    error('cpd_lsb:U','size(T,n) should equal size((d)U{n},1).');
end

% Check the state structure.
if nargin < 4 || ~isstruct(state), state = struct; end
if ~isfield(state,'iterations')
    error('cpd_lsb:state','State must contain the field "iterations".');
end

% Compute the step length and objective function value.
alpha = (state.iterations+1)^(1/3);
output = struct;
