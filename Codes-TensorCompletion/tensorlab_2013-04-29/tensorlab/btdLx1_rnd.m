function [U,output] = btdLx1_rnd(size_tens,L,Q,options)
%BTDLX1_RND Pseudorandom initialization for (rank-L(r) x rank-1) BTD.
%   U = btdLx1_rnd(size_tens,L,Q) generates pseudorandom factor matrices
%   U{1}, ..., U{N} that can be used to initialize algorithms that compute
%   the (rank-L(r) x rank-1) block term decomposition of an N-th order
%   tensor of size size_tens. The vector Q defines a list of modes which
%   correspond to the rank-1 part of each term. For example, Q == 3 &&
%   length(size_tens) == 3 generates factor matrices for a
%   rank-(L(r),L(r),1) BTD.
%
%   btdLx1_rnd(T,L,Q) is shorthand for btdLx1_rnd(size(T),L,Q) if T is
%   Real. If T is complex, then U{n} will generated as complex matrices
%   by default (cf. options).
%
%   btdLx1_rnd(size_tens,L,Q,options) and btdLx1_rnd(T,L,Q,options) may be
%   used to set the following options:
%
%      options.Real =        - The type of random number generator used to
%      [{@randn}|@rand|0]      generate the real part of each factor
%                              matrix. If 0, there is no real part.
%      options.Imag =        - The type of random number generator used to
%      [@randn|@rand|0|...     generate the imaginary part of each factor
%       {'auto'}]              matrix. If 0, there is no imaginary part.
%                              On 'auto', options.Imag is 0 unless the
%                              first argument is a complex tensor T, in
%                              which case it is equal to options.Real.
%      options.Orth =        - If true, the generated factor matrices are
%      [true|false|{'auto'}]   orthogonalized using a QR factorization.
%                              On 'auto', options.Orth is false if the
%                              first argument is a vector size_tens, and
%                              true if the first argument is a tensor T.
%
%   See also cpd_rnd, lmlra_rnd, btdLx1gen.

%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)

% Process input.
isSizeVector = isvector(size_tens);
if ~isSizeVector, T = size_tens; size_tens = size(size_tens); end
N = length(size_tens);

% Check the options structure.
isfunc = @(f)isa(f,'function_handle');
if nargin < 4, options = struct; end
if ~isfield(options,'Real'), options.Real = @randn; end
if ~isfunc(options.Real), options.Real = @zeros; end
if ~isfield(options,'Imag'), options.Imag = 'auto'; end
if ischar(options.Imag) && strcmpi(options.Imag,'auto')
    if ~isSizeVector && ~isreal(T)
        options.Imag = options.Real;
    else
        options.Imag = 0;
    end
end
if ~isfield(options,'Orth'), options.Orth = 'auto'; end
if ischar(options.Orth) && strcmpi(options.Orth,'auto')
	options.Orth = ~isSizeVector;
end

% Generate factor matrices.
R = sum(L);
U = arrayfun(@(n)options.Real(size_tens(n),R),1:N,'UniformOutput',0);
if isfunc(options.Imag)
    Ui = arrayfun(@(n)options.Imag(size_tens(n),R),1:N,'UniformOutput',0);
    U = cellfun(@(ur,ui)ur+1i*ui,U,Ui,'UniformOutput',0);
end
for n = Q
    Un = U{n};
    U{n} = Un(:,1:length(L));
end
for n = 1:N*options.Orth
    if size(U{n},1) >= size(U{n},2), [U{n},~] = qr(U{n},0);
    else [Q,~] = qr(U{n}.',0); U{n} = Q.'; end
end
output = struct;
