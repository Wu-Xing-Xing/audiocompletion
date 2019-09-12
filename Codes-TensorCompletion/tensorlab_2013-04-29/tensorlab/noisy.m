function [Y,N] = noisy(X,SNR,options)
%NOISY Generate a noisy version of a given array.
%   [Y,N] = noisy(X,SNR) computes a noisy version of X as Y = X + N, where
%   the noise term N is generated as sigma*randn(size(X)) if X is real and
%   as sigma*(randn(size(X))+randn(size(X))*1i) if X is complex. The scalar
%   sigma is chosen such that 10*log10((X(:)'*X(:))/(N(:)'*N(:))) = SNR dB.
%   By default, SNR is 20. If X is a cell array, a noisy version of each of
%   its elements is computed and returned in the cell array Y.
%
%   noisy(X,SNR,options) may be used to set the following options:
%
%      options.Noise = 'auto' - A function handle to a function that
%                               generates an unscaled noise term N/sigma,
%                               given X. On 'auto', the function is
%                               @(X)randn(size(X)) if X is real and
%                               @(X)randn(size(X))+randn(size(X))*1i if X
%                               is complex.
%
%   See also randn, rand.

%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)

% Check the options structure.
isarray = ~iscell(X);
if nargin < 2, SNR = 20; end
if nargin < 3, options = struct; end
if ~isfield(options,'Noise') || strcmpi(options.Noise,'auto')
    options.Noise = @(x)randn(size(x))+(~isreal(x)*randn(size(x))*1i);
end

% Add noise to the (cell array of) array(s).
if isarray, X = {X}; end
N = cellfun(options.Noise,X,'UniformOutput',false);
addNoise = @(x,n)x+sqrt((x(:)'*x(:))*10^(-SNR/10)/(n(:)'*n(:)))*n;
Y = cellfun(addNoise,X,N,'UniformOutput',false);
if isarray, Y = Y{1}; end
