function T = btdLx1gen(U,L)
%BTDLX1GEN Generate full tensor given a (rank-L(r) x rank-1) BTD.
%   T = btdLx1gen(U,L) computes the tensor T as the sum of R (rank-L(r) x
%   rank-1) tensors defined by the columns of the factor matrices U{n} and
%   the ranks L(r).
%
%   See also cpdgen, lmlragen.

%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)

R = sum(L);
E = full(sparse(sum(bsxfun(@gt,1:R,cumsum(L)'),1)+1,1:R,1));
for n = find(cellfun('size',U(:).',2) == size(E,1)), U{n} = U{n}*E; end
T = reshape(U{1}*kr(U(end:-1:2)).',cellfun('size',U(:).',1));
