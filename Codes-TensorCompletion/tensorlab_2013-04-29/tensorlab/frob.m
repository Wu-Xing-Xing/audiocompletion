function f = frob(T)
%FROB Frobenius norm.
%   frob(T) returns the Frobenius norm of the tensor T.
%
%   See also norm.

%   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
%            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)

f = norm(T(:));
