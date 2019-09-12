function [V,rank_vi] = optimize_V(X,Y,Z,V,ita,tsize,vsize,para_ST)
%% Initialization
num_V = numel(V);
num_L = size(para_ST.H,1);
rank_vi = zeros(1,num_V);

%% Update by linearization
for i = 1 : num_V
    list = 1:num_V;
    list(i) = [];
    Bi = reshape(shiftdim(TensorChainProductT(Z,V,list),i-1),vsize(i),[]);
    Ci = -Bi*(reshape(shiftdim(Y,i-1),tsize(i),[])+ita*reshape(shiftdim(X,i-1),tsize(i),[]))';
    Bi = (Bi*Bi')*ita/2;
    
    Ai = zeros(tsize(i));
    for j = 1  : num_L
        if para_ST.VSet{j}(i)
            list = 1:num_V;
            list = list(para_ST.VSet{j});
            for k = 1 : numel(list)
                U{k} = V{list(k)}*para_ST.Ds{j,list(k)};
            end
            idx = find(list==i);
            list = 1:numel(list);
            list(idx) = [];
            if para_ST.Rate(j)==1, lidx=1; else lidx=idx; end;
            T = reshape(shiftdim(TensorChainProduct(para_ST.H{j,lidx},U,list),idx-1),tsize(i),[]);
            T = T*T';
            Ai = Ai+(T+T')/2;
        end
    end
    sig = norm(Ai+Ai')+norm(Bi+Bi');
    grad = V{i}*(Ai+Ai')+(Bi+Bi')*V{i}+Ci;
    [Vi r] = optimize_LRM(V{i},grad,sig,para_ST.alpha(i));
    V{i} = Vi;
    rank_vi(i) = r;
end

function [optimum r] = optimize_LRM(Mk,grad,sig,alpha)
%% M = arg min { alpha*|| M ||_* + <grad(Mk),M-Mk> + (sig/2)*|| M-Mk ||_F }
[U S V] = svd(Mk-grad/(sig),'econ');
S = diag(S);
[S r] = shrinkage(S,alpha/(sig));
if r==0
    optimum = Mk;
else
    optimum = U*diag(S)*V';
end