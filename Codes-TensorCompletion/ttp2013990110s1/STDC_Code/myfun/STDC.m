function [V,Core,info,X,itr,t] = STDC(X,mark,para_ST,mode,Xg)
%% Simultaneous Tensor Decomposition and Completion
% Input:
%  -- X: an input N-th order tensor object
%  -- mark: boolean index of missing entries (1 indicates missing; 0 otherwise)
%  -- para_ST: parameters used for the proposed STDC algorithm
%  -- mode: determining whether the N-th submanifold (V_N, usually ignored in multilinear model analysis) is computed (set as 0) or not (set as 1)
%  -- Xg: the ground truth of X; if not available, just using X instead

% Parameters of para_ST:
%  -- maxitr: maximum iteration
%  -- ita_rate: increased penalty of equality constraint
%  -- tau: the 1st threshold in shrinkage operation
%  -- kappa: the weight of MGE
%  -- omega: (||Z||_F)^2 ~= omega*(ita*(||X-Zx1V1x2...xnVn||_F)^2)/2
%  -- print_mode: determining the message printed iteratively (1) or finally (0)
%  -- Rate: an Mx1 vector, which indicates the downsampling rate of M multilinear graphs
%  -- VSet: an Mx1 cell-array; each element is an Nx1 boolean vector, which indicates the L-dependent subset of {V_1,...,V_N}
%  -- H: an MxK cell-array (K is the maximum of # submanifolds in M subsets);
%        each element denotes a tensor, whose unfolding matrix is the 
%        cholskey decomposition of the corresponding Laplacian matrix (H'H = L);
%        if the m-th graph is not downsampled, users only have to specify
%        H{m,1}; otherwise, user have to specify H{m,1} to H{m,k} for the
%        k (k<=K) submanifolds in the m-th subset

%% Initialization
if isfield(para_ST,'maxitr')==0, maxitr=50; else maxitr=para_ST.maxitr; end;
if isfield(para_ST,'ita_rate')==0, ita_rate=1.1; else ita_rate=para_ST.ita_rate; end;
if isfield(para_ST,'print_mode')==0, print_mode = boolean(0); else print_mode=para_ST.print_mode; end;
if isfield(para_ST,'tau')==0, tau=0.1; else tau=para_ST.tau; end;
if isfield(para_ST,'kappa')==0, kappa=10; else kappa=para_ST.kappa; end;
if isfield(para_ST,'omega')==0, omega=10^-3; else omega=para_ST.omega; end;

tsize = size(X);
if isfield(para_ST,'H')==0
    para_ST.H = [];
else
    for i = 1 : size(para_ST.H,1)
        if size(para_ST.H,2)==1 || numel(para_ST.H{i,2})==0
            for j = 1 : numel(tsize)
                para_ST.Ds{i,j} = eye(tsize(j));
            end
        else
            for j = 1 : numel(tsize)
                A = randn(tsize(j)); 
                B = imresize(A,[tsize(j) round(tsize(j)*para_ST.Rate(i))],'bilinear');
                para_ST.Ds{i,j} = A\B;
            end
        end
    end
end

vsize = tsize;
N = numel(tsize);
if mode==1, N = N-1; end;
for i =1 : N
    V{i} = eye(tsize(i));
    vsize(i) = size(V{i},1);
end
Z = X;
Y = zeros(tsize);
norm_gt = norm(Xg(:));
norm_x = norm(X(:));

para_ST.alpha = ones(N,1);
para_ST.gamma = (omega/tau)/(norm_x^2);
xxt = reshape(X,tsize(1),[]);
xxt = norm(xxt*xxt');
ita = 1/(tau*xxt);

ct = zeros(1,numel(tsize));
for i = 1 : size(para_ST.H,1)
    ct = ct+double(para_ST.VSet{i});
end
for i = 1 : size(para_ST.H,1)
    list = 1 : N;
    list = list(para_ST.VSet{i});
    for j = 1 : size(para_ST.H,2)
        U{j} = para_ST.Ds{i,j};
    end
    for j = 1 : size(para_ST.H,2)
        llt = reshape(TensorChainProduct(para_ST.H{i,j},U,[1:j-1 j+1:size(para_ST.H,2)]),tsize(list(j)),[]);
        llt = norm(llt*llt');
        para_ST.H{i,j} = para_ST.H{i,j}*kappa*sqrt(ita*xxt/(2*llt*ct(list(j))));
    end
end
%% Main algorithm
tic
for itr = 1 : maxitr
    % update V1,...,Vn
    [V,rank_vi] = optimize_V(X,Y,Z,V,ita,tsize,vsize,para_ST);
    
    % update Z
    Z = optimize_Zt(Z,V,X,Y,ita,para_ST.gamma);
    
    % update X
    Xt = TensorChainProductT(Z,V,1:numel(V));
    X(mark) = Xt(mark)-Y(mark)/ita;
    residual = norm(X(:)-Xt(:));
    
    % update Y
    Y = Y+ita*(X-Xt);
    
    % assessment
    info.rse(itr) = norm(X(mark)-Xg(mark))/norm_gt;
    info.rank_vi(:,itr) = rank_vi;
    info.residual(:,itr) = residual;
    
    if print_mode, fprintf(['STDC completed at ',int2str(itr),'-th iteration step within ',num2str(toc),' seconds...\n']); end;
    if residual<10^-4*norm_x, break; end;
    ita = ita*ita_rate;
end
if ~print_mode, t=toc;fprintf(['STDC completed at ',int2str(itr),'-th iteration step within ',num2str(t),' seconds...\n']); end;
Core = Z;
for i = 1 : N
    V{i} = V{i}';
end
