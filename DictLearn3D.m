function D = DictLearn3D(I0,Is,sparsity,factor)
% Is: patch size, Is = [Is1,Is2,Is3]
% sparsity: how spars the tensor is
% D: learn dictinoary
% factor: Overcomplete factor size of Dictionary D is [Is(k), factor*Is(k)]
% Using factor=1 means to have a square dictionary  D of size [Is(k),Is(k)]
% Example: D = DictLearn3D(I0, [2,3,4], 0.1, 2.0)

[S1,S2,S3] = size(I0);
Ns = 10000; % Number of samples
Niter = 100;
threshold = 10^-8;
K = round(sparsity*prod(Is)); % Number of nonzero entries
Ic = factor*Is; %Numer of columns in dictionaries

% Build 4D tensor of 3D patches
I = [Is,Ns];
Y = zeros(I);
for sample = 1:Ns
    i = round(rand*(S1-Is(1)))+1;
    j = round(rand*(S2-Is(2)))+1;
    k = round(rand*(S3-Is(3)))+1;

    Y(:,:,:,sample) = I0(i:i+Is(1)-1,j:j+Is(2)-1,k:k+Is(3)-1);
end
Y1 = zeros(Is(1),Is(2)*Is(3)*Ns);
Y2 = zeros(Is(2),Is(1)*Is(3)*Ns);
Y3 = zeros(Is(3),Is(1)*Is(2)*Ns);

% Built Y1
s = 1;
for b=1:Is(2)*Is(3):(Ns-1)*Is(2)*Is(3)+1
    patch = squeeze(Y(:,:,:,s));
    Y1(:,b:b+Is(2)*Is(3)-1) = reshape(patch,[Is(1),Is(2)*Is(3)]);
    s = s + 1;
end 
% Built Y2
s = 1;
for b=1:Is(1)*Is(3):(Ns-1)*Is(1)*Is(3)+1
    patch = squeeze(Y(:,:,:,s));
    Y2(:,b:b+Is(1)*Is(3)-1) = reshape(permute(patch,[2,1,3]),[Is(2),Is(1)*Is(3)]);
    s = s + 1;
end 
% Built Y3
s = 1;
for b=1:Is(1)*Is(2):(Ns-1)*Is(1)*Is(2)+1
    patch = squeeze(Y(:,:,:,s));
    Y3(:,b:b+Is(1)*Is(2)-1) = reshape(permute(patch,[3,1,2]),[Is(3),Is(1)*Is(2)]);
    s = s + 1;
end 

Y = tensor(Y);

% Initialization of dictionary
D{1} = normalize(randn(Is(1),round(factor*Is(1))),'norm');
D{2} = normalize(randn(Is(2),round(factor*Is(2))),'norm');
D{3} = normalize(randn(Is(3),round(factor*Is(3))),'norm');
B1 = zeros(Ic(1),Is(2)*Is(3)*Ns);
B2 = zeros(Ic(2),Is(1)*Is(3)*Ns);
B3 = zeros(Ic(3),Is(1)*Is(2)*Ns);
G = zeros([Ic,Ns]);
error(1) = inf;
delta = Inf;

iter = 1;
while (delta > threshold) 
    disp(['Iter=', num2str(iter), ' delta=', num2str(delta) ])
    D{1}=normalize(D{1},'norm');
    D{2}=normalize(D{2},'norm');
    D{3}=normalize(D{3},'norm');
    % compute sparse G
    Gram = kron(D{3}'*D{3},kron(D{2}'*D{2},D{1}'*D{1}));
    DT = kron(D{3},kron(D{2},D{1}));
    G = full(omp(DT'*reshape(double(Y),[prod(Is),Ns]),Gram,K));
    G = reshape(G,[Ic,Ns]);
    % compute error    
    Yap = ttm(tensor(G),{D{1},D{2},D{3}},[1,2,3]);
    error(iter+1) = norm(Y-Yap)/(Ns*prod(Is));
    delta = abs(error(iter+1) - error(iter));
    % update D1
    s = 1;
    for b=1:Is(2)*Is(3):(Ns-1)*Is(2)*Is(3)+1
        patchG = tensor(G(:,:,:,s));
        Taux = ttm(patchG,{D{2},D{3}},[2,3]);
        B1(:,b:b+Is(2)*Is(3)-1) = reshape(Taux,[Ic(1),Is(2)*Is(3)]);
        s = s + 1;
    end    
    D{1} = Y1*pinv(B1);
    %update D2
    s = 1;
    for b=1:Is(1)*Is(3):(Ns-1)*Is(1)*Is(3)+1      
        patchG = tensor(G(:,:,:,s));
        Taux = ttm(patchG,{D{1},D{3}},[1,3]);
        B2(:,b:b+Is(1)*Is(3)-1) = reshape(permute(Taux,[2,1,3]),[Ic(2),Is(1)*Is(3)]);
        s = s + 1;
    end
    D{2} = Y2*pinv(B2);   
    %update D3
    s = 1;
    for b=1:Is(1)*Is(2):(Ns-1)*Is(1)*Is(2)+1
        patchG = tensor(G(:,:,:,s));
        Taux = ttm(patchG,{D{1},D{2}},[1,2]);
        B3(:,b:b+Is(1)*Is(2)-1) = reshape(permute(Taux,[3,1,2]),[Ic(3),Is(1)*Is(2)]);
        s = s + 1;
    end
    D{3} = Y3*pinv(B3);    
    iter = iter + 1;
end

