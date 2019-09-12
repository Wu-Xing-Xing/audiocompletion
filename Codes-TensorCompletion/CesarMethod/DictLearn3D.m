function D = DictLearn3D(I0,Is,sparsity)
% Is: patch size
% sparsity: how spars the tensor is
% D: learn dictinoary

[S1,S2,S3] = size(I0);
Ns = 10000; % Number of samples
Niter = 100;
threshold = 10^-8;
K = round(sparsity*Is*Is*Is); % Number of nonzero entries

% Build 4D tensor of 3D patches
I = [Is,Is,Is,Ns];
Y = zeros(I);
for sample = 1:Ns
    i = round(rand*(S1-Is))+1;
    j = round(rand*(S2-Is))+1;
    k = round(rand*(S3-Is))+1;

    Y(:,:,:,sample) = I0(i:i+Is-1,j:j+Is-1,k:k+Is-1);
end
Y1 = zeros(Is,Is*Is*Ns);
Y2 = zeros(Is,Is*Is*Ns);
Y3 = zeros(Is,Is*Is*Ns);
s = 1;
for b=1:Is*Is:(Ns-1)*Is*Is+1
    patch = squeeze(Y(:,:,:,s));
    Y1(:,b:b+Is*Is-1) = reshape(patch,[Is,Is*Is]);
    Y2(:,b:b+Is*Is-1) = reshape(permute(patch,[2,1,3]),[Is,Is*Is]);
    Y3(:,b:b+Is*Is-1) = reshape(permute(patch,[3,1,2]),[Is,Is*Is]);
    s = s + 1;
end    
Y = tensor(Y);

% Initialization of dictionary
D{1} = normalize(randn(Is));
D{2} = normalize(randn(Is));
D{3} = normalize(randn(Is));
B1 = zeros(Is,Is*Is*Ns);
B2 = zeros(Is,Is*Is*Ns);
B3 = zeros(Is,Is*Is*Ns);
G = zeros(Is,Is,Is,Ns);
error(1) = inf;
delta = Inf;

iter = 1;
while (delta > threshold)  
    D{1}=normalize(D{1});
    D{2}=normalize(D{2});
    D{3}=normalize(D{3});
    % compute sparse G
    Gram = kron(D{3}'*D{3},kron(D{2}'*D{2},D{1}'*D{1}));
    DT = kron(D{3},kron(D{2},D{1}));
    G = full(omp(DT'*reshape(double(Y),[Is*Is*Is,Ns]),Gram,K));
    G = reshape(G,[Is,Is,Is,Ns]);
    % compute error    
    Yap = ttm(tensor(G),{D{1},D{2},D{3}},[1,2,3]);
    error(iter+1) = norm(Y-Yap)/(Ns*Is*Is*Is);
    delta = abs(error(iter+1) - error(iter));
    % update D1
    s = 1;
    for b=1:Is*Is:(Ns-1)*Is*Is+1
        patchG = tensor(G(:,:,:,s));
        Taux = ttm(patchG,{D{2},D{3}},[2,3]);
        B1(:,b:b+Is*Is-1) = reshape(Taux,[Is,Is*Is]);
        s = s + 1;
    end    
    D{1} = Y1*pinv(B1);
    %update D2
    s = 1;
    for b=1:Is*Is:(Ns-1)*Is*Is+1      
        patchG = tensor(G(:,:,:,s));
        Taux = ttm(patchG,{D{1},D{3}},[1,3]);
        B2(:,b:b+Is*Is-1) = reshape(permute(Taux,[2,1,3]),[Is,Is*Is]);
        s = s + 1;
    end
    D{2} = Y2*pinv(B2);   
    %update D3
    s = 1;
    for b=1:Is*Is:(Ns-1)*Is*Is+1
        patchG = tensor(G(:,:,:,s));
        Taux = ttm(patchG,{D{1},D{2}},[1,2]);
        B3(:,b:b+Is*Is-1) = reshape(permute(Taux,[3,1,2]),[Is,Is*Is]);
        s = s + 1;
    end
    D{3} = Y3*pinv(B3);    
    iter = iter + 1;
end

