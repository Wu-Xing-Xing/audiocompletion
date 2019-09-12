%% Validation of Image "Lena" in Section 4.3
clear;clc;
addpath 'myfun';
addpath 'tensor_basicfun';
X = double(imread('lena.bmp'))/255;
tsize = size(X);

% m_rate = 0.8;
% rand('seed',0);%rng(0);
% idx = randperm(numel(X));
% mark = zeros(tsize);
% mark(idx(1:floor(m_rate*numel(X)))) = 1;
% mark = boolean(mark);

load('E:\PhD\Experiments\Simultaneous Tensor Decomposition and Completion\data\image\mark_m60.mat');
Xm = X;
Xm(mark) = 0;
%% Construction of Factor-related Graph
W1 = zeros(tsize(1));
W2 = zeros(tsize(2));
for i = 1 : tsize(1)
    for j = 1 : tsize(1)
        W1(i,j) = exp(-abs(i-j));
    end
end
for i = 1 : tsize(2)
    for j = 1 : tsize(2)
        W2(i,j) = exp(-abs(i-j));
    end
end
L1 = diag(sum(W1,2))-W1;
L2 = diag(sum(W2,2))-W2;
L1 = cholcov(L1)';
L2 = cholcov(L2)';
%% Image Completion
para_ST.print_mode = boolean(1);
para_ST.maxitr = 50;
para_ST.tau = 0.1;
para_ST.kappa = 10^0.2;
para_ST.omega = 10^-1;
para_ST.H{1,1} = L1;
para_ST.H{2,1} = L2;
para_ST.VSet{1} = boolean([1 0 0]);
para_ST.VSet{2} = boolean([0 1 0]);
para_ST.Rate = [1 1];

[~,~,info,Xr] = STDC(Xm,mark,para_ST,1,X);
figure;plot(info.rse);xlabel('iteration');ylabel('RSE');
figure;
subplot(1,2,1);imshow(Xm);xlabel('incomplete image');
subplot(1,2,2);imshow(Xr);xlabel('our result');


