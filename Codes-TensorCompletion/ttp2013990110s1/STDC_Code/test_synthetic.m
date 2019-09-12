%% Validation of Synthetic Data in Section 4.1
clear;clc;
addpath 'myfun';
addpath 'tensor_basicfun';
load('ex1.mat');

m_rate = 0.6;
rand('seed',0);%rng(0);
idx = randperm(numel(X));
mark = zeros(tsize);
mark(idx(1:floor(m_rate*numel(X)))) = 1;
mark = boolean(mark);
Xm = X;
Xm(mark) = 0;

%% STDC-Lx
para_Lx.print_mode = boolean(0);
para_Lx.maxitr = 100;
para_Lx.tau = 0.1;
para_Lx.omega = 10^0.4;
[~,~,info_Lx,~] = STDC(Xm,mark,para_Lx,0,X);
%% STDC-L1
para_L1.print_mode = boolean(0);
para_L1.maxitr = 100;
para_L1.tau = 0.1;
para_L1.kappa = 10^0.4;
para_L1.omega = 10^-1;
para_L1.H{1,1} = L1;
para_L1.H{2,1} = L2;
para_L1.H{3,1} = L3;
para_L1.VSet{1} = boolean([1 0 0]);
para_L1.VSet{2} = boolean([0 1 0]);
para_L1.VSet{3} = boolean([0 0 1]);
para_L1.Rate = [1 1 1]';
[~,~,info_L1,~] = STDC(Xm,mark,para_L1,0,X);
%% STDC-L2
para_L2.print_mode = boolean(0);
para_L2.maxitr = 100;
para_L2.tau = 0.1;
para_L2.kappa = 10^0.6;
para_L2.omega = 10^-1.4;
para_L2.H{1,1} = L12;
para_L2.H{2,1} = L23;
para_L2.H{3,1} = L13;
para_L2.VSet{1} = boolean([1 1 0]);
para_L2.VSet{2} = boolean([0 1 1]);
para_L2.VSet{3} = boolean([1 0 1]);
para_L2.Rate = [1 1 1]';
[~,~,info_L2,~] = STDC(Xm,mark,para_L2,0,X);
%% STDC-L3
para_L3.print_mode = boolean(0);
para_L3.maxitr = 100;
para_L3.tau = 0.1;
para_L3.kappa = 10^1.7;
para_L3.omega = 10^-2.5;
para_L3.H{1,1} = L;
para_L3.VSet{1} = boolean([1 1 1]);
para_L3.Rate = [1]';
[~,~,info_L3,~] = STDC(Xm,mark,para_L3,0,X);
%% STDC-Ls
para_Ls.print_mode = boolean(0);
para_Ls.maxitr = 100;
para_Ls.tau = 0.1;
para_Ls.kappa = 10^1.5;
para_Ls.omega = 10^-2.5;
para_Ls.H{1,1} = Ls{1};
para_Ls.H{1,2} = Ls{2};
para_Ls.H{1,3} = Ls{3};
para_Ls.VSet{1} = boolean([1 1 1]);
para_Ls.Rate = [1/2]';
[~,~,info_Ls,~] = STDC(Xm,mark,para_Ls,0,X);
%% RSE
figure;
plot(info_Lx.rse,'k.-');hold on;
plot(info_L1.rse,'r.-');
plot(info_L2.rse,'g.-');
plot(info_L3.rse,'b.-');
plot(info_Ls.rse,'c.-');hold off;
xlabel('iteration');ylabel('RSE');
legend('STDC-Lx','STDC-L1','STDC-L2','STDC-L3','STDC-Ls');