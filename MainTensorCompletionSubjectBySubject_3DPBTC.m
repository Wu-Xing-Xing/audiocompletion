% EEG tensor completion, subject by subject
% BCI data from Qibin Zhao
% 
% Fitxers revisats i correctes
% RIKEN - 2017
%
% Jordi Sol?-Casals

clear all
close all
clc
addpath(genpath(pwd));


% load data (all subjects except the best one, used for tunning all the parameters of the algorithms)
nomSub = {'SubA_6chan_2LR_s1','SubA_6chan_2LR_s2','SubC_6chan_2LR_s1','SubC_6chan_2LR_s2','SubC_6chan_2LR_s3','SubC_6chan_2LR_s4','SubF_6chan_2LR','SubG_6chan_2LR','SubH_6chan_2LR'};
num_subjects = size(nomSub,2);
nom_Mask = {'Mask1_001', 'Mask1_005', 'Mask1_01', 'Mask1_015', 'Mask1_02'}; % we have 1%, 5%, 10%, 15% and 20% of missing fibers
num_mask = size(nom_Mask,2);

sparsity = 0.1; % obtimal value obtained with the best subject: SubC_6chan_2LR_s5 
ps = 4;
load matinput     

for tt = 1:1 % run over subjects/sessions
    
    %load (nomSub{tt})
    DIM = size(EEGDATA);
    
    % original data
    X = EEGDATA;
    itemmax = max(X(:)); 
    X = X/max(X(:));
    
    for m = 3:3
        % mask for this subject
        % nomM = [nom_Mask{m},'_', nomSub{tt},];
        nomM = [nom_Mask{m},'r_', nomSub{tt},]; % totally random
        %load (nomM); 
        
        % observed data
        Y = X.*O;
        
        D = DictLearn3D(Y,ps,sparsity)
    
        tStart = tic;
        [X_rec] = CesarTC(X,O,ps,D,sparsity);
        % We maintain the original (non-missing) entries...
        X_rec = X.*O+X_rec.*not(O); 
        % Calculation time
        Temps(m,1) = toc(tStart);
        % Performance
        err = X_rec(:) - X(:);
        rrse = sqrt(sum(err(O==0).^2)/sum(X(O==0).^2))
        Performance(m,tt) = rrse;
        
        X = X*itemmax;
        Y = X.*O;
        X_hat1 = X_rec*itemmax;
        save matoutput X Y X_hat1 rrse;
        
        % save results
        % nom_fitxer = ['New_2015_3DPBTC_resultats_',nomM];
        %nom_fitxer = ['3DPBTC_resultats_',nomM];
        %save (nom_fitxer)
        %fprintf('==== Done subject %1d and mask %d ====\n',tt,m);
    end
end

