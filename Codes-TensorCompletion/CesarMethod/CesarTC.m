% 3D Inpainting based on Sparse Tucker Representation
% 
% Some missed fibers are reconstructed from data using specific dictionary

function [IMrec_1] = CesarTC(IM,Mask,ps,D0,sparsity)
% IM: original data (tensor)
% Mask: used to eliminate some data and obtain observed tensor (with missing entries)
% ps: patch size (should be the same value used for learning the dictionary)
% D0: specific dictionary for this data
% sparsity: how sparse the tensor is (should be the same value used for learning the dictionary)

%% tensor completion
IS = size(IM);
IMrec = zeros(size(IM));
COUNT = zeros(size(IM));
% patch dimensions
I = [ps,ps,ps];
epsilon = 0.055;
end1 = IS(1)-I(1)+1;
end2 = IS(2)-I(2)+1;
end3 = IS(3)-I(3)+1;

iter = 1;
for i1 = 1:1:end1  
    for i2 = 1:1:end2
        for i3 = 1:1:end3
            areai = i1:i1+I(1)-1;
            areaj = i2:i2+I(2)-1;
            areak = i3:i3+I(3)-1;
            I0 = IM(areai,areaj,areak);
            I = size(I0);
            [Yap, X] = Inpainting3Dn(I0, Mask(areai,areaj,areak), D0, epsilon,sparsity,'noshow');
            IMrec(areai,areaj,areak) = IMrec(areai,areaj,areak) + Yap;
            COUNT(areai,areaj,areak) = COUNT(areai,areaj,areak) + 1;
            M = max(I0(:));
            m = min(I0(:));
            mag =400;
            iter = iter + 1;
        end
        strcat('iter :', num2str(iter),' of :', num2str(end1*end2*end3))
    end
end
IMrec = IMrec./COUNT;
IMrec_1 = IM.*Mask+IMrec.*not(Mask); % we maintain the original values where the Mask is equal to 1 and the recovered values where tha Mask is equal to 0
clear IM;
% ALERTA, aix? ara est? millor, generalitzat per a qualsevol tipus de
% m?scara... Abans (2014) estava malament perqu? nom?s permetia m?scares amb
% fibres a zero, per? no amb columnes a zero!!!
