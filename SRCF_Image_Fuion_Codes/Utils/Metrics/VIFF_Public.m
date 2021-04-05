%% -----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
% Copyright (c) 2012, Yu Han,(Chinese Name: HAN Yu) All rights reserved.
%       The name of this code is "image fusion performance metric based on visual information fidelity".
% Permission to use and copy this software and its documentation for educational and 
% research purposes only and without fee is hereby granted, provided that this 
% copyright notice and the original authors names appear on all copies and supporting 
% documentation. 
%   The authors are acknowledged in any publication that reports research using this software.
%   The work is to be cited in the bibliography as:
%   	[]Yu Han, Yunze Cai, Yin Cao, Xiaoming Xu, A new image fusion performance metric 
%   	based on visual information fidelity, information fusion, Volume 14, Issue 2, April 2013, Pages 127¨C135
%   This code shall not be used, rewritten, or adapted as the basis of a commercial 
% software or hardware product without hand-writing permission of the authors. The authors 
% make no representations about the suitability of this software for any purpose. It is 
% provided "as is" without express or implied warranty.
%% -----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------
function F=VIFF_Public(Im1,Im2,ImF)
% Cite this work as
% []Yu Han, Yunze Cai, Yin Cao, Xiaoming Xu, A new image fusion performance metric based on visual information fidelity, information fusion, Volume 14, Issue 2, April 2013, Pages 127¨C135
% input:
%       Im1, source image 1
%       Im2, source image 2
%       ImF, fused image
% output:
%       F, fusion assessment value
%
% visual noise
sq=0.005*255*255;
% error comaprison parameter
C=1e-7;

[r,s,l]=size(Im1);
%color space transformation
if l==3
    cform = makecform('srgb2lab');
    T1 = applycform(Im1,cform);
    T2 = applycform(Im2,cform);
    TF = applycform(ImF,cform);
    Ix1=T1(:,:,1);
    Ix2=T2(:,:,1);
    IxF=TF(:,:,1); 
else
    Ix1=Im1;
    Ix2=Im2;
    IxF=ImF;
end

T1p=double(Ix1);
T2p=double(Ix2);
Trp=double(IxF);

p=[1,0,0.15,1]./2.15;
[T1N,T1D,T1G]=ComVidVindG(T1p,Trp,sq);
[T2N,T2D,T2G]=ComVidVindG(T2p,Trp,sq);
VID=[];
VIND=[];
%i multiscale image level
for i=1:4
    M_Z1=cell2mat(T1N(i));
    M_Z2=cell2mat(T2N(i));
    M_M1=cell2mat(T1D(i));
    M_M2=cell2mat(T2D(i));
    M_G1=cell2mat(T1G(i));
    M_G2=cell2mat(T2G(i));
    L=M_G1<M_G2;
    M_G=M_G2;
    M_G(L)=M_G1(L);
    M_Z12=M_Z2;
    M_Z12(L)=M_Z1(L);
    M_M12=M_M2;
    M_M12(L)=M_M1(L);
    
    VID=sum(sum((M_Z12+C)));
    VIND=sum(sum((M_M12+C)));
    F(i)=VID/VIND;
end
F=sum(F.*p);


function [Tg1,Tg2,Tg3]=ComVidVindG(ref,dist,sq)
% this part is mainly from the work:
% [] H.R.Sheikh and A.C.Bovik, Image information and visual quality[J], IEEE Transactions on Image Processing 15(2), pp. 430¨C444, 2006.
% And we have a little revision in our code
% input:
%       ref, source image
%       dist,fused image
%       sq, visual noise
% output:
%       Tg1, the matrix of visual information with distortion information (VID)
%       Tg2, the matrix of visual information without distortion information (VIND)
%       Tg3, the matrix of scalar value gi
sigma_nsq=sq;

for scale=1:4
   
    N=2^(4-scale+1)+1;
    win=fspecial('gaussian',N,N/5);
    
    if (scale >1)
        ref=filter2(win,ref,'valid');
        dist=filter2(win,dist,'valid');
        ref=ref(1:2:end,1:2:end);
        dist=dist(1:2:end,1:2:end);
    end
    
    mu1   = filter2(win, ref, 'valid');
    mu2   = filter2(win, dist, 'valid');
    mu1_sq = mu1.*mu1;
    mu2_sq = mu2.*mu2;
    mu1_mu2 = mu1.*mu2;
    sigma1_sq = filter2(win, ref.*ref, 'valid') - mu1_sq;
    sigma2_sq = filter2(win, dist.*dist, 'valid') - mu2_sq;
    sigma12 = filter2(win, ref.*dist, 'valid') - mu1_mu2;
    
    sigma1_sq(sigma1_sq<0)=0;
    sigma2_sq(sigma2_sq<0)=0;
    
    g=sigma12./(sigma1_sq+1e-10);
    sv_sq=sigma2_sq-g.*sigma12;
    
    g(sigma1_sq<1e-10)=0;
    sv_sq(sigma1_sq<1e-10)=sigma2_sq(sigma1_sq<1e-10);
    sigma1_sq(sigma1_sq<1e-10)=0;
    
    g(sigma2_sq<1e-10)=0;
    sv_sq(sigma2_sq<1e-10)=0;
    
    sv_sq(g<0)=sigma2_sq(g<0);
    g(g<0)=0;
    sv_sq(sv_sq<=1e-10)=1e-10;
    
     G(scale)={g};
     VID=log10(1+g.^2.*sigma1_sq./(sv_sq+sigma_nsq));
     VIND=log10(1+sigma1_sq./sigma_nsq);
     Num(scale)={VID};
     Den(scale)={VIND};    
end
Tg1=Num;
Tg2=Den;
Tg3=G;