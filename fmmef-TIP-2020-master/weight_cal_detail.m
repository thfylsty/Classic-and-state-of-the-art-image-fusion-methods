% ========================================================================
% Fast Multi-Scale Structural Patch Decomposition for Multi-Exposure Image Fusion, TIP,2020
% algorithm Version 1.0
% Copyright(c) 2020, Hui Li, Kede Ma, Yongwei Yong and Lei Zhang
% All Rights Reserved.
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is hereby
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
% Please refer to the following paper:
% H. Li et al., "Fast Multi-Scale Structural Patch Decomposition for Multi-Exposure Image Fusion, 2020" In press
% IEEE Transactions on Image Processing
% Please kindly report any suggestions or corrections to xiaohui102788@126.com
%----------------------------------------------------------------------
function [sMap,Cmax,i_mean2] = weight_cal_detail(I,p,r)
R=2*r+1;

[h,w,~,~]=size(I);
N1 = boxfilter(ones(h, w), r);

% countWindowt = ones(R, R);
% countWindow=countWindowt./sum(countWindowt(:));
if size(I,4)==1
N = size(I,3);
else
    N = size(I,4);
end
    
C = zeros(size(I,1),size(I,2),N);
i_mean2=zeros(size(I,1),size(I,2),N);
for i = 1:N
    
    if size(I,4)==1
    img = I(:,:,i);
        i_mean2(:,:,i)=boxfilter(img, r)./ N1;
     i_var2= boxfilter(img.*img, r) ./ N1- i_mean2(:,:,i).* i_mean2(:,:,i);
      
    i_var2=sqrt(max(i_var2,0));
    
    C(:,:,i) = i_var2 * sqrt( R^2  )+ 1e-12; % signal strengh
    else
       img = I(:,:,:,i);
       
       
       i_mean2(:,:,i)=boxfilter(img(:,:,1), r)./ N1+boxfilter(img(:,:,2), r)./ N1+boxfilter(img(:,:,3), r)./ N1;
        i_mean2(:,:,i)= i_mean2(:,:,i)./3;
        
        
%      i_var22= (boxfilter(img(:,:,1).*img(:,:,1), r)+ boxfilter(img(:,:,2).*img(:,:,2), r)+...
%          boxfilter(img(:,:,3).*img(:,:,3), r))./ N1- i_mean2(:,:,i).* i_mean2(:,:,i);
   i_var2= boxfilter(img(:,:,1).*img(:,:,1), r)./ N1+ boxfilter(img(:,:,2).*img(:,:,2), r)./ N1+...
         boxfilter(img(:,:,3).*img(:,:,3), r)./ N1- i_mean2(:,:,i).* i_mean2(:,:,i)*3;
     
    i_var2 = i_var2./3; 
     

% %      b =  i_mean2(:,:,i) - a .*  i_mean2(:,:,i);
%       i_mean(:,:,i)= i_mean2(:,:,i).*(1-edg(:,:,i))+edg(:,:,i).*img;
      
%       i_mean(:,:,i)=guidedfilter(img, img, r, 0.01);
      
    i_var2=sqrt(max(i_var2,0));
%        sigma = sqrt( max( sigmaSq, 0 ) );
%     ed = sigma * sqrt( wSize^2 * s3 ) +1e-12 ; % signal strengh
    
    C(:,:,i) = i_var2 * sqrt( 3*R^2  )+ 1e-12; % signal strengh
       
%  img = mean(I(:,:,i),3);
    end
    
%     i_mean2=filter2(countWindow,img,'same');
%     i_var2=filter2(countWindow,img.*img,'same')-i_mean2.*i_mean2;
    
      
    
end
Cmax=max(C,[],3);
% Cmax=repmat(Cmax,[1 1 3]);


sMap1 = C.^p; % signal structure weighting map
sMap2 = C.^(p-1);
sMap = sMap1 + 1e-12;
normalizer = sum(sMap,3);
sMap = sMap2 ./ repmat(normalizer,[1, 1, N]) ;