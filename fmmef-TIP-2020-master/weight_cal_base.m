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
function W= weight_cal_base(I,m,brightness)

r = size(I,1);
c = size(I,2);
N = size(I,3);


W = ones(r,c,N);

%compute the measures and combines them into a weight map
contrast_parm = m(1);
sat_parm = m(2);
wexp_parm = m(3);
fangcha_parm = m(4);
gradient_parm=m(5);
wexp2_parm=m(6);
mask_parm=m(7);


if (contrast_parm > 0)
    W = W.*contrast(I).^contrast_parm;
end
if (sat_parm > 0)
    W = W.*saturation(I).^sat_parm;
end
if (wexp_parm > 0)
    W = W.*well_exposedness(I).^wexp_parm;
end
if (fangcha_parm > 0)
    W = W.*fangcha(I).^fangcha_parm;
end
if (gradient_parm > 0)
    W = W.*gradient(I).^gradient_parm;
end

if (wexp2_parm > 0)
    W = W.*well_exposedness2(I,brightness).^wexp2_parm;
end
if (mask_parm > 0)
    W = W.*mask(I).^mask_parm;
end


%normalize weights: make sure that weights sum to one for each pixel
W = W + 1e-12; %avoids division by zero
W = W./repmat(sum(W,3),[1 1 N]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% contrast measure
function C = contrast(I)
h = [0 1 0; 1 -4 1; 0 1 0]; % laplacian filter
N = size(I,3);
C = zeros(size(I,1),size(I,2),N);
for i = 1:N
  
%     mono = rgb2gray(I(:,:,:,i));
  mono = I(:,:,i);
  
    C(:,:,i) = abs(imfilter(mono,h,'replicate'));
end

% std measure

function C = fangcha(I)
countWindowt = ones(11, 11);
countWindow=countWindowt./sum(countWindowt(:));
N = size(I,3);
C = zeros(size(I,1),size(I,2),N);
for i = 1:N
  
%     img = rgb2gray(I(:,:,:,i));
   img=I(:,:,i);
i_mean2=filter2(countWindow,img,'same');
  i_var2=filter2(countWindow,img.*img,'same')-i_mean2.*i_mean2;
    C(:,:,i)=sqrt(max(i_var2,0));
end

% gradient measure

function C = gradient(I)

N = size(I,3);
C = zeros(size(I,1),size(I,2),N);
for i = 1:N
   
%     img = rgb2gray(I(:,:,:,i));
   img=I(:,:,i);
   
    C(:,:,i) = imgradient(img);
end



% saturation measure
function C = saturation(I)
N = size(I,3);
C = zeros(size(I,1),size(I,2),N);
for i = 1:N
    % saturation is computed as the standard deviation of the color channels
    R = I(:,:,1,i);
    G = I(:,:,2,i);
    B = I(:,:,3,i);
    mu = (R + G + B)/3;
    C(:,:,i) = sqrt(((R - mu).^2 + (G - mu).^2 + (B - mu).^2)/3);
end





% well-exposedness measure
function C = well_exposedness(I)
sig = .2;
N = size(I,3);
C = zeros(size(I,1),size(I,2),N);
for i = 1:N
    R = exp(-.5*(I(:,:,1,i) - .5).^2/sig.^2);
    G = exp(-.5*(I(:,:,2,i) - .5).^2/sig.^2);
    B = exp(-.5*(I(:,:,3,i) - .5).^2/sig.^2);
    C(:,:,i) = R.*G.*B;
end


function C = well_exposedness2(I,brightness)
% sig = .2;
N = size(I,3);
C = zeros(size(I,1),size(I,2),N);
for i = 1:N
   
%  img = rgb2gray(I(:,:,:,i));
     img=I(:,:,i);
     
%      img((img>=0.95)|(img<=0))=1;
     
 M=ones(size(I,1),size(I,2))*mean(img(:));
 C(:,:,i)=meanfun(img,M,brightness);
end


function C = mask(I)
% sig = .2;
N = size(I,3);
C = zeros(size(I,1),size(I,2),N);
Ct=ones(size(I,1),size(I,2));
% C((imgs_gray>=0.8)|(imgs_gray<=0.2))=0;
for i = 1:N
    
%  img = rgb2gray(I(:,:,:,i));

  img=I(:,:,i);
 Ct((img>=0.95)|(img<=0.05))=0;
 C(:,:,i)=Ct;
end

