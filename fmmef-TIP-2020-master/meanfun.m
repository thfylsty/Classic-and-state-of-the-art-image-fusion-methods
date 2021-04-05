

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

function Y=meanfun(X,X_mean,index,width,xielv)


switch index
    
    case 'direct'
       Y=X;
Y(Y>=0.5)=1-Y(Y>=0.5); 
    case 'truncated'
if 0==exist('width','var')
    width=0.8;%????????
end
if 0==exist('xielv','var')
   
    xielv=0.2; %????????
end        
Mexp=1-(max(abs(X-0.5),0.0)-0.0)/(0.5-0.0);
Mexp=min(Mexp,width)/width;
m=0.5;
Y=(atan((Mexp-m).*(tan(pi*m-xielv*m)-tan(-pi*m+xielv*m)))+pi*m-xielv*m)./(pi-xielv);  

    case 'gussian_lg'
        lSig=0.5;
       gSig=0.2;
      Y =  exp( -.5 * ( (X_mean - .5).^2 /gSig.^2 +  (X - .5).^2 /lSig.^2 ) );     
     case 'gussian_l'
         lSig=0.2;
         Y=exp((-(X-0.5).^2)./(lSig.^2*2));
    case 'our'    %% arctan
     Y5=2/pi*atan(20*(X.*(X<=0.5)));
Y6=2/pi*atan(20*((1-X).*(X>0.5)));
Y7=Y5+Y6;
Y=Y7;
% Y=Y7./repmat(max(Y7),size(Y7,1),1);
end

% Y=(atan((Mexp-m).*(tan(pi*m-xielv*m)-tan(-pi*m+xielv*m)))+pi*m-xielv*m)./(pi-xielv);
% Y=exp(-(X-0.5).^2)./(lSig.^2*2);
% Y =  exp( -.5 * ( (X_mean - .5).^2 /gSig.^2 +  (X - .5).^2 /lSig.^2 ) ); 



%% huatu
%  width=0.4;%????????
%  xielv=0.8; %????????
% X=0:0.01:1;
% Mexp=1-(max(abs(X-0.5),0.0)-0.0)/(0.5-0.0);
% Mexp=min(Mexp,width)/width;
% m=0.5;
% Y=(atan((Mexp-m).*(tan(pi*m-xielv*m)-tan(-pi*m+xielv*m)))+pi*m-xielv*m)./(pi-xielv);
% plot(X,Y)


% sig=0.2;
% Y=exp(-(X-0.5).^2)./(sig.^2*2);

% Y=X;
% Y(Y>0.5)=1-Y(Y>0.5);
% plot(X,Y)