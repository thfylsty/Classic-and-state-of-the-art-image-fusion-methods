for num =1:50
clearvars -except num;
warning off;
addpath(genpath(cd));

path1 = ['../road/ir/',num2str(num),'.jpg'];
path2 = ['../road/vi/',num2str(num),'.jpg'];
fused_path = ['result/',num2str(num),'.bmp'];

I=double(imread(path1))/255;
V=double(imread(path2))/255;

% image_left = ['../road/ir/',num2str(i),'.jpg'];
% image_right = ['../road/vi/',num2str(i),'.jpg'];
% fused_path = ['result/',num2str(i),'.bmp'];
% 
% x{1}=imread(image_left);
% x{2}=imread(image_right);   

calc_metric = 1; % Calculate the metrices is time consuming, it is used for quantitative evaluation. Set it to 0 if you do not want to do it.

%%
%The proposed GTF
nmpdef;
pars_irn = irntvInputPars('l1tv');

pars_irn.adapt_epsR   = 1;
pars_irn.epsR_cutoff  = 0.01;   % This is the percentage cutoff
pars_irn.adapt_epsF   = 1;
pars_irn.epsF_cutoff  = 0.05;   % This is the percentage cutoff
pars_irn.pcgtol_ini = 1e-4;
pars_irn.loops      = 5;
pars_irn.U0         = I-V;
pars_irn.variant       = NMP_TV_SUBSTITUTION;
pars_irn.weight_scheme = NMP_WEIGHTS_THRESHOLD;
pars_irn.pcgtol_ini    = 1e-2;
pars_irn.adaptPCGtol   = 1;

tic;
U = irntv(I-V, {}, 4, pars_irn);
t0=toc;

X=U+V;
X=im2gray(X);
imwrite(X,fused_path);
% imwrite(X,['F/GTF/',num2str(num),'.png'],'png');
% if calc_metric, Result = Metric(uint8(abs(I)*255),uint8(abs(V)*255),uint8(abs(X*255))); end

%%
%The laplacian pyramid as a compact image code(1983)
% level=4;
% tic;
% X1 = lp_fuse(I, V, level, 3, 3);       %LP
% t1=toc;
% X1=im2gray(X1);
% % imwrite(X1,['F/1/',num2str(num),'.png'],'png');
% if calc_metric, Result1 = Metric(uint8(abs(I)*255),uint8(abs(V)*255),uint8(abs(X1*255))); end

%%
%Image fusion by a ratio of low pass pyramid(1989)
% tic;
% X2 = rp_fuse(I, V, level, 3, 3);      %RP
% t2=toc;
% X2=im2gray(X2);
% imwrite(X2,['F/2/',num2str(num),'.png'],'png');
% if calc_metric, Result2 = Metric(uint8(abs(I)*255),uint8(abs(V)*255),uint8(abs(X2*255))); end

%%
% Wavelet
% fusion by taking the mean for both approximations and details
% tic;
% X3 = wfusimg(I,V,'db2',5,'mean','mean');
% X3=im2gray(X3);
% t3=toc;
% imwrite(X3,['F/3/',num2str(num),'.png'],'png');
% imwrite(X3,fused_path);
% if calc_metric, Result3 = Metric(uint8(abs(I)*255),uint8(abs(V)*255),uint8(abs(X3*255))); end

%%
%Pixel-and region-based image fusion with complex wavelets(2007)
% [M,N]=size(I);
% I4=imresize(I,[M+mod(M,2) N+mod(N,2)]);
% V4=imresize(V,[M+mod(M,2) N+mod(N,2)]);
% tic;
% X4 = dtcwt_fuse(I4, V4,level);           %DTCWT
% t4=toc;
% X4=im2gray(X4);
% imwrite(X4,['F/4/',num2str(num),'.png'],'png');
% if calc_metric, Result4 = Metric(uint8(abs(I4)*255),uint8(abs(V4)*255),uint8(abs(X4*255))); end
% 
% %%
% %Remote sensing image fusion using the curvelet transform(2007)
% tic;
% X5 = curvelet_fuse(I4, V4,level+1);      %CVT
% t5=toc;
% X5=im2gray(X5);
% imwrite(X5,['F/5/',num2str(num),'.png'],'png');
% if calc_metric, Result5 = Metric(uint8(abs(I4)*255),uint8(abs(V4)*255),uint8(abs(X5*255))); end

%%
%Image Fusion technique using Multi-resolution singular Value decomposition(2011)
%apply MSVD
% tic;
% [Y1, U1] = MSVD(I4);
% [Y2, U2] = MSVD(V4);
% 
% %fusion starts
% X6.LL = 0.5*(Y1.LL+Y2.LL);
% 
% D  = (abs(Y1.LH)-abs(Y2.LH)) >= 0; 
% X6.LH = D.*Y1.LH + (~D).*Y2.LH;
% D  = (abs(Y1.HL)-abs(Y2.HL)) >= 0; 
% X6.HL = D.*Y1.HL + (~D).*Y2.HL;
% D  = (abs(Y1.HH)-abs(Y2.HH)) >= 0; 
% X6.HH = D.*Y1.HH + (~D).*Y2.HH;
% 
% %XX = [X.LL, X.LH; X.HL, X.HH];
% U = 0.5*(U1+U2);
% 
% %apply IMSVD
% X6 = IMSVD(X6,U);
% t6=toc;
% X6=im2gray(X6);
% imwrite(X6,fused_path);
% if calc_metric, Result6 = Metric(uint8(abs(I4)*255),uint8(abs(V4)*255),uint8(abs(X6*255))); end
% 
% %%
% %Image Fusion with Guided Filtering(2013)
% %run('F:\Code\Lichang\16.Image fusion total variation\Image fusion with guided filtering\Demo.m');
% I7=load_images('.\img',1);% the folder of source image
% tic;
% X7 = double(GFF(I7,5,10^-6,5,10^-6));
% %X7=rgb2gray(X7);
% t7=toc;
% imwrite(X7,['F/7/',num2str(num),'.png'],'png');
% %if calc_metric, Result7 = Metric(uint8(abs(I)*255),uint8(abs(V)*255),uint8(X7)); end
% 
% %%
% %A general framework for image fusion based on multi-scale transform and sparse representation(2014)
% overlap = 6;                    
% epsilon=0.1;
% level=4;
% load('D_100000_256_8.mat');
% tic;
% X8= lp_sr_fuse(I,V,level,3,3,D,overlap,epsilon);      %LP-SR
% t8=toc;
% X8=im2gray(X8);
% imwrite(X8,['F/8/',num2str(num),'.png'],'png');
% if calc_metric, Result8 = Metric(uint8(abs(I)*255),uint8(abs(V)*255),uint8(abs(X8*255))); end
end
