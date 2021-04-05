% clc,clear
% %%% gray image fusion
% I = load_images( '.\sourceimages\grayset',1); 
% F = GFF(I);
% imshow(F);
%%%% color image fusion
I7=load_images('E:\lichang\1.Image fusion total variation\TNO_Image_Fusion_Dataset\Triclobs_images\Kaptein_01',1);% the folder of source image
tic;
X7 = double(GFF(I7,5,10^-6,5,10^-6));
t7=toc;
X7(X7<0)=0;
imwrite(X7/255,'Kaptein_01_7.png','png');
Result7 = Metric(uint8(abs(I)*255),uint8(abs(V)*255),uint8(X7));

Metric_Total=[Result.Total, Result1.Total, Result2.Total, Result3.Total, Result4.Total, Result5.Total, Result6.Total, ...
Result7.Total, Result8.Total, Result9.Total, Result10.Total, Result11.Total];