% This code is in association with the following paper
% "Ma J, Zhou Z, Wang B, et al. Infrared and visible image fusion based on visual saliency map and weighted least square optimization[J].
% Infrared Physics & Technology, 2017, 82:8-17."
% Authors: Jinlei Ma, Zhiqiang Zhou, Bo Wang, Hua Zong
% Code edited by Jinlei Ma, email: majinlei121@163.com

clear all
close all

label='';
for k = 1:5
if k==1
    label='_gau_001';
end
if k==2
    label='_gau_0005';
end
if k==3
    label='_sp_01';
end
if k==4
    label='_sp_02';
end
if k==5
    label='_poi';
end
disp(label);

for i=1:10
index = i;

% path1 = ['./MF_images/image',num2str(index),'_left.png'];
% path2 = ['./MF_images/image',num2str(index),'_right.png'];
% fused_path = ['./fused_mf/fused',num2str(index),'_wls.png'];

path1 = ['./mf_noise_images/image',num2str(i),label,'_left.png'];
path2 = ['./mf_noise_images/image',num2str(i),label,'_right.png'];
fused_path = ['./fused_mf_noise/fused',num2str(i),label,'_wls.png'];

% I1 is a visible image, and I2 is an infrared image.
I1 = imread(path1); 
I2 = imread(path2);

I1 = im2double(I1);
I2 = im2double(I2);

% figure;imshow(I1);
% figure;imshow(I2);
tic
fused = WLS_Fusion(I1,I2);
toc

% figure;imshow(fused);
imwrite(fused,fused_path,'png');

end
end

