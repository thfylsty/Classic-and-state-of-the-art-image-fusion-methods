clc;
clear all;
for num=1:19
  I_ir=im2double(imread(strcat('',num2str(num-1),'.png')));  
  [I,v1,v2]=rgb2ihs(I_ir); 
  imwrite(I, strcat('',num2str(num),'.bmp'));
%   imwrite(v1,strcat('',num2str(num),'.bmp'));
%   imwrite(v2,strcat('',num2str(num),'.bmp'));
  
end