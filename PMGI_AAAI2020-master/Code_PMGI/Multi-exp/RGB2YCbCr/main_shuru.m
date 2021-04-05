clc;
clear all;
for num=1:40
  I_ir=(imread(strcat('',num2str(num),'.bmp')));  
  [Y,Cb,Cr]=RGB2YCbCr(I_ir); 
  imwrite(Y, strcat('',num2str(num),'.bmp'));
%   imwrite(v1,strcat('',num2str(num),'.bmp'));
%   imwrite(v2,strcat('',num2str(num),'.bmp'));  
end