clc;
clear all;
for num=1:25
%   I=(imread(strcat('C:\Users\张浥尘\Desktop\Second_Fusion\Medical_image\中间处理\输入分离\I\',num2str(num),'.bmp')));
%   v1=(imread(strcat('C:\Users\张浥尘\Desktop\Second_Fusion\Medical_image\中间处理\输入分离\S\',num2str(num),'.bmp')));
%   v2=(imread(strcat('C:\Users\张浥尘\Desktop\Second_Fusion\Medical_image\中间处理\输入分离\H\',num2str(num),'.bmp')));
%   ISH=cat(3,I,v1,v2);
%   imwrite(ISH,strcat('C:\Users\张浥尘\Desktop\Second_Fusion\Medical_image\中间处理\输入分离\ISH_save_cat\',num2str(num),'.bmp'));

%   I=im2double(imread(strcat('C:\Users\张浥尘\Desktop\Second_Fusion\Medical_image\中间处理\输入分离\I\',num2str(num),'.bmp')));
%   v1=im2double(imread(strcat('C:\Users\张浥尘\Desktop\Second_Fusion\Medical_image\中间处理\输入分离\S\',num2str(num),'.bmp')));
%   v2=im2double(imread(strcat('C:\Users\张浥尘\Desktop\Second_Fusion\Medical_image\中间处理\输入分离\H\',num2str(num),'.bmp')));
%   ISH(:,:,1)=I;
%   ISH(:,:,2)=v1;
%   ISH(:,:,3)=v2;
% 
%   imwrite(ISH,strcat('C:\Users\张浥尘\Desktop\Second_Fusion\Medical_image\中间处理\输入分离\ISH_save_fed\',num2str(num),'.bmp'));


% 
   I_ir=(imread(strcat('C:\Users\张浥尘\Desktop\Second_Fusion\Medical_image\中间处理\bmp\ir\',num2str(num),'.bmp')));  
   [I,v1,v2]=rgb2ihs(I_ir); 

  ISH=cat(3,I,v1,v2);
  imwrite(ISH,strcat('C:\Users\张浥尘\Desktop\Second_Fusion\Medical_image\中间处理\输入分离\ISH_cat\',num2str(num),'.bmp'));



%    I_ir=im2double(imread(strcat('C:\Users\张浥尘\Desktop\Second_Fusion\Medical_image\中间处理\bmp\ir\',num2str(num),'.bmp')));  
%    [I,v1,v2]=rgb2ihs(I_ir); 
%   ISH(:,:,1)=I;
%   ISH(:,:,2)=v1;
%   ISH(:,:,3)=v2;
%   imwrite(ISH,strcat('C:\Users\张浥尘\Desktop\Second_Fusion\Medical_image\中间处理\输入分离\ISH_fed\',num2str(num),'.bmp'));
end