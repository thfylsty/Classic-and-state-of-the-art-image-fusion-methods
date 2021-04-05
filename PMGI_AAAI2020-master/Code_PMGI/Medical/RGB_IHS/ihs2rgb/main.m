clc;
clear all;
for num=1:70
  for i=1:19
      if i<=10
         I_result=im2double(imread(strcat('',num2str(num),'\','F9_0',num2str(i-1),'.bmp')));  
      else
         I_result=im2double(imread(strcat('',num2str(num),'\','F9_',num2str(i-1),'.bmp')));
      end
      I_init=im2double(imread(strcat('',num2str(i),'.bmp')));
      [I,V1,V2]=rgb2ihs(I_init);
      I_final_ISH=cat(3,I_result,V1,V2);
      I_final_RGB=ihs2rgb(I_final_ISH);
      if ~exist(strcat('',num2str(num))) ;
         mkdir(strcat('',num2str(num)));
      end
      if i<=10
         imwrite(I_final_RGB, strcat('',num2str(num),'\','F9_0',num2str(i-1),'.bmp')); 
      else
         imwrite(I_final_RGB, strcat('',num2str(num),'\','F9_',num2str(i-1),'.bmp')); 
      end
    end
end