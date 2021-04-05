clc;
clear all;
for i=1:200
    I=imread(strcat('\',num2str(i),'.tif'));
    Imatch_L=imread(strcat('',num2str(i),'.tif'));
 
    I1=I(:,:,1);
    I2=I(:,:,2);
    I3=I(:,:,3);
    I4=I(:,:,4);
     
    Imatch_1=imresize(Imatch_L(:,:,1),4);
    Imatch_2=imresize(Imatch_L(:,:,2),4);
    Imatch_3=imresize(Imatch_L(:,:,3),4);
    Imatch_4=imresize(Imatch_L(:,:,4),4);
    
    Jmatch1=imhist(Imatch_1);
    Iout1=histeq(I1,Jmatch1);
    
    Jmatch2=imhist(Imatch_2);
    Iout2=histeq(I2,Jmatch2);
    
    Jmatch3=imhist(Imatch_3);
    Iout3=histeq(I3,Jmatch3);   
    
    Jmatch4=imhist(Imatch_4);
    Iout4=histeq(I4,Jmatch4);  
    
    I_out=cat(3,Iout1,Iout2,Iout3,Iout4);
    
    imwrite(I_out,strcat('',num2str(i),'.tif'))
 
end