for i=1:                                        %%the number of test_data
      I1=imread(strcat('',num2str(i),'.tif'));  %%output of the network  Band 1
      I2=imread(strcat('',num2str(i),'.tif'));  %%output of the network  Band 2
      I3=imread(strcat('',num2str(i),'.tif'));  %%output of the network  Band 3
      I4=imread(strcat('',num2str(i),'.tif'));  %%output of the network  Band 4
        
     h = fspecial('gaussian',[3,3]); 
     I1_base = imfilter(I1, h, 'replicate');
     I1_detail = I1-I1_base; 
     I2_base = imfilter(I2, h, 'replicate');
     I2_detail = I2-I2_base; 
     I3_base = imfilter(I3, h, 'replicate');
     I3_detail = I3-I3_base; 
     I4_base = imfilter(I4, h, 'replicate');
     I4_detail = I4-I4_base;  
 
 
    I1_hist=imread(strcat('',num2str(i),'.tif'));  %%low resolution MS  Band 1
    I2_hist=imread(strcat('',num2str(i),'.tif'));  %%low resolution MS  Band 2
    I3_hist=imread(strcat('',num2str(i),'.tif'));  %%low resolution MS  Band 3
    I4_hist=imread(strcat('',num2str(i),'.tif'));  %%low resolution MS  Band 4   
    
  Jmatch_1=imhist(I1_hist);
  Iout_1=histeq(I1_base,Jmatch_1)+I1_detail;
 
  Jmatch_2=imhist(I2_hist);
  Iout_2=histeq(I2_base,Jmatch_2)+I2_detail;
  
  Jmatch_3=imhist(I3_hist);
  Iout_3=histeq(I3_base,Jmatch_3)+I3_detail;
  
  Jmatch_4=imhist(I4_hist);
  Iout_4=histeq(I4_base,Jmatch_4)+I4_detail;
  
  A=cat(3,Iout_1,Iout_2,Iout_3,Iout_4);
  imwrite(uint8(A),strcat('',num2str(i),'.tif'));  %%%%%% ×îÖÕ½á¹û
end