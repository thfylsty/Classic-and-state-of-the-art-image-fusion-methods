clc;
clear all;
for num=1:30
  for i=1:40
      if i<=10
         I_result=double(imread(strcat('',num2str(num-1),'\','F9_0',num2str(i-1),'.bmp')));  
      else
         I_result=double(imread(strcat('',num2str(num-1),'\','F9_',num2str(i-1),'.bmp')));
      end
      I_init_ir=double(imread(strcat('',num2str(i),'.bmp')));
      I_init_vi=double(imread(strcat('',num2str(i),'.bmp')));
      [Y1,Cb1,Cr1]=RGB2YCbCr(I_init_ir);
      [Y2,Cb2,Cr2]=RGB2YCbCr(I_init_vi);
      
      [H,W]=size(Cb1);
      Cb=ones([H,W]);
      Cr=ones([H,W]);
      
      for k=1:H
          for n=1:W
           if (abs(Cb1(k,n)-128)==0&&abs(Cb2(k,n)-128)==0)  
              Cb(k,n)=128;
           else
                middle_1= Cb1(k,n)*abs(Cb1(k,n)-128)+Cb2(k,n)*abs(Cb2(k,n)-128);
                middle_2=abs(Cb1(k,n)-128)+abs(Cb2(k,n)-128);
                Cb(k,n)=middle_1/middle_2;
           end   
            if (abs(Cr1(k,n)-128)==0&&abs(Cr2(k,n)-128)==0)      
               Cr(k,n)=128;  
            else
                middle_3=Cr1(k,n)*abs(Cr1(k,n)-128)+Cr2(k,n)*abs(Cr2(k,n)-128);
                middle_4=abs(Cr1(k,n)-128)+abs(Cr2(k,n)-128); 
                Cr(k,n)=middle_3/middle_4;
           end
%                 middle_1= Cb1(k,n)*(abs(Cb1(k,n)-128)+eps)+Cb2(k,n)*(abs(Cb2(k,n)-128)+eps);
%                 middle_2=(abs(Cb1(k,n)-128)+eps)+(abs(Cb2(k,n)-128)+eps);
%                 middle_3=Cr1(k,n)*(abs(Cr1(k,n)-128)+eps)+Cr2(k,n)*(abs(Cr2(k,n)-128)+eps);
%                 middle_4=(abs(Cr1(k,n)-128)+eps)+(abs(Cr2(k,n)-128)+eps); 
%                 Cb(k,n)=middle_1/middle_2;
%                 Cr(k,n)=middle_3/middle_4;
%               
          end
      end

      
      I_final_YCbCr=cat(3,I_result,Cb,Cr);
      
      I_final_RGB=YCbCr2RGB(I_final_YCbCr);
      
      
      
      
      if ~exist(strcat('E:\Second_Fusion\multi_exposure\RGB\结果\Ours\1.5_1.5_16_16\最终结果\result\epoch',num2str(num-1))) ;
         mkdir(strcat('E:\Second_Fusion\multi_exposure\RGB\结果\Ours\1.5_1.5_16_16\最终结果\result\epoch',num2str(num-1)));
      end
      if i<=10
         imwrite(uint8(I_final_RGB), strcat('E:\Second_Fusion\multi_exposure\RGB\结果\Ours\1.5_1.5_16_16\最终结果\result\epoch',num2str(num-1),'\','F9_0',num2str(i-1),'.bmp')); 
      else
         imwrite(uint8(I_final_RGB), strcat('E:\Second_Fusion\multi_exposure\RGB\结果\Ours\1.5_1.5_16_16\最终结果\result\epoch',num2str(num-1),'\','F9_',num2str(i-1),'.bmp')); 
      end
    end
end