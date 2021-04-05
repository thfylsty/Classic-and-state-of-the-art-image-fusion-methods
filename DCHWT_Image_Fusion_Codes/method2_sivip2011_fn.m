function fuse_im=method2_sivip2011_fn(inp_wt,Nlevels,detail_exponent)

%%% method2_sivip2011_fn: Fuses the 2 images contained in inp_wt using the method discussed in
%%% Multifocus & multispectral image fusion based on pixel significance using multiresolution
%%% decomposition, SIViP,2011 (DOI: 10.1007/s11760-011-0219-7)
%%% (SIViP 2011)-> detail_exponent=1; 
%%%
%%% fuse_im=method2_sivip2011_fn(x,3,1) 
%%%
%%% Author : B. K. SHREYAMSHA KUMAR 
%%% Created on 21-10-2011.
%%% Updated on 21-02-2012.


NoOfBands=3*Nlevels+1;

%%% Significance factor of all subbands who has childrens.
for k=NoOfBands-1:-1:4
   rr=1;
   level_cnt=Nlevels-ceil(k/3)+1;
   for mm=level_cnt:Nlevels
      sband1{rr}=cell2mat(inp_wt{1}(k-3*(mm-level_cnt)));
      sband2{rr}=cell2mat(inp_wt{2}(k-3*(mm-level_cnt)));
      rr=rr+1;
   end
   [p,q]=size(sband1{1});
   abs_sum1=0; abs_sum2=0;
   for ii=1:p
      for jj=1:q
         abs_sum1=abs_sum1+abs(sband1{1}(ii,jj));
         abs_sum2=abs_sum2+abs(sband2{1}(ii,jj));         
         for tt=2:length(sband1)
            temp1=sband1{tt}(2^(tt-1)*ii-[2^(tt-1)-1:-1:0],2^(tt-1)*jj-[2^(tt-1)-1:-1:0]);
            temp2=sband2{tt}(2^(tt-1)*ii-[2^(tt-1)-1:-1:0],2^(tt-1)*jj-[2^(tt-1)-1:-1:0]);
            abs_sum1=abs_sum1+sum(abs(temp1(:)));
            abs_sum2=abs_sum2+sum(abs(temp2(:)));
            clear temp1 temp2;
         end
         sig_temp1(ii,jj)=abs_sum1; 
         sig_temp2(ii,jj)=abs_sum2;
         abs_sum1=0; abs_sum2=0;
      end
   end
   sig_mat1{k}=sig_temp1;
   sig_mat2{k}=sig_temp2;
   clear sband1 sig_temp1;
   clear sband2 sig_temp2;
end

%%% Significance factor of subbands who do not has childrens.
wsize=3;
hwsize=(wsize-1)/2;
for k=3:-1:1
   temp1=cell2mat(inp_wt{1}(k));
   temp2=cell2mat(inp_wt{2}(k));
   %%% Periodic extension to take care of boundary conditions.
   temp1_ext=per_extn_im_fn(temp1,wsize);
   temp2_ext=per_extn_im_fn(temp2,wsize);   
   [p,q]=size(temp1_ext);
   for ii=hwsize+1:p-hwsize
      for jj=hwsize+1:q-hwsize
         rpt1=ii-hwsize; rpt2=ii+hwsize;
         cpt1=jj-hwsize; cpt2=jj+hwsize;
         temp1_energy=temp1_ext(rpt1:rpt2,cpt1:cpt2).^2;
         temp2_energy=temp2_ext(rpt1:rpt2,cpt1:cpt2).^2;
         sig_temp1(ii-hwsize,jj-hwsize)=sum(temp1_energy(:))/wsize^2;
         sig_temp2(ii-hwsize,jj-hwsize)=sum(temp2_energy(:))/wsize^2;
      end
   end
   sig_mat1{k}=sig_temp1;
   sig_mat2{k}=sig_temp2;
   clear sig_temp1 sig_temp2;
end

%%% Significance factor of approximation subbands.
sig_mat1{NoOfBands}=sig_mat1{NoOfBands-1}+sig_mat1{NoOfBands-2}+(sig_mat1{NoOfBands-3}).^detail_exponent; %% Vertical,Horizontal,Diagonal resp
sig_mat2{NoOfBands}=sig_mat2{NoOfBands-1}+sig_mat2{NoOfBands-2}+(sig_mat2{NoOfBands-3}).^detail_exponent;

%%% Fusion.
for k=1:NoOfBands
   [p,q]=size(sig_mat1{k});
   tt1=cell2mat(inp_wt{1}(k));
   tt2=cell2mat(inp_wt{2}(k));
   for ii=1:p
      for jj=1:q
         fuse_im{k}(ii,jj)=(tt1(ii,jj)*sig_mat1{k}(ii,jj)+tt2(ii,jj)*sig_mat2{k}(ii,jj))/(sig_mat1{k}(ii,jj)+sig_mat2{k}(ii,jj));
      end
   end
end