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

fused_path = ['./fused_mf_noise/fused',num2str(i),label,'_cbf.png'];
fused_path_new = ['./fused_mf_noise/fused',num2str(i),label,'_convsr.png'];

im=imread(fused_path);
imwrite(im,fused_path_new);
delete(fused_path);
end
end