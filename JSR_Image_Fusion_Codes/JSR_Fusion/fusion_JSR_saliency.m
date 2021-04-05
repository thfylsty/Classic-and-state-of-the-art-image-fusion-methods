% C.H. Liu, Y. Qi, W.R. Ding. Infrared and visible image fusion method based on saliency detection in sparse domain[J]. Infrared Physics & Technology.
% saliency map

% function fusion_JSR_saliency
addpath(genpath('./dictionary'));
for inde=1:10
index = inde;
disp(num2str(index));

matName = strcat('D_unit7_im',num2str(index));
matName = strcat(matName, '.dat');
load(matName, '-mat');

path1 = ['../../_________________________DATA/mid/Test_ir/',num2str(index),'.bmp'];
path2 = ['../../_________________________DATA/mid/Test_ir/',num2str(index),'.bmp'];
fused_path = ['../../融合结果/3/',num2str(index),'.bmp'];

source_image1 = imread(path1);
source_image2 = imread(path2);

I1 = im2double(source_image1);
I2 = im2double(source_image2);

[m,n] = size(I1);

unit = 7;
step = 1;

row_unit = unit*unit;

disp(strcat('开始计算分块数据'));
count = 0;
for i=4:step:(m-3)
    for j=4:step:(n-3)
        count = count+1;
        patch1 = I1((i-3):(i+3),(j-3):(j+3));
        patch2 = I2((i-3):(i+3),(j-3):(j+3));
        
        Vi1(:, count) = patch1(:);
        Vi2(:, count) = patch2(:);
    end
end
disp(strcat('结束计算分块数据'));

patch_num = count;

% KSVD训练子字典
dic_size = 256;
k = 16;

V_Joint = zeros(2*row_unit, patch_num);
V_Joint(1:(row_unit),:) = Vi1;
V_Joint((row_unit+1):(2*row_unit),:) = Vi2;

D_Joint = zeros(2*row_unit, dic_size*3);
D_Joint(1:row_unit,1:dic_size) = D/sqrt(2);
D_Joint(1:row_unit,(dic_size+1):2*dic_size) = D;
D_Joint((row_unit+1):2*row_unit,1:dic_size) = D/sqrt(2);
D_Joint((row_unit+1):2*row_unit,(dic_size*2+1):3*dic_size) = D;

disp('OMP-求解系数');
tic
% C = zeros(dic_size*3, count);
% for i=1:count
%     c = omp(D_Joint, V_Joint(:,i),[], k);
%     C(:,i) = c;
% end
C = omp(D_Joint, V_Joint,[], k);
toc
disp('OMP-求解系数 结束');

coe_c = (C(1:dic_size,:))/sqrt(2);
coe_u1 = C((dic_size+1):2*dic_size,:);
coe_u2= C((2*dic_size+1):3*dic_size,:);

% V_com = D*coe_c;
% V_u1 = D*coe_u1;
% V_u2 = D*coe_u2;

%对系数进行融合
disp('融合开始');
tic
% m_unit = floor(m/unit);
% n_unit = floor(n/unit);
coe_index = 0;
coe_fusion = zeros(dic_size, count);

saliency_part1 = zeros(m,n);
saliency_part2 = zeros(m,n);

saliency_gobal1 = zeros(m,n);
saliency_gobal2 = zeros(m,n);

saliency_map1 = zeros(m,n);
saliency_map2 = zeros(m,n);

for i=4:step:(m-3)
    for j=4:step:(n-3)
        coe_index = coe_index+1;
        n1 = norm(coe_u1(:,coe_index),1);
        n2 = norm(coe_u2(:,coe_index),1);
        
        if (i<6 || j<6) || (i>(m-5) || j>(n-5))
            w1 = n1/max(n1,n2);
            w2 = n2/max(n1,n2);
            coe_fusion(:,coe_index) = coe_c(:,coe_index)+w1*coe_u1(:,coe_index)+w2*coe_u2(:,coe_index);
        else
            d_exp1 = 0;
            d_exp2 = 0;
            for p=(i-2):(i+2)
                for q=(j-2):(j+2)
                    if p~=i || q~=j
                        temp_index = (p-4)*(n-unit+1)+(q-3);
                        du1 = sqrt(sum((coe_u1(:,coe_index)-coe_u1(:,temp_index)).^2));
                        dp1 = sqrt((p-i)^2+(q-j)^2);
                        d_exp1 = d_exp1+(du1/(1+2*dp1));
                        
                        du2 = sqrt(sum((coe_u2(:,coe_index)-coe_u2(:,temp_index)).^2));
                        dp2 = sqrt((p-i)^2+(q-j)^2);
                        d_exp2 = d_exp2+(du2/(1+2*dp2));
                    end
                end
            end
            d_exp1 = d_exp1/24;
            sl1 = 1-exp(-d_exp1);
            d_exp2 = d_exp2/24;
            sl2 = 1-exp(-d_exp2);
            
            sg1 = n1;
            sg2 = n2;
            
            S1 = 0.4*sl1+0.6*sg1;
            S2 = 0.4*sl2+0.6*sg2;
            
            saliency_part1(i,j) = sl1;
            saliency_part2(i,j) = sl2;
            
            saliency_gobal1(i,j) = sg1;
            saliency_gobal2(i,j) = sg2;
            
            saliency_map1(i,j) = S1;
            saliency_map2(i,j) = S2;
            
            wu1 = S1/max(S1,S2);
            wu2 = S2/max(S1,S2);
            coe_fusion(:,coe_index) = coe_c(:,coe_index)+wu1*coe_u1(:,coe_index)+wu2*coe_u2(:,coe_index);
        end
    end
end
toc
disp('融合结束');

disp('开始重构');
V_fusion = D*coe_fusion;
fusion = zeros(m,n);
countt = 0;
for i=4:step:(m-3)
    for j=4:step:(n-3)
        countt = countt+1;
        patch1 = V_fusion(:, countt);
        pr = reshape(patch1,[unit,unit]);
        temp = fusion((i-3):(i+3),(j-3):(j+3));
        fusion((i-3):(i+3),(j-3):(j+3))=(pr+temp)/2;
    end
end
disp('重构结束');
figure;
imshow(fusion);

imwrite(fusion,fused_path,'png');

clear Vi1;
clear Vi2;
end

% end



