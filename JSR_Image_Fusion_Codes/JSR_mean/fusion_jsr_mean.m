% JSR+mean 融合算法
% load D_ksvd.dat -mat;
load D_k5_ksvd.dat -mat;

% for index=1:5
index = 2;
num = index;
% num = 1;
path1 = ['./test_images/IR',num2str(num),'.jpg'];
path2 = ['./test_images/VIS',num2str(num),'.jpg'];
fused_path = ['./fused_test/fused_',num2str(num),'_jsr_mean_k5.png'];

I1 = imread(path1);
I2 = imread(path2);
I1 = im2double(I1);
I2 = im2double(I2);

[m,n] = size(I1);

unit = 8;
count = 0;

for i=1:(m-unit+1)
    if rem(i,50)==0
        str = ['index = ',num2str(i)];
        disp(strcat(str));
    end
    for j=1:(n-unit+1)
        count = count+1;
        patch1 = I1(i:(i+7),j:(j+7));
        patch2 = I2(i:(i+7),j:(j+7));
        
        mean1 = sum(sum(patch1))/(unit*unit);
        mean2 = sum(sum(patch2))/(unit*unit);
        p1 = patch1-mean1;
        p2 = patch2-mean2;
        % 图像块向量
        Ve1(:,count) = p1(:);
        Ve2(:,count) = p2(:);
        % 计算权值矩阵 - 列向量
        Ve1_l2 = norm(p1(:),2);
        Ve2_l2 = norm(p2(:),2);
%         T_v(1:unit*unit, count) = 1/(1+exp((-1)*(Ve1_l2-Ve2_l2)));
        T_v(1:256, count) = 1/(1+exp((-1)*(Ve1_l2-Ve2_l2)));
        % 均值
        Me1(1:unit*unit, count) = mean1;
        Me2(1:unit*unit, count) = mean2;
        % 计算权值矩阵 - 平均值
%         Me1_l2 = 8*mean1;
%         Me2_l2 = 8*mean2;
        Me1_l2 = mean1;
        Me2_l2 = mean2;
        T_m(1:unit*unit, count) = 1/(1+exp((-1)*(Me1_l2-Me2_l2)));
    end
end

% JSR 分解
dic_size = 256;
% k = 16;
k = 5;
row_unit = unit*unit;
patch_num = count;

V_Joint = zeros(2*row_unit, patch_num);
V_Joint(1:(row_unit),:) = Ve1;
V_Joint((row_unit+1):(2*row_unit),:) = Ve2;

D_Joint = zeros(2*row_unit, dic_size*3);
D_Joint(1:row_unit,1:dic_size) = D/sqrt(2);
D_Joint(1:row_unit,(dic_size+1):2*dic_size) = D;
D_Joint((row_unit+1):2*row_unit,1:dic_size) = D/sqrt(2);
D_Joint((row_unit+1):2*row_unit,(dic_size*2+1):3*dic_size) = D;

disp('OMP-求解系数');
tic
C = omp(D_Joint, V_Joint,[], k);
toc
disp('OMP-求解系数 结束');

coe_c = (C(1:dic_size,:))/sqrt(2);
coe_u1 = C((dic_size+1):2*dic_size,:);
coe_u2= C((2*dic_size+1):3*dic_size,:);

% c_f = D*coe_c;
% u1_f = D*coe_u1;
% u2_f = D*coe_u2;
% 融合
% v_f = c_f + T_v.*u1_f + (1-T_v).*u2_f;
v_f = D*(coe_c + T_v.*coe_u1 + (1-T_v).*coe_u2);
m_f = T_m.*Me1 + (1-T_m).*Me2;
Ve_f = v_f + m_f;

disp('开始重构');
V_fusion = Ve_f;

TV_fusion = T_v;
TM_fusion = T_m;

VF_fusion = v_f;
MF_fusion = m_f;

fusion = zeros(m,n);

fusion_tv = zeros(m,n);
fusion_tm = zeros(m,n);

fusion_vf = zeros(m,n);
fusion_mf = zeros(m,n);

countt = 0;
for ii=1:(m-unit+1)
    for jj=1:(n-unit+1)
        countt = countt+1;
        patch1 = V_fusion(:, countt);
        pr = reshape(patch1,[unit,unit]);
        temp = fusion(ii:(ii+7),jj:(jj+7));
        fusion(ii:(ii+7),jj:(jj+7))=(pr+temp)/2;
        
        %权值-coe
        patch_tv = TV_fusion(:, countt);
        pr_tv = patch_tv(1);
        fusion_tv(ii,jj)=pr_tv;
        %权值-mean
        patch_tm = TM_fusion(:, countt);
        pr_tm = patch_tm(1);
        fusion_tm(ii,jj)=pr_tm;
        
        %融合系数
        patch_vf = VF_fusion(:, countt);
        pr_vf = reshape(patch_vf,[unit,unit]);
        temp_vf = fusion_vf(ii:(ii+7),jj:(jj+7));
        fusion_vf(ii:(ii+7),jj:(jj+7))=(pr_vf+temp_vf)/2;
        
        patch_mf = MF_fusion(:, countt);
        pr_mf = reshape(patch_mf,[unit,unit]);
        temp_mf = fusion_mf(ii:(ii+7),jj:(jj+7));
        fusion_mf(ii:(ii+7),jj:(jj+7))=(pr_mf+temp_mf)/2;
        
    end
end
disp('重构结束');

figure;imshow(fusion_tv);
figure;imshow(fusion_tm);

figure;imshow(1-fusion_tv);
figure;imshow(1-fusion_tm);

figure;imshow(fusion_vf);
figure;imshow(fusion_mf);

figure;imshow(fusion);

% imwrite(fusion,fused_path,'png');

% clear Ve1;
% clear Ve2;
% clear V_Joint;
% clear v_f;
% clear T_v;
% clear T_m;
% clear Me1;
% clear Me2;
% end


