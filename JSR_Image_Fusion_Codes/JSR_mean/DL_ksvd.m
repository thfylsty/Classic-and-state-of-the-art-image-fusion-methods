%% 使用KSVD字典学习算法，学习到字典D，大小为64*256
% 图像集使用USC-SIPI 数据集

path_file = './USC_SIPI_database/';

path_name = [path_file,'*.tiff'];
fileP=dir(path_name);
fileCount = length(fileP); % 文件数量
% 图像块大小为8
unit = 8;
v_num = 50;

V = zeros(unit*unit, v_num*fileCount);
disp(strcat('开始'));
for index = 1:fileCount
    str = ['index = ',num2str(index)];
    disp(strcat(str));
    path = [path_file,fileP(index).name];
    I = imread(path);
    d = size(I,3);
    if d>1
        I = rgb2gray(I);
    end
    I = im2double(I);
    Vi = patchVector(I, unit, v_num);
    V(:, ((index-1)*v_num+1):(index*v_num)) = Vi;
end
save('V_image_patch.dat','V');

dic_size = 256;
k=5;
disp('KSVD-计算字典 开始');
params.data = V;
params.Tdata = k;
params.dictsize = dic_size;
params.iternum = 50;
params.memusage = 'high';
[D,X,err] = ksvd(params,'');
disp('KSVD-计算字典 结束');

save('D_k5_ksvd.dat','D');











