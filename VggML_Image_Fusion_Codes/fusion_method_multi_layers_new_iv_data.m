%

clear all
clc

addpath(genpath('D:\develop\matconvnet\matlab\')); % matconvnet path
%load vgg19
net = load('./models/imagenet-vgg-verydeep-19.mat');
net = vl_simplenn_tidy(net);

% testing dataset
test_path_ir = './IV_images/IR/';
fileFolder_ir=fullfile(test_path_ir);
dirOutput_ir =dir(fullfile(fileFolder_ir,'*'));
num_ir = length(dirOutput_ir);

test_path_vis = replace(test_path_ir, '/IR/', '/VIS/');
fileFolder_vis=fullfile(test_path_vis);
dirOutput_vis =dir(fullfile(fileFolder_vis,'*'));

for i=3:num_ir
index = i;
disp(num2str(index));

path1 = [test_path_ir,dirOutput_ir(i).name]; % IR image
path2 = [test_path_vis,dirOutput_vis(i).name]; % VIS image
fused_path = ['./fused_iv/fused_vggml_',dirOutput_ir(i).name];

image1 = imread(path1);
image2 = imread(path2);
image1 = im2double(image1);
image2 = im2double(image2);

[h,w,d] = size(image1);
isRe = 0;
if d > 1
    image1 = rgb2gray(image1);
    image2 = rgb2gray(image2);
end
if h < 224
    isRe = 1;
    image1 = imresize(image1, [224, w]);
    image2 = imresize(image2, [224, w]);
end
if w < 224
    isRe = 1;
    image1 = imresize(image1, [h, 224]);
    image2 = imresize(image2, [h, 224]);
end

tic;
% Highpass filter test image
npd = 16;
fltlmbd = 5;
[I_lrr1, I_saliency1] = lowpass(image1, fltlmbd, npd);
[I_lrr2, I_saliency2] = lowpass(image2, fltlmbd, npd);

%% fuison lrr parts
F_lrr = (I_lrr1+I_lrr2)/2;
% figure;imshow(F_lrr);

%% fuison saliency parts use VGG19
% disp('VGG19-saliency');
saliency_a = make_3c(I_saliency1);
saliency_b = make_3c(I_saliency2);
saliency_a = single(saliency_a) ; % note: 255 range
saliency_b = single(saliency_b) ; % note: 255 range

res_a = vl_simplenn(net, saliency_a);
res_b = vl_simplenn(net, saliency_b);

%% relu1_1
% disp('relu1_1');
out_relu1_1_a = res_a(2).x;
out_relu1_1_b = res_b(2).x;
unit_relu1_1 = 1;

l1_featrues_relu1_a = extract_l1_feature(out_relu1_1_a);
l1_featrues_relu1_b = extract_l1_feature(out_relu1_1_b);
% average
[m,n,k] = size(out_relu1_1_a);

[F_saliency_relu1, l1_featrues_relu1_ave_a, l1_featrues_relu1_ave_b] = ...
            fusion_strategy(l1_featrues_relu1_a, l1_featrues_relu1_b, I_saliency1, I_saliency2, unit_relu1_1);
        
%% relu2_1
% disp('relu2_1');
out_relu2_1_a = res_a(7).x;
out_relu2_1_b = res_b(7).x;
unit_relu2_1 = 2;

l1_featrues_relu2_a = extract_l1_feature(out_relu2_1_a);
l1_featrues_relu2_b = extract_l1_feature(out_relu2_1_b);

[F_saliency_relu2, l1_featrues_relu2_ave_a, l1_featrues_relu2_ave_b] = ...
            fusion_strategy(l1_featrues_relu2_a, l1_featrues_relu2_b, I_saliency1, I_saliency2, unit_relu2_1);

%% relu3_1
% disp('relu3_1');
out_relu3_1_a = res_a(12).x;
out_relu3_1_b = res_b(12).x;
unit_relu3_1 = 4;

l1_featrues_relu3_a = extract_l1_feature(out_relu3_1_a);
l1_featrues_relu3_b = extract_l1_feature(out_relu3_1_b);

[F_saliency_relu3, l1_featrues_relu3_ave_a, l1_featrues_relu3_ave_b] = ...
            fusion_strategy(l1_featrues_relu3_a, l1_featrues_relu3_b, I_saliency1, I_saliency2, unit_relu3_1);

%% relu4_1
% disp('relu4_1');
out_relu4_1_a = res_a(21).x;
out_relu4_1_b = res_b(21).x;
unit_relu4_1 = 8;

l1_featrues_relu4_a = extract_l1_feature(out_relu4_1_a);
l1_featrues_relu4_b = extract_l1_feature(out_relu4_1_b);

[F_saliency_relu4, l1_featrues_relu4_ave_a, l1_featrues_relu4_ave_b] = ...
            fusion_strategy(l1_featrues_relu4_a, l1_featrues_relu4_b, I_saliency1, I_saliency2, unit_relu4_1);

%% fusion strategy
% disp('choose max');
F_saliency = max(F_saliency_relu1, F_saliency_relu2);
F_saliency = max(F_saliency, F_saliency_relu3);
F_saliency = max(F_saliency, F_saliency_relu4);
% figure;imshow(F_saliency);

% imwrite(F_saliency,'./decomposition/F_s.png','png');

fusion_im = F_lrr + F_saliency;
toc;
% figure;imshow(fusion_im);
if isRe==1
    fusion_im = imresize(fusion_im, [h, w]);
end
imwrite(fusion_im,fused_path,'png');
end


