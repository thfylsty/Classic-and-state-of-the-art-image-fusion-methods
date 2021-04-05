%% VGG19 + ZCA & nuclear norm
%load vgg19
net = load('./models/imagenet-vgg-verydeep-19.mat');
net = vl_simplenn_tidy(net);

n = 21;
time = zeros(n,1);
for i=1:n
disp(num2str(i));
index = i;

path1 = ['./IV_images/IR',num2str(index),'.png'];
path2 = ['./IV_images/VIS',num2str(index),'.png'];

% l1-norm
% fuse_path1 = ['./fused_infrared/fused',num2str(index),'_vgg19_l1_zca1.png'];
% fuse_path2 = ['./fused_infrared/fused',num2str(index),'_vgg19_l1_zca2.png'];
% fuse_path3 = ['./fused_infrared/fused',num2str(index),'_vgg19_l1_zca3.png'];
% fuse_path4 = ['./fused_infrared/fused',num2str(index),'_vgg19_l1_zca4.png'];

% l2-norm
fuse_path1 = ['./fused_infrared/fused',num2str(index),'_vgg19_l2_zca1.png'];
fuse_path2 = ['./fused_infrared/fused',num2str(index),'_vgg19_l2_zca2.png'];
fuse_path3 = ['./fused_infrared/fused',num2str(index),'_vgg19_l2_zca3.png'];
fuse_path4 = ['./fused_infrared/fused',num2str(index),'_vgg19_l2_zca4.png'];

% nuclear norm
% fuse_path1 = ['./fused_infrared/fused',num2str(index),'_vgg19_nu_zca1.png'];
% fuse_path2 = ['./fused_infrared/fused',num2str(index),'_vgg19_nu_zca2.png'];
% fuse_path3 = ['./fused_infrared/fused',num2str(index),'_vgg19_nu_zca3.png'];
% fuse_path4 = ['./fused_infrared/fused',num2str(index),'_vgg19_nu_zca4.png'];


% path1 = ['./MF_images/image',num2str(index),'_left.png'];
% path2 = ['./MF_images/image',num2str(index),'_right.png'];
% fuse_path = ['./fused_multifocus/fused',num2str(index),'_vgg19_zca.png'];

image1 = imread(path1);
image2 = imread(path2);
image1 = im2double(image1);
image2 = im2double(image2);

tic;
%% Extract features
disp('ResNet-saliency');
if size(image1, 3)<3
    I1 = make_3c(image1);
end
if size(image2, 3)<3
    I2 = make_3c(image2);
end
I1 = single(I1) ; % note: 255 range
I2 = single(I2) ; % note: 255 range
%% VGG19
disp('VGG19');
res_a = vl_simplenn(net, I1);
res_b = vl_simplenn(net, I2);

%% relu1_1
disp('relu1_1');
out_relu1_1_a = res_a(2).x;
out_relu1_1_b = res_b(2).x;
disp('relu2_1');
out_relu2_1_a = res_a(7).x;
out_relu2_1_b = res_b(7).x;
disp('relu3_1');
out_relu3_1_a = res_a(12).x;
out_relu3_1_b = res_b(12).x;
disp('relu4_1');
out_relu4_1_a = res_a(21).x;
out_relu4_1_b = res_b(21).x;

%% extract features - whitening operation
disp('extract features(whitening operation) - I1');
feature1_1 = whitening_norm(out_relu1_1_a);
feature2_1 = whitening_norm(out_relu2_1_a);
feature3_1 = whitening_norm(out_relu3_1_a);
feature4_1 = whitening_norm(out_relu4_1_a);
disp('extract features(whitening operation) - I2');
feature1_2 = whitening_norm(out_relu1_1_b);
feature2_2 = whitening_norm(out_relu2_1_b);
feature3_2 = whitening_norm(out_relu3_1_b);
feature4_2 = whitening_norm(out_relu4_1_b);

%% fusion strategy
[F_relu1, weight1_a, weight1_b] = fusion_strategy(feature1_1, feature1_2, image1, image2);
[F_relu2, weight2_a, weight2_b] = fusion_strategy(feature2_1, feature2_2, image1, image2);
[F_relu3, weight3_a, weight3_b] = fusion_strategy(feature3_1, feature3_2, image1, image2);
[F_relu4, weight4_a, weight4_b] = fusion_strategy(feature4_1, feature4_2, image1, image2);
time(i) = toc;
% figure;imshow(fusion_im);

imwrite(F_relu1,fuse_path1,'png');
imwrite(F_relu2,fuse_path2,'png');
imwrite(F_relu3,fuse_path3,'png');
imwrite(F_relu4,fuse_path4,'png');
end


