%% 第一次高斯平滑

%设置高斯平滑滤波模板
% h = fspecial('gaussian',3,7);
disp(strcat('开始'));
for i = 6:9
    fileName1 = ['image',num2str(i),'_left.png'];
    I1 = imread(fileName1);
    image1 = double(I1);
    [m,n] = size(image1);
    m1 = floor(m/2);
    n1 = floor(n/2);
    image1 = imresize(image1,[m1 n1]);
    image1 = uint8(image1);
% %     image1_noise = imnoise(I1,'gaussian',0,0.005);
%     image1_noise = imnoise(image1,'salt & pepper',0.05);
    imagewrite = ['image',num2str(i),'_left_re.png'];

    figure;
    imshow(image1);
    imwrite(image1,imagewrite,'png');
    
    fileName2 = ['image',num2str(i),'_right.png'];
    I2 = imread(fileName2);
    image2 = double(I2);
    [m,n] = size(image2);
    m1 = floor(m/2);
    n1 = floor(n/2);
    image2 = imresize(image2,[m1 n1]);
    image2 = uint8(image2);
% %     image1_noise = imnoise(I1,'gaussian',0,0.005);
%     image2_noise = imnoise(image2,'salt & pepper',0.05);
    imagewrite = ['image',num2str(i),'_right_re.png'];

    figure;
    imshow(image2);
    imwrite(image2,imagewrite,'png');
end
disp(strcat('完成'));

