%% gaussian filtrer
h = fspecial('gaussian',3,7);
disp(strcat('开始'));
for i = 1:20
    disp(num2str(i));
    fileName = ['image_new_',num2str(i),'.png'];
    image = imread(fileName);
    image_noise = filter2(h,image);
    
    image = double(image);
    image_noise = double(image_noise);
    
    [r,c] = size(image);
    left = image;
    right = image;
    
    uc = floor(c/7);
    u = floor(r/10);
    te = zeros(r,c);
    
    step = 5;
    for m=1:r
        t = 3*uc+uc*sin(m*0.02);
        se = round(t);
        te(m,1:se) = 1;
    end
    
    half1 = image_noise.*te;
    half2 = image.*(1-te);
    f1 = half1+half2;
    imagewrite1 = ['../image',num2str(i),'_left.png'];
    imwrite(uint8(f1),imagewrite1,'png');
    
    half3 = image_noise.*(1-te);
    half4 = image.*(te);
    f2 = half3+half4;
    imagewrite1 = ['../image',num2str(i),'_right.png'];
    imwrite(uint8(f2),imagewrite1,'png');
    
end
disp(strcat('完成'));












