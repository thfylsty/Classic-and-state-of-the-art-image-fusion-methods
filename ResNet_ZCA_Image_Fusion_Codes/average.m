for i=1:21
    index = i;
    disp(num2str(index));
    path1 = ['./IV_images/IR',num2str(index),'.png'];
    path2 = ['./IV_images/VIS',num2str(index),'.png'];
    fuse_path_ave = ['./fused_infrared/fused',num2str(index),'_average.png'];
    
    image1 = im2double(imread(path1));
    image2 = im2double(imread(path2));

    image_ave = (image1+image2)/2;

    imwrite(image_ave,fuse_path_ave,'png');
end
