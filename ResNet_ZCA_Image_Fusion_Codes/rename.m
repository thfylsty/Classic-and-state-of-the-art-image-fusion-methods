for i=1:21
    index = i;
    disp(num2str(index));

    fuse_path1 = ['./fused_infrared/fused',num2str(index),'_vgg19_zca1.png'];
    fuse_path2 = ['./fused_infrared/fused',num2str(index),'_vgg19_zca2.png'];
    fuse_path3 = ['./fused_infrared/fused',num2str(index),'_vgg19_zca3.png'];
    fuse_path4 = ['./fused_infrared/fused',num2str(index),'_vgg19_zca4.png'];
%     fuse_path5 = ['./fused_infrared/fused',num2str(index),'_vgg19_zca5.png'];
    
    image1 = imread(fuse_path1);
    image2 = imread(fuse_path2);
    image3 = imread(fuse_path3);
    image4 = imread(fuse_path4);
%     image5 = imread(fuse_path5);

    imwrite(image1,['./fused_infrared/fused',num2str(index),'_vgg19_nu_zca1.png'],'png');
    imwrite(image2,['./fused_infrared/fused',num2str(index),'_vgg19_nu_zca2.png'],'png');
    imwrite(image3,['./fused_infrared/fused',num2str(index),'_vgg19_nu_zca3.png'],'png');
    imwrite(image4,['./fused_infrared/fused',num2str(index),'_vgg19_nu_zca4.png'],'png');
%     imwrite(image5,['./fused_infrared/fused',num2str(index),'_vgg19_nu_zca5.png'],'png');

    delete(fuse_path1);
    delete(fuse_path2);
    delete(fuse_path3);
    delete(fuse_path4);
%     delete(fuse_path5);
end
