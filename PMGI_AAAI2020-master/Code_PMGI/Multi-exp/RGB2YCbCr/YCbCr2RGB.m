function [image_RGB]=YCbCr2RGB(img_YCbCr)
Y=img_YCbCr(:,:,1);
Cb=img_YCbCr(:,:,2);
Cr=img_YCbCr(:,:,3);

% rgbתihs
R = 1.164*(Y-16)+1.596*(Cr-128);
G = 1.164*(Y-16)-0.392*(Cb-128)-0.813*(Cr-128);
B = 1.164*(Y-16)+2.017*(Cb-128);

image_RGB=cat(3,R,G,B);
end

