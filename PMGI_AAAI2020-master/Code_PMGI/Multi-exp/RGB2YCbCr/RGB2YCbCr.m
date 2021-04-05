function [Y,Cb,Cr]=RGB2YCbCr(img_rgb)
R=img_rgb(:,:,1);
G=img_rgb(:,:,2);
B=img_rgb(:,:,3);

% rgbתihs
Y   = 0.257*R+0.564*G+0.098*B+16;
Cb = -0.148*R-0.291*G+0.439*B+128;
Cr  = 0.439*R-0.368*G-0.071*B+128;

end

