function [ img_ihs] = rgb2ihs( img_rgb )
r=img_rgb(:,:,1);
g=img_rgb(:,:,2);
b=img_rgb(:,:,3);

% rgbתihs
I=1/sqrt(3).*r+1/sqrt(3).*g+1/sqrt(3).*b;
v1=1/sqrt(6).*r+1/sqrt(6).*g-2/sqrt(6).*b;
v2=1/sqrt(2).*r-1/sqrt(2).*g;
img_ihs=cat(3,I,v1,v2);
end

