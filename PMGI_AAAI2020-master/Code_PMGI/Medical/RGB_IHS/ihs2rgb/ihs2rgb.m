function [ img_rgb ] = ihs2rgb( img_ihs)
I=img_ihs(:,:,1);
V1=img_ihs(:,:,2);
V2=img_ihs(:,:,3);

%rgbתihs
% r=1/sqrt(3)*I+1/sqrt(6)*S.*sin(H)+1/sqrt(2)*S.*cos(H);
% g=1/sqrt(3)*I+1/sqrt(6)*S.*sin(H)-1/sqrt(2)*S.*cos(H);
% b=1/sqrt(3)*I-2/sqrt(6)*S.*sin(H);
r=1/sqrt(3)*I+1/sqrt(6)*V1+1/sqrt(2)*V2;
g=1/sqrt(3)*I+1/sqrt(6)*V1-1/sqrt(2)*V2;
b=1/sqrt(3)*I-2/sqrt(6)*V1;
img_rgb=cat(3,r,g,b);
end

