% get image block - 4
function [img1, img2, img3, img4] = getImgBlock(img)

[h, w] = size(img);
h_cen = floor(h/2);
w_cen = floor(w/2);

img1 = img(1:h_cen+2, 1:w_cen+2);
img2 = img(1:h_cen+2, w_cen-1:w);
img3 = img(h_cen-1:h, 1:w_cen+2);
img4 = img(h_cen-1:h, w_cen-1:w);

% figure;imshow(img1);
% figure;imshow(img2);
% figure;imshow(img3);
% figure;imshow(img4);
% 
% F1 = cat(2,img1, img2);
% figure;imshow(F1);
% F2 = cat(2,img3, img4);
% figure;imshow(F2);
% F = cat(1,F1, F2);
% figure;imshow(F);

end
