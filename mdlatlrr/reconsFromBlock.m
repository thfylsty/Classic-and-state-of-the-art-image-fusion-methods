% reconstructioin from image block - 4
function F = getImgBlock(img_temp,img1, img2, img3, img4)

count = img_temp;
[h, w] = size(img_temp);
temp_ones = ones(h, w);
h_cen = floor(h/2);
w_cen = floor(w/2);

img_temp(1:h_cen+2, 1:w_cen+2) = img_temp(1:h_cen+2, 1:w_cen+2)+img1;
count(1:h_cen+2, 1:w_cen+2) = count(1:h_cen+2, 1:w_cen+2)+temp_ones(1:h_cen+2, 1:w_cen+2);

img_temp(1:h_cen+2, w_cen-1:w) = img_temp(1:h_cen+2, w_cen-1:w)+img2;
count(1:h_cen+2, w_cen-1:w) = count(1:h_cen+2, w_cen-1:w)+temp_ones(1:h_cen+2, w_cen-1:w);

img_temp(h_cen-1:h, 1:w_cen+2) = img_temp(h_cen-1:h, 1:w_cen+2)+img3;
count(h_cen-1:h, 1:w_cen+2) = count(h_cen-1:h, 1:w_cen+2)+temp_ones(h_cen-1:h, 1:w_cen+2);

img_temp(h_cen-1:h, w_cen-1:w) = img_temp(h_cen-1:h, w_cen-1:w)+img4;
count(h_cen-1:h, w_cen-1:w) = count(h_cen-1:h, w_cen-1:w)+temp_ones(h_cen-1:h, w_cen-1:w);

F = img_temp./count;
end