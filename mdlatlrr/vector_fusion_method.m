function F = vector_fusion_method(img1_or, img2_or, L, unit, isZCA, de_level, norm, is_overlap, stride)
% fusion method - lrr parts and salient parts

[t1,t2] = size(img1_or);
% calculate image patch number, h_num and w_num
h = t1;
w = t2;
s = stride;
if is_overlap == 1
    % resize input image
    h_dis = ceil((t1-unit)/s)*(s) - (t1-unit);
    w_dis = ceil((t2-unit)/s)*(s) - (t2-unit);
    h = h_dis + t1;
    w = w_dis + t2;
    h_num = ceil((h-unit)/s)+1;
    w_num = ceil((w-unit)/s)+1;
end
img1 = zeros(h, w);
img2 = zeros(h, w);
img1(1:t1, 1:t2) = img1_or(:,:);
img2(1:t1, 1:t2) = img2_or(:,:);
if h_dis ~= 0
    img1(t1:t1+h_dis, 1:t2) = img1_or(t1-h_dis:end, 1:end);
    img2(t1:t1+h_dis, 1:t2) = img2_or(t1-h_dis:end, 1:end);
end
if w_dis ~= 0
    img1(:, t2:t2+w_dis) = img1(:, t2-w_dis:t2);
    img2(:, t2:t2+w_dis) = img2(:, t2-w_dis:t2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% is_overlap = 0;
[I_b1,I_d1_deep, show_matrix1, count1, I_vector1] = vector_deep_latent(img1, de_level, L, unit, is_overlap, stride, w_num);
[I_b2,I_d2_deep, show_matrix2, count2, I_vector2] = vector_deep_latent(img2, de_level, L, unit, is_overlap, stride, w_num);
count_all = count1;
% fuion for base parts
I_bf = fuison_base_parts(I_b1, I_b2);
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fuion for salient parts
if is_overlap==1
    I_d_temp = zeros(h, w);
    for i=1:de_level
        temp1 = I_d1_deep(:,:,i);
        temp2 = I_d2_deep(:,:,i);
        [temp_vector] = vector_fuison_detail_parts(temp1, temp2, norm, isZCA);
        c1 = 0;
        for ii=1:s:h-unit+1
            c2 = 0;
            c1 = c1+1;
            for jj=1:s:w-unit+1
                c2 = c2+1;
                temp = temp_vector(:, (c1-1)*(w_num)+c2);
                I_d_temp(ii:(ii+unit-1), jj:(jj+unit-1)) = I_d_temp(ii:(ii+unit-1), jj:(jj+unit-1)) + reshape(temp, [unit unit]);
            end
        end
        % average operation for overlapping position
        I_df_temp(:,:,i) = I_d_temp./count_all;
    end
    I_df = sum(I_df_temp,3);
else
    if de_level>1
        temp1 = sum(I_d1_deep,3);
        temp2 = sum(I_d2_deep,3);
    else
        temp1 = I_d1_deep;
        temp2 = I_d2_deep;
    end

    I_df_vector = vector_fuison_detail_parts(temp1, temp2, norm, isZCA);

    [t1,t2] = size(img1);
    I_df = col2im(I_df_vector,[unit, unit],[t1,t2],'distinct');
    I_df(I_df<0) = 0;
end
% figure;imshow(I_df);

% fused image
F = I_bf + I_df;
F = F(1:t1, 1:t2);
% figure;imshow(F);

end

