function [I_b_deep,I_d_deep, show_matrix, count_all, I_vector] = vector_deep_latent(img, de_level, L, unit, is_overlap, stride, w_num)

[t1,t2] = size(img);
show_matrix = [];
% I_d = zeros(t1,t2,de_level);
I_b_deep = zeros(t1,t2);
for i=1:de_level
    if i==1
        [temp_b, temp_d, temp_d_v, count_m, I_vector] = vector_decomposition(img, L, unit, is_overlap, stride, w_num);
    else
        [temp_b, temp_d, temp_d_v, count_m, I_vector] = vector_decomposition(temp_b, L, unit, is_overlap, stride, w_num);
    end
    I_d_deep(:,:,i) = temp_d_v(:,:);
    show_matrix = cat(2,show_matrix, temp_d);
    
end
% show_matrix = cat(2,show_matrix, temp_b);
% figure;imshow(show_matrix);
% imwrite(temp_b,['./fused_vector/features/base_level_',num2str(i),'.png'],'png');

I_b_deep(:,:) = temp_b(:,:);
count_all = count_m;
% I_d_deep = I_d;

end