% auto weighting
function [gen, soft_a, soft_b] = fusion_strategy_multi_scale_auto(features_a, features_b, source_a, source_b)

[h,w, c] = size(features_a);

k = [[1,1,1], [1,1,1], [1,1,1]];
features_a_l1 = convn(abs(features_a), k, 'same');
features_b_l1 = convn(abs(features_b), k, 'same');

sum_a_l1 = zeros(h,w,c);
sum_a_l1(:,:,1) = sum(features_a_l1, 3);
sum_a_l1(:,:,2) = sum(features_a_l1, 3);
sum_a_l1(:,:,3) = sum(features_a_l1, 3);
sum_a_l1(:,:,4) = sum(features_a_l1, 3);

sum_b_l1 = zeros(h,w,c);
sum_b_l1(:,:,1) = sum(features_b_l1, 3);
sum_b_l1(:,:,2) = sum(features_b_l1, 3);
sum_b_l1(:,:,3) = sum(features_b_l1, 3);
sum_b_l1(:,:,4) = sum(features_b_l1, 3);

weight_a = features_a_l1./sum_a_l1;
weight_b = features_b_l1./sum_b_l1;

features_a_sum = sum((features_a.*weight_a), 3);
features_b_sum = sum((features_b.*weight_b), 3);

soft_a = features_a_sum./(features_a_sum+features_b_sum);
soft_b = features_b_sum./(features_a_sum+features_b_sum);

% weight_a_cat = cat(2, source_a, features_a_sum);
% weight_a_cat = cat(2, weight_a_cat, soft_a);
% weight_b_cat = cat(2, source_b, features_b_sum);
% weight_b_cat = cat(2, weight_b_cat, soft_b);
% weight_cat = cat(1, weight_a_cat, weight_b_cat);
% figure;imshow(weight_cat);
% imwrite(weight_cat, './figures/image2_weight_multi_scale_soft.png', 'png');
gen = soft_a.*source_a + soft_b.*source_b;
% figure;imshow(gen);
end

