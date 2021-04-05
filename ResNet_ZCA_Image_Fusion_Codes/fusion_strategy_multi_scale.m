% stable weight values
function [gen, soft_a, soft_b] = fusion_strategy_multi_scale(features_a, features_b, source_a, source_b)

[h,w, c] = size(features_a);

weight = zeros(h,w,c);
weight(:,:,1) = 1/(1+exp(-1)+exp(-2)+exp(-3));
weight(:,:,2) = exp(-1)/(1+exp(-1)+exp(-2)+exp(-3));
weight(:,:,3) = exp(-2)/(1+exp(-1)+exp(-2)+exp(-3));
weight(:,:,4) = exp(-3)/(1+exp(-1)+exp(-2)+exp(-3));
features_a_sum = sum((features_a.*weight), 3);
features_b_sum = sum((features_b.*weight), 3);

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

