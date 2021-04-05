% whitening operation
function output = whitening_norm_multi_scale(features, image)
    features = im2double(features);
    output = whitening_zca(features);
    output = nuclear(output, image);
%     output = l1_norm(output);
end

function output = nuclear(features, image)
    [h,w,c] = size(features);
    [h1, w1] = size(image);
    output = zeros(h, w);
    unit = 5;
    if c>512
        unit = 3;
    end
    % padding
    pad = (unit-1)/2;
    feature_temp = zeros(h+unit-1, w+unit-1, c);
    feature_temp(1+pad:h+pad, 1+pad:w+pad, :) = features;
    for i=1+pad:h+pad
        for j=1+pad:w+pad
            temp = reshape(feature_temp(i-pad:i+pad, j-pad:j+pad, :), [unit*unit c]);
            [U, S, V] = svd(temp, 'econ');
            nu_norm = sum(diag(S));
            output(i-pad, j-pad) = nu_norm;
        end
    end
    output = imresize(output, [h1,w1]);
end

function output = l1_norm(features, image)
    [h,w,c] = size(features);
    [h1, w1] = size(image);
    output = zeros(h, w);
    unit = 5;
%     if c>512
%         unit = 3;
%     end
    % padding
    pad = (unit-1)/2;
    feature_temp = zeros(h+unit-1, w+unit-1, c);
    feature_temp(1+pad:h+pad, 1+pad:w+pad, :) = features;
    for i=1+pad:h+pad
        for j=1+pad:w+pad
            temp = feature_temp(i-pad:i+pad, j-pad:j+pad, :);
            norm = sum(sum(sum(abs(temp),3)))/(unit*unit);
            output(i-pad, j-pad) = norm;
        end
    end
    output = imresize(output, [h1,w1]);
end



