function [I_b, I_d, I_d_v, count_matric, I_vector] = vector_decomposition(img, L, unit, is_overlap, stride, w_num)
% lrr parts and salient parts
% stride = 4;
% image, using reflecting to resize input image
[t1,t2] = size(img);
h = t1;
w = t2;
s = stride;

% figure;imshow(img_or);
% figure;imshow(img);

% salient parts by latlrr
I_d = zeros(h, w);
% the matrices for overlapping
count_matric = zeros(h, w);
ones_matric = ones(unit,unit);

if is_overlap == 1
    c1 = 0;
    for i=1:s:h-unit+1
        c2 = 0;
        c1 = c1+1;
        for j=1:s:w-unit+1
            c2 = c2+1;
            temp = img(i:(i+unit-1), j:(j+unit-1));
            temp_vector(:,(c1-1)*(w_num)+c2) = temp(:);
            % record the overlapping number
            count_matric(i:(i+unit-1), j:(j+unit-1)) =...
                    count_matric(i:(i+unit-1), j:(j+unit-1)) + ones_matric(:, :);
        end
    end
    img_vector = temp_vector;
    % calculate features
    I_d_v = L*temp_vector;
    c1 = 0;
    for ii=1:s:h-unit+1
        c2 = 0;
        c1 = c1+1;
        for jj=1:s:w-unit+1
            c2 = c2+1;
            temp = I_d_v(:, (c1-1)*(w_num)+c2);
            I_d(ii:(ii+unit-1), jj:(jj+unit-1)) = I_d(ii:(ii+unit-1), jj:(jj+unit-1)) + reshape(temp, [unit unit]);
        end
    end
    % average operation for overlapping position
    I_d = I_d./count_matric;
%     I_d = I_d(1:t1, 1:t2);
else
    % stride = unit
    I_col = im2col(img, [unit, unit], 'distinct');
    salient = L*I_col;
    I_d_v = salient;

    I_d = col2im(salient,[unit, unit],[t1,t2],'distinct');
    I_d(I_d<0) = 0;
end
% base parts
I_b = img - I_d;
I_vector = img_vector;
end