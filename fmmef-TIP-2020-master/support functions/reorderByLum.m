function I = reorderByLum(I)

I = double(I);
m = squeeze(sum(sum(sum(I, 1), 2), 3));
[~, idx] = sort(m);
I = I(:, :, :, idx);