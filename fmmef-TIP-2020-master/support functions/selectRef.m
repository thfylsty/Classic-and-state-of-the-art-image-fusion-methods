function idx = selectRef(I, winSize, exposureThres)

if ~exist('winSize', 'var')
   winSize = 3;
end

if ~exist('exposureThres', 'var')
   exposureThres = 0.01;
end

I = double(I);
I = reorderByLum(I);
[~, ~, s3, s4] = size(I);

if s4 == 3
    idx = 2;
else
    window = ones(winSize, winSize, s3);
    window = window / sum(window(:));
    p = zeros(s4,1);
    for i = 1 : s4
        mu = convn(I(:, :, :, i), window, 'valid');
        p(i) = sum( sum( mu < exposureThres | mu > 1 - exposureThres) );
    end
    [~, idx] = min(p);
end

