function dI = downSample(I, maxSize)

if ~exist('maxSize', 'var')
   maxSize = 512;
end

I = double(I);
m = size(I, 1);
n = size(I, 2);

if m >= n && m > maxSize
    sampleFactor = m / maxSize;
    dI = imresize(I, [maxSize, floor(n / sampleFactor)],'bicubic');
elseif m < n && n > maxSize
    sampleFactor = n / maxSize;
    dI = imresize(I, [floor(m / sampleFactor), maxSize],'bicubic');
else
    dI = I;
end


