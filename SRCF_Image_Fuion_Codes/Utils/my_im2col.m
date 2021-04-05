function [blocks,idx] = my_im2col(I,blkSize,slidingDis)
if (slidingDis==1)
    blocks = im2col(I,blkSize,'sliding');
    idx = [1:size(blocks,2)];
    return
end

idxMat = zeros(size(I)-blkSize+1);
idxMat([[1:slidingDis:end-1],end],[[1:slidingDis:end-1],end]) = 1; % take blocks in distances of 'slidingDix', but always take the first and last one (in each row and column).
idx = find(idxMat);
[rows,cols] = ind2sub(size(idxMat),idx);
blocks = zeros(prod(blkSize),length(idx));
for i = 1:length(idx)
    currBlock = I(rows(i):rows(i)+blkSize(1)-1,cols(i):cols(i)+blkSize(2)-1);
    blocks(:,i) = currBlock(:);
end
