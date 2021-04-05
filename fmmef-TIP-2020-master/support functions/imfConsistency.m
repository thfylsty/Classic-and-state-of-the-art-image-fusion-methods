function cMap = imfConsistency(mu, refIdx, consistencyThres)

[s1, s2, s3] = size(mu);
cMap = zeros(s1, s2, s3);
cMap(:,:,refIdx) = ones(s1, s2);

refMu = mu(:,:,refIdx);
N = 256;
for i = 1 : s3
    if i ~= refIdx
          cMu  = imhistmatch(mu(:,:,i), refMu, N);
          diff = abs(cMu - refMu);
          cMap(:,:,i) = diff <= consistencyThres;          
    end
end

          
        









