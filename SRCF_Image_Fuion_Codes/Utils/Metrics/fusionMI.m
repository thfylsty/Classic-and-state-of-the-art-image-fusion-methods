function fMI = fusionMI(imgA,imgB,imgF)
% the overall mutual information between source images and fused image
%--------------------------------------------------------------------
[pAF, pA, pF] = estpab(imgA,imgF);
MIAF = estmutualinfo(pAF, pA, pF);

[pBF, pB, pF] = estpab(imgB,imgF);
MIBF = estmutualinfo(pBF, pB, pF);

fMI = MIAF + MIBF;
return