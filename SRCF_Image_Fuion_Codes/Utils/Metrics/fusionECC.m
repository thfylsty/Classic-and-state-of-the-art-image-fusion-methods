function ECC = fusionECC(imgA,imgB,imgF)
% the overall entropy correlation coefficient (ECC) between source 
% images and fused image.
% 
% this measure has been used in paper: Image Fusion with Guilded Filtering
% Ref:
% J. Astola and I. Virtanen, “Entropy correlation coefficient, a measure
% of statistical dependence for categorized data,” in Proc. Univ. Vaasa,
% Discussion Papers, Finland, 1982, no. 44.
%--------------------------------------------------------------------
[pAF, pA, pF] = estpab(imgA,imgF);
MIAF = estmutualinfo(pAF, pA, pF);

[pBF, pB, pF] = estpab(imgB,imgF);
MIBF = estmutualinfo(pBF, pB, pF);

% pA = estpa(A);
HA = estentropy(pA);
HB = estentropy(pB);
HF = estentropy(pF);

ECCaf = 2*MIAF/(HA+HF);
ECCbf = 2*MIBF/(HB+HF);
ECC = ECCaf + ECCbf;
return