function I = mutualinfo1(A,B)
%==================================================
%This is a prog in the MutualInfo 0.9 package written by 
% Hanchuan Peng.
%
% I = mutualinfo(vec1,vec2)
% calculate the mutual information of two vectors
% 
% For small images this function is faster than 
% function 'mutualinfo2'. 
%==================================================

pA = estpa(A);
HA = estentropy(pA);

pB = estpa(B);
HB = estentropy(pB);

pAB = estpab(A,B);
HAB = estjointentropy(pAB);

I = HA + HB - HAB;
return