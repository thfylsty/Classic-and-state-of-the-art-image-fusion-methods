function I = mutualinfo2(A,B)
%==================================================
%This is a prog in the MutualInfo 0.9 package written by 
% Hanchuan Peng.
%
% I = mutualinfo(vec1,vec2)
% calculate the mutual information of two vectors
% 
% 
% For larg images this function is faster than 
% function 'mutualinfo1'. 
%==================================================

[pAB, pA, pB] = estpab(A,B);
I = estmutualinfo(pAB, pA, pB);
return