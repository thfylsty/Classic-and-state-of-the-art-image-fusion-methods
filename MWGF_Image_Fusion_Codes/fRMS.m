function [res]=fRMS(f1, f0)

[m,n]=size(f1);

s=(m-2)*(n-2);

f1=f1(2: m-1, 2:n-1);
f0=f0(2: m-1, 2:n-1);

d=f1-f0;
d=d(:);

res=d'*d/s;
