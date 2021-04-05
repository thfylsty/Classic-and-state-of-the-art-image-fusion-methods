function result=edge_association(A,B,F)
% Author:  Qu Xiao-Bo    <quxiaobo [at] xmu.edu.cn>  and Hu Chang-wei Hu , June 26, 2009
%          Postal address:
% Rom 509, Scientific Research Building # 2,Haiyun Campus, Xiamen University,Xiamen,Fujian, P. R. China, 361005
% Website: http://quxiaobo.go.8866.org

%reference paper:Objective image fusion performance measure
A=double(A);
B=double(B);
F=double(F);
[row,column]=size(A);
[gA,aA]=get_g_a(A);
[gB,aB]=get_g_a(B);
[gF,aF]=get_g_a(F);
GAF=get2G(gA,gF);
GBF=get2G(gB,gF);
AAF=get2A(aA,aF);
ABF=get2A(aB,aF);
QgAF=getQg(GAF,0.9994,-15,0.5);
QgBF=getQg(GBF,0.9994,-15,0.5);
QaAF=getQa(AAF,0.9879,-22,0.8);
QaBF=getQa(ABF,0.9879,-22,0.8);
QAF=getQ(QgAF,QaAF);
QBF=getQ(QgBF,QaBF);
a=sum(sum(QAF.*gA+QBF.*gB));
b=sum(sum(gA+gB));
result=a/(b+eps);
%%
function [g,a]=get_g_a(im)

s1=[1 2 1; 0 0 0; -1 -2 -1];
s2=[-1 0 1;-2 0 2;-1 0 1];
sx=conv2(im,s1,'same');
sy=conv2(im,s2,'same');
g=sqrt(sx.^2+sy.^2);
a=atan(sy./(sx+eps));

end
%%
function Q=getQ(Qg,Qa)
Q=Qg.*Qa;
end
%%
function Qa=getQa(A,Ta,Ka,Oa)
Qa=Ta./(1+exp(Ka.*(A-Oa)));
end
%%
function Qg=getQg(G,Tg,Kg,Og)
Qg=Tg./(1+exp(Kg.*(G-Og)));
end
%%
function A=get2A(aIm1,aIm2)
A=1-abs(((aIm1-aIm2).*2)./pi);
end
%%
function G=get2G(gIm1,gIm2)
G=min(gIm1,gIm2)./(max(gIm1,gIm2)+eps);
end
%%
end