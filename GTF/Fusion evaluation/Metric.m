function  Result = Metric(I,V,X)
I=double(I);
V=double(V);
X=double(X);
grey_level=256;
Result.Total = [];
Result.EN=entropy_fusion(X,grey_level);
Result.Total = [Result.Total; Result.EN];

Result.MI=mutural_information(I,V,X,grey_level);
Result.Total = [Result.Total; Result.MI];

Result.Q_G=Qp_ABF(I,V,X);
Result.Total = [Result.Total; Result.Q_G];

%Result.FMI = fmi(I,V,X, 'none', 3);  % it is time consuming
%Result.Total = [Result.Total; Result.FMI];






