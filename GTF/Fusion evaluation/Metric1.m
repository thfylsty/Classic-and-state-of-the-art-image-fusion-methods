function  Result = Metric1(I,V,X)
I=double(I);
V=double(V);
X=double(X);
grey_level=256;
Result.SD_I=std(I(:),1);
Result.SD_V=std(V(:),1);
Result.SD_X=std(X(:),1);
Result.EN_I=entropy_fusion(I,grey_level);
Result.EN_V=entropy_fusion(V,grey_level);
Result.EN_X=entropy_fusion(X,grey_level);
Result.SF_I=func_evaluate_spatial_frequency(I);
Result.SF_V=func_evaluate_spatial_frequency(V);
Result.SF_X=func_evaluate_spatial_frequency(X);
Result.Total=[Result.SD_I;Result.SD_V;Result.SD_X;Result.EN_I;Result.EN_V;Result.EN_X;Result.SF_I;Result.SF_V;Result.SF_X];






