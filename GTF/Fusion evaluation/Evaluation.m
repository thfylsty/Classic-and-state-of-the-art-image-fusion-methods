function Result=Evaluation(grey_matrixA,grey_matrixB,fusion_matrix,grey_level)
% Author:  Qu Xiao-Bo    <quxiaobo [at] xmu.edu.cn>    June 26, 2009
%          Postal address:
% Rom 509, Scientific Research Building # 2,Haiyun Campus, Xiamen University,Xiamen,Fujian, P. R. China, 361005
% Website: http://quxiaobo.go.8866.org

Result=zeros(1,2);
Result(1,1)=mutural_information(grey_matrixA,grey_matrixB,fusion_matrix,grey_level);
Result(1,2)=edge_association(grey_matrixA,grey_matrixB,fusion_matrix);
disp('|| MI || QAB/F ||')