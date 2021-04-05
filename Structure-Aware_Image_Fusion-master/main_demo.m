clear;
close all
addpath(genpath(pwd));
for i=1:10
    I1 = imread(['../ir/',num2str(i),'.bmp']);
    I2 = imread(['../vis/',num2str(i),'.bmp']);
    J(:,:,1) = I1;
    J(:,:,2) = I2;
    F_echo_dtf = IJF(I1,I2);
    imwrite(F_echo_dtf,[num2str(i),'.bmp']);
    clear J;
end

Q_echo_dtf = Qp_ABF(I1, I2, F_echo_dtf)