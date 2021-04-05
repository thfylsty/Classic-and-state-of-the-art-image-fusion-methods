function  [err_max,err_l1,err_l2,SNR,mean_Q,SAM,RMSE,ERGAS,D_dist] = metrics(X,Y,S)
%         [err_max,err_l1,err_l2,Q,SAM,QAVE,ERGAS] = metrics(Xmh_hat,Xmh)
%
%  fusion image metrics,
%
%  X reference HR image
%  Y fused estimeted HR image
%  S downsampling factor
%
%  Band based indexes: err_max,err_l1,err_l2,Q
%  Global  indexes: SAM,ERGAS
%
%  
%
[nr,nc,p] = size(X);

% ig th downsamplig factor is not input, set it to S=4;
if nargin == 2
    S=4;
end

% convert into matrices

X = reshape(X, nr*nc,p)';
Y = reshape(Y, nr*nc,p)';

% max l1 error
err_max = max(abs(Y-X),[],2);

err_l1 = mean(abs(Y-X),2);

err_l2 = sqrt(mean(abs(Y-X).^2,2));
% err_l2 = mean(sqrt(abs(Y-X).^2),2);
SNR=10*log10(mean2(X.^2)/mean2((Y-X).^2));

mX = mean(X,2);
varX = var(X,0,2);
mY = mean(Y,2);
varY = var(Y,0,2);

covXY =  mean(X.*Y,2) - mX.*mY;

Q = 4*covXY.*mX.*mY./(varX + varY)./(mX.^2 + mY.^2);
mean_Q=mean(Q);

SAM = mean(angBvec(X,Y));

ERGAS = 100/(S)*sqrt(mean((err_l2./mean(X,2)).^2));

% degreee of distortion
D_dist=mean2(abs(Y(:)-X(:)));

% compute RMSE
RMSE = sqrt(mean((X(:)-Y(:)).^2));