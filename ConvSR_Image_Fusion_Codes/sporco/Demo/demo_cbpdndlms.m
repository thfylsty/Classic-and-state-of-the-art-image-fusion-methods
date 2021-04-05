% Script demonstrating usage of the cbpdndlms function.
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2017-04-29
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'Copyright' and 'License' files
% distributed with the library.


% Training images
S0 = zeros(512, 512, 2, 'single');
S0(:,:,1) = single(stdimage('lena.grey')) / 255;
S0(:,:,2) = single(stdimage('barbara.grey')) / 255;


%Reduce images size to speed up demo script
tmp = zeros(128, 128, 2, 'single');
for k = 1:size(S0,3),
  tmp(:,:,k) = imresize(S0(:,:,k), 0.25);
end
S0 = tmp;


% Filter input images and compute highpass images
npd = 16;
fltlmbd = 5;
[Sl, Sh] = lowpass(S0, fltlmbd, npd);


% Construct weight matrix and padded test image set
Shp = padarray(Sh, [7 7], 'post');
t = 0.5;
W = randn(size(Sh));
W(abs(W) > t) = 1;
W(abs(W) < t) = 0;
W = padarray(W, [7 7], 'post');
ShW = W .* Shp;


% Construct initial dictionary
D0 = zeros(8,8,32, 'single');
D0(3:6,3:6,:) = single(randn(4,4,32));


% Set up cbpdndl parameters
lambda = 0.05;
opt = [];
opt.Verbose = 1;
opt.MaxMainIter = 500;
opt.rho = 50*lambda + 0.5;
opt.sigma = size(Sh,3);
opt.AutoRho = 1;
opt.AutoRhoPeriod = 10;
opt.AutoSigma = 1;
opt.AutoSigmaPeriod = 10;
opt.XRelaxParam = 1.8;
opt.DRelaxParam = 1.8;


% Do standard dictionary learning and reconstruct
[D1, X1, optinf1] = cbpdndl(D0, ShW, lambda, opt);
DX1 = ifft2(bsxfun(@times, fft2(D1, size(X1,1), size(X1,2)), fft2(X1)), ...
           'symmetric');
Sr1 = squeeze(sum(DX1,3)) + padarray(Sl, [7 7], 'post');

% Do dictionary learning with additive mask simulation and reconstruct
opt.W = W;
[D2, X2, optinf2] = cbpdndlms(D0, ShW, lambda, opt);
DX2 = ifft2(bsxfun(@times, fft2(D2, size(X2,1), size(X2,2)), fft2(X2)), ...
           'symmetric');
Sr2 = squeeze(sum(DX2,3)) + padarray(Sl, [7 7], 'post');


% Display dictionaries
figure;
subplot(1,2,1);
imdisp(tiledict(D1));
title('Standard DL');
subplot(1,2,2);
imdisp(tiledict(D2));
title('DL with additive mask simulation');


% Display reconstructions
figure;
subplot(2,2,1);
imdisp(Sr1(:,:,1));
title('Standard DL');
subplot(2,2,2);
imdisp(Sr2(:,:,1));
title('DL with additive mask simulation');
subplot(2,2,3);
imdisp(Sr1(:,:,2));
title('Standard DL');
subplot(2,2,4);
imdisp(Sr2(:,:,2));
title('DL with additive mask simulation');


% Plot functional value evolution
figure;
subplot(1,2,1);
semilogx(optinf1.itstat(:,2), 'LineWidth', 2);
ylim([15, 45]);
xlabel('Iterations');
ylabel('Functional value');
title('Standard DL');
subplot(1,2,2);
semilogx(optinf2.itstat(:,2), 'LineWidth', 2);
ylim([15, 45]);
xlabel('Iterations');
ylabel('Functional value');
title('DL with additive mask simulation');
