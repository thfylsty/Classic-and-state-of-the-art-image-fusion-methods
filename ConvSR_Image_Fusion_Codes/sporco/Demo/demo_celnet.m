% Script demonstrating usage of the celnet function.
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-04-09
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'Copyright' and 'License' files
% distributed with the library.


% Load dictionary
load([sporco_path '/Data/ConvDict.mat']);
dmap = containers.Map(ConvDict.Label, ConvDict.Dict);
D = dmap('12x12x36');


% Load test image
s = single(rgbtogrey(stdimage('lena')))/255;
if isempty(s),
  error('Data required for demo scripts has not been installed.');
end

scnv = @(d,x) ifft2(sum(bsxfun(@times, fft2(d, size(x,1), size(x,2)), ...
                               fft2(x)),3), 'symmetric');

% Highpass filter test image
npd = 16;
fltlmbd = 5;
[sl, sh] = lowpass(s, fltlmbd, npd);

% Compute representation using cbpdn
lambda = 0.01;
opt = [];
opt.Verbose = 1;
opt.MaxMainIter = 500;
opt.rho = 100*lambda + 1;
opt.RelStopTol = 1e-3;
opt.AuxVarObj = 0;
opt.HighMemSolve = 1;
[X1, optinf1] = cbpdn(D, sh, lambda, opt);

% Compute reconstruction
DX1 = scnv(D, X1);

% Compute representation using celnet
mu = 0.05;
[X2, optinf2] = celnet(D, sh, lambda, mu, opt);

% Compute reconstruction
DX2 = scnv(D, X2);



figure;
subplot(1,2,1);
imagesc(sum(abs(X1),3));
axis image; axis off;
title('Sum of |X1|');
subplot(1,2,2);
imagesc(sum(abs(X2),3));
axis image; axis off;
title('Sum of |X2|');



figure;
subplot(2,3,1);
imdisp(s);
title('Original image');
subplot(2,3,2);
imdisp(DX1 + sl);
title(sprintf('Reconstructed image from X1 (SNR: %.2fdB)', snr(s, DX1 + sl)));
subplot(2,3,3);
imagesc(DX2 + sl - s);
axis image; axis off;
title('Difference between original and X1');
subplot(2,3,4);
imdisp(DX1 - DX2);
title('Difference between X1 and X2');
subplot(2,3,5);
imdisp(DX2 + sl);
title(sprintf('Reconstructed image from X2 (SNR: %.2fdB)', snr(s, DX2 + sl)));
subplot(2,3,6);
imagesc(DX2 + sl - s);
axis image; axis off;
title('Difference between original and X2');
