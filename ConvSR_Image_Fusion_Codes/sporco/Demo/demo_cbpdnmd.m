% Script demonstrating usage of the cbpdnmd function.
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2016-05-10
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

% Highpass filter test image
npd = 16;
fltlmbd = 5;
[sl, sh] = lowpass(s, fltlmbd, npd);

% Pad input to cbpdnmd
shp = padarray(sh, [11 11], 'post');
W = ones(size(sh));
W = padarray(W, [11 11], 'post');


% Compute representation using cbpdnmd
lambda = 0.01;
opt = [];
opt.Verbose = 1;
opt.MaxMainIter = 500;
opt.rho = 100*lambda + 1;
opt.AutoRho = 1;
opt.AutoRhoPeriod = 10;
opt.StdResiduals = 0;
opt.RhoRsdlTarget = 0.1;
opt.RelStopTol = 5e-3;
opt.AuxVarObj = 0;
opt.RelaxParam = 1.8;
opt.W = W;
opt.HighMemSolve = 1;
[X, optinf] = cbpdnmd(D, shp, lambda, opt);

% Compute reconstruction
scnv = @(d,x) ifft2(sum(bsxfun(@times, fft2(d, size(x,1), size(x,2)), ...
                               fft2(x)),3), 'symmetric');
DX = scnv(D, X);
DXcrp = DX(1:(end-11), 1:(end-11));
s1 = sl + DXcrp;



figure;
subplot(1,3,1);
plot(optinf.itstat(:,2));
xlabel('Iterations');
ylabel('Functional value');
subplot(1,3,2);
semilogy(optinf.itstat(:,5));
xlabel('Iterations');
ylabel('Primal residual');
subplot(1,3,3);
semilogy(optinf.itstat(:,6));
xlabel('Iterations');
ylabel('Dual residual');


figure;
imagesc(sum(abs(X),3));
title('Sum of absolute value of coefficient maps');


figure;
subplot(1,3,1);
imdisp(s);
title('Original image');
subplot(1,3,2);
imdisp(s1);
title(sprintf('Reconstructed image (SNR: %.2fdB)', snr(s, s1)));
subplot(1,3,3);
imagesc(s1 - s);
axis image; axis off;
title('Difference');


figure;
imagesc(DX .* (1-W));
title('Reconstruction in masked-out region');
