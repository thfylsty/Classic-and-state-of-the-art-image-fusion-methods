% Script demonstrating usage of the bpdndl function.
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-07-30
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'Copyright' and 'License' files
% distributed with the library.


% Training images
S0 = zeros(512, 512, 5);
S0(:,:,1) = single(stdimage('lena.grey')) / 255;
S0(:,:,2) = single(stdimage('barbara.grey')) / 255;
S0(:,:,3) = single(stdimage('kiel.grey')) / 255;
S0(:,:,4) = single(rgb2gray(stdimage('mandrill'))) / 255;
tmp = single(stdimage('man.grey')) / 255;
S0(:,:,5) = tmp(101:612, 101:612);


%Reduce images size to speed up demo script
tmp = zeros(256, 256, 5);
for k = 1:size(S0,3),
  tmp(:,:,k) = imresize(S0(:,:,k), 0.5);
end
S0 = tmp;


% Extract all 8x8 image blocks, reshape, and subtract block means
SB = imageblocks(S0, [8 8]);
SB = reshape(SB, size(SB,1)*size(SB,2), size(SB,3));
S = bsxfun(@minus, SB, mean(SB, 1));


% Construct initial dictionary
D0 = randn(size(S,1), 64);


% Set up bpdndl parameters
lambda = 0.2;
opt = [];
opt.Verbose = 1;
opt.MaxMainIter = 1000;
opt.rho = 50*lambda + 0.5;
opt.sigma = size(S,2)/200;
opt.AutoRho = 1;
opt.AutoRhoPeriod = 10;
opt.RhoRsdlRatio = 2;
opt.RhoScaling = 5;
opt.AutoRhoScaling = 1;
opt.AutoSigma = 1;
opt.AutoSigmaPeriod = 10;
opt.XRelaxParam = 1.8;
opt.DRelaxParam = 1.8;

% Do dictionary learning
[D, X, optinf] = bpdndl(D0, S, lambda, opt);


% Display learned dictionary
figure;
imdisp(tiledict(D, [8 8]));

% Plot functional value evolution
figure;
plot(optinf.itstat(:,2));
xlabel('Iterations');
ylabel('Functional value');
