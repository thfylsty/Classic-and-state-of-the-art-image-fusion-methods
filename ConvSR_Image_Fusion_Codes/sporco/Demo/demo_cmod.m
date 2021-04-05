% Script demonstrating usage of the cmod function.
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2014-12-19
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


% Load dictionary
load([sporco_path '/Data/ConvDict.mat']);
dmap = containers.Map(ConvDict.Label, ConvDict.Dict);
D0 = reshape(dmap('8x8x64'), [64 64]);


% Set up bpdn parameters
lambda = 0.1;
opt = [];
opt.Verbose = 1;
opt.MaxMainIter = 200;
opt.RelStopTol = 1e-3;

% Compute sparse representation on current dictionary
[X, optinf] = bpdn(D0, S, lambda, opt);


% Set up cmod parameters
opt = [];
opt.Verbose = 1;
opt.MaxMainIter = 500;
opt.sigma = size(S,2)/500;
opt.AutoSigma = 1;
opt.AutoSigmaPeriod = 10;
opt.RelaxParam = 1.8;
opt.AuxVarObj = 1;

% Update dictionary for training set S
[D1, optinf] = cmod(X, S, opt);


% Display dictionaries
figure;
subplot(1,2,1);
imdisp(tiledict(D0, [8 8]));
title('D0');
subplot(1,2,2);
imdisp(tiledict(D1, [8 8]));
title('D1');

% Plot functional value and residuals
figure;
subplot(1,3,1);
plot(optinf.itstat(:,2));
xlabel('Iterations');
ylabel('Functional value');
subplot(1,3,2);
semilogy(optinf.itstat(:,4));
xlabel('Iterations');
ylabel('Primal residual');
subplot(1,3,3);
semilogy(optinf.itstat(:,5));
xlabel('Iterations');
ylabel('Dual residual');
