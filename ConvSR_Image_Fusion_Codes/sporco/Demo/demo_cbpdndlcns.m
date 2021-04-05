% Script demonstrating usage of the cbpdndlcns function.
%
% Author: Brendt Wohlberg <brendt@lanl.gov>
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'Copyright' and 'License' files
% distributed with the library.


% Training images
S0 = zeros(512, 512, 5, 'single');
S0(:,:,1) = single(stdimage('lena.grey')) / 255;
S0(:,:,2) = single(stdimage('barbara.grey')) / 255;
S0(:,:,3) = single(stdimage('kiel.grey')) / 255;
S0(:,:,4) = single(rgb2gray(stdimage('mandrill'))) / 255;
tmp = single(stdimage('man.grey')) / 255;
S0(:,:,5) = tmp(101:612, 101:612);


%Reduce images size to speed up demo script
tmp = zeros(256, 256, 5, 'single');
for k = 1:size(S0,3),
  tmp(:,:,k) = imresize(S0(:,:,k), 0.5);
end
S0 = tmp;


% Filter input images and compute highpass images
npd = 16;
fltlmbd = 5;
[Sl, Sh] = lowpass(S0, fltlmbd, npd);

% Construct initial dictionary
D0 = zeros(8,8,32, 'single');
D0(3:6,3:6,:) = single(randn(4,4,32));


% Set up cbpdndl parameters
lambda = 0.2;
opt = [];
opt.Verbose = 1;
opt.MaxMainIter = 250;
opt.rho = 50*lambda + 0.5;
opt.sigma = size(Sh,3);
opt.AutoRho = 1;
opt.AutoRhoPeriod = 10;
opt.AutoSigma = 1;
opt.AutoSigmaPeriod = 10;
opt.XRelaxParam = 1.8;
opt.DRelaxParam = 1.8;

% Do dictionary learning
[D, X, optinf] = cbpdndlcns(D0, Sh, lambda, opt);


% Display learned dictionary
figure;
imdisp(tiledict(D));

% Plot functional value evolution
figure;
plot(optinf.itstat(:,2));
xlabel('Iterations');
ylabel('Functional value');
