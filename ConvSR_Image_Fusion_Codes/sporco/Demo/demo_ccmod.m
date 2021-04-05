% Script demonstrating usage of the ccmod function.
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-04-09
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


% Filter input images and compute highpass images
npd = 16;
fltlmbd = 5;
[Sl, Sh] = lowpass(S0, fltlmbd, npd);

% Load dictionary
load([sporco_path '/Data/ConvDict.mat']);
dmap = containers.Map(ConvDict.Label, ConvDict.Dict);
D0 = dmap('12x12x36');


% Set up cbpdn parameters
lambda = 0.1;
opt = [];
opt.Verbose = 1;
opt.MaxMainIter = 200;
opt.AutoRho = 1;
opt.AutoRhoPeriod = 1;
opt.RelaxParam = 1.8;

% Compute sparse representation on current dictionary
[X, optinf] = cbpdn(D0, Sh, lambda, opt);


% Set up ccmod parameters
opt = [];
opt.Verbose = 1;
opt.MaxMainIter = 500;
opt.sigma = size(Sh,3);
opt.AutoSigma = 1;
opt.AutoSigmaPeriod = 1;
opt.RelaxParam = 1.8;
opt.AuxVarObj = 1;

% Update dictionary for training set S
[D1, optinf1] = ccmod(X, Sh, size(D0), opt);


% Plot functional value and residuals
figure;
subplot(1,3,1);
plot(optinf1.itstat(:,2));
xlabel('Iterations');
ylabel('Functional value');
subplot(1,3,2);
semilogy(optinf1.itstat(:,4));
xlabel('Iterations');
ylabel('Primal residual');
subplot(1,3,3);
semilogy(optinf1.itstat(:,5));
xlabel('Iterations');
ylabel('Dual residual');


% Update dictionary with new filter sizes for training set S
dsz = [repmat([12 12]', [1 24]) repmat([8 8]', [1 12])];
[D2, optinf2] = ccmod(X, Sh, dsz, opt);


% Display dictionaries
figure;
subplot(1,3,1);
imdisp(tiledict(D0));
title('D0');
subplot(1,3,2);
imdisp(tiledict(D1));
title('D1');
subplot(1,3,3);
imdisp(tiledict(D2, dsz));
title('D2');


% Plot functional value evolution
figure;
plot(optinf1.itstat(:,2), 'r');
hold on;
plot(optinf2.itstat(:,2), 'b');
hold off;
xlabel('Iterations');
ylabel('Functional value');
legend('D2', 'D3');
