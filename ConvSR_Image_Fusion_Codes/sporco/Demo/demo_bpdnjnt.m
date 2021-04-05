% Script demonstrating usage of the bpdnjnt function.
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-03-05
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'Copyright' and 'License' files
% distributed with the library.


% Signal and dictionary size
N = 512;
M = 4*N;
% Number of non-zero coefficients in generator
L = 32;
% Construct random dictionary
D = randn(N, M);
% Noise level
sigma = 5.0;
% Construct ensemble of L-sparse coefficients and corresponding signals
K = 8;
X0 = zeros(M, K);
si = randperm(M);
si = si(1:L);
for l = 1:K,
  X0(si,l) = randn(L, 1);
end
S0 = D*X0;
S = S0 + sigma*randn(N, K);


% BPDN for simultaneous recovery of sparse coefficient matrix
lambda = 200;
opt = [];
opt.Verbose = 1;
opt.rho = 2000;
opt.RelStopTol = 1e-6;
[X1, optinf1] = bpdn(D, S, lambda, opt);


% BPDN with joint sparsity penalty for simultaneous recovery of
% sparse coefficient matrix
lambda = 70;
mu = 250;
opt = [];
opt.Verbose = 1;
opt.rho = 500;
opt.RelStopTol = 1e-6;
[X2, optinf2] = bpdnjnt(D, S, lambda, mu, opt);


disp(sprintf('Coefficient estimation errors:'));
disp(sprintf('  BPDN: %3.6f   BPDNJNT: %3.6f', norm(X1(:)-X0(:)),...
             norm(X2(:)-X0(:))));


figure;
subplot(1,3,1);
imagesc(abs(X0));
title('Reference');
subplot(1,3,2);
imagesc(abs(X1));
title('BPDN');
subplot(1,3,3);
imagesc(abs(X2));
title('BPDN with joint sparsity term');

