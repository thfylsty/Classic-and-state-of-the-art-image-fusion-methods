% Script demonstrating usage of the bpdngrp function.
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-03-05
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'Copyright' and 'License' files
% distributed with the library.


% Signal size
N = 512;
% Number of groups
P = 4;
% Group index indicator
vec = @(x) x(:);
g = vec(repmat(1:P, N, 1));
% Dictionary size
M = P*N;
% Number of non-zero coefficients in generator
K = 32;
% Noise level
sigma = 1;

% Construct random dictionary and random sparse coefficients with
% group structure (i.e. only one dictionary block has non-zero
% coefficients)
D = randn(N, M);
x0 = zeros(M, 1);
gi = randint(P); % randi only available in recent versions of Matlab
gir = find(g == gi);
si = randperm(max(gir)-min(gir)+1) + min(gir) - 1;
si = si(1:K);
x0(si) = randn(K, 1);
% Construct reference and noisy signal
s0 = D*x0;
s = s0 + sigma*randn(N,1);


% BPDN for recovery of sparse representation
lambda = 35;
opt = [];
opt.Verbose = 1;
opt.rho = 2800;
opt.RelStopTol = 1e-6;
[x1, optinf1] = bpdn(D, s, lambda, opt);


% BPDN for recovery of sparse representation with group sparsity penalty
lambda = 30;
mu = 70;
opt = [];
opt.Verbose = 1;
opt.rho = 2800;
opt.RelStopTol = 1e-6;
[x2, optinf2] = bpdngrp(D, s, lambda, mu, g, opt);


disp(sprintf('Coefficient estimation errors:'));
disp(sprintf('  BPDN: %3.3f   BPDNGRP: %3.3f', norm(x1(:)-x0(:)),...
             norm(x2(:)-x0(:))));

figure;
plot(x0,'r');
hold on;
plot(x1,'b');
hold off;
legend('Reference', 'Recovered', 'Location', 'SouthEast');
title('Dictionary Coefficients: BPDN');


figure;
plot(x0,'r');
hold on;
plot(x2,'b');
hold off;
legend('Reference', 'Recovered', 'Location', 'SouthEast');
title('Dictionary Coefficients: BPDN with group sparsity');

