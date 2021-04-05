% Script demonstrating usage of the bpdn function.
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
% Noise level
sigma = 0.5;

% Construct random dictionary and random sparse coefficients
D = randn(N, M);
x0 = zeros(M, 1);
si = randperm(M);
si = si(1:L);
x0(si) = randn(L, 1);
% Construct reference and noisy signal
s0 = D*x0;
s = s0 + sigma*randn(N,1);

% BPDN for recovery of sparse representation
lambda = 20;
opt = [];
opt.Verbose = 1;
opt.rho = 100;
opt.RelStopTol = 1e-6;
[x1, optinf] = bpdn(D, s, lambda, opt);


figure('position', [100, 100, 1200, 300]);
subplot(1,4,1);
plot(optinf.itstat(:,2));
xlabel('Iterations');
ylabel('Functional value');
subplot(1,4,2);
semilogy(optinf.itstat(:,5));
xlabel('Iterations');
ylabel('Primal residual');
subplot(1,4,3);
semilogy(optinf.itstat(:,6));
xlabel('Iterations');
ylabel('Dual residual');
subplot(1,4,4);
semilogy(optinf.itstat(:,9));
xlabel('Iterations');
ylabel('Penalty parameter');


figure;
plot(x0,'r');
hold on;
plot(x1,'b');
hold off;
legend('Reference', 'Recovered', 'Location', 'SouthEast');
title('Dictionary Coefficients');



% Illustrate restart capability: do 40 iterations and compare with
% 20 iterations and then restart for an additional 20 iterations
opt2 = opt;
opt2.MaxMainIter = 40;
[x2, optinf2] = bpdn(D, s, lambda, opt2);

opt3 = opt2;
opt3.MaxMainIter = 20;
[x3, optinf3] = bpdn(D, s, lambda, opt3);

opt4 = opt3;
opt4.Y0 = optinf3.Y;
opt4.U0 = optinf3.U;
opt4.rho = optinf3.rho;
[x4, optinf4] = bpdn(D, s, lambda, opt4);


figure;
plot(optinf2.itstat(:,1), optinf2.itstat(:,2), 'r');
hold on;
plot(optinf3.itstat(:,1), optinf3.itstat(:,2), 'g');
plot(optinf4.itstat(:,1) + optinf3.itstat(end,1), optinf4.itstat(:,2), 'b');
xlabel('Iterations');
ylabel('Functional value');
legend('Uninterrupted', 'Stop at 20 iterations', 'Restart at 20 iterations');



% Construct ensemble of L-sparse coefficients and corresponding signals
K = 8;
X0 = zeros(M, K);
for l = 1:K,
  si = randperm(M);
  si = si(1:L);
  X0(si,l) = randn(L, 1);
end
S0 = D*X0;
S = S0 + sigma*randn(N, K);

% BPDN for simultaneous recovery of sparse coefficient matrix
lambda = 20;
opt = [];
opt.Verbose = 1;
opt.rho = 100;
opt.RelStopTol = 1e-6;
[X1, optinf] = bpdn(D, S, lambda, opt);


figure;
subplot(1,2,1);
imagesc(X0);
title('Reference');
subplot(1,2,2);
imagesc(X1);
title('Recovered');
