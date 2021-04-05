function [D, optinf] = olcdl_sgd(D0, S, lambda, opt)

% olcdl_sgd -- Online Convolutional Dictionary Learning
%              (frequency domain SGD)
%
%         argmin_{x_m,d_m} (1/2) \sum_k ||\sum_m d_m * x_k,m - s_k||_2^2 +
%                           lambda \sum_k \sum_m ||x_k,m||_1
%
%         The solution is computed using frequency domain SGD
%         (see liu-2017-online2).
%
% Usage:
%       [D, optinf] = olcdl_sgd(D0, S, lambda, opt);
%
% Input:
%       D0          Initial dictionary
%       S           Input images
%       lambda      Regularization parameter
%       opt         Options/algorithm parameters structure (see below)
%
% Output:
%       D           Dictionary filter set (3D array)
%       optinf      Details of optimisation
%
%
% Options structure fields:
%   Verbose          Flag determining whether iteration status is displayed.
%                    Fields are iteration number, the difference of
%                    the current dictionary with the last one, and
%                    the image index.
%   MaxMainIter      Maximum main iterations
%   eta_a            The "a" in "eta = a / (b + t)" in liu-2017-online2.
%   eta_b            The "b" in "eta = a / (b + t)" in the liu-2017-online2.
%   DictFilterSizes  Array of size 2 x M where each column specifies the
%                    filter size (rows x columns) of the corresponding
%                    dictionary filter
%   ZeroMean         Force learned dictionary entries to be zero-mean
%
%
% Authors: Jialin Liu <danny19921123@gmail.com>
%          Brendt Wohlberg <brendt@lanl.gov>
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.

if nargin < 4,
  opt = [];
end
opt = defaultopts(opt);

eta_a = opt.eta_a;
eta_b = opt.eta_b;

% Set up status display for verbose operation
nsep = 35;
if opt.Verbose && opt.MaxMainIter > 0,
  disp(char('-' * ones(1,nsep)));
end

Nimg = size(S,3);

if size(S,3) > 1,
  xsz = [size(S,1)  size(S,2)  size(D0,3) size(S,3)];
  S = reshape(S, [size(S,1) size(S,2) 1 size(S,3)]);
else
  xsz = [size(S,1)  size(S,2)  size(D0,3) 1];
end

if isempty(opt.DictFilterSizes),
  dsz = [size(D0,1) size(D0,2)];
else
  dsz = opt.DictFilterSizes;
end

% Mean removal and normalisation projections
Pzmn = @(x) bsxfun(@minus, x, mean(mean(x,1),2));
Pnrm = @(x) bsxfun(@rdivide, x, max(sqrt(sum(sum(x.^2, 1), 2)),1));

% Projection of filter to full image size and its transpose
% (zero-pad and crop respectively)
Pzp = @(x) zpad(x, xsz(1:2));
PzpT = @(x) bcrop(x, dsz);

% Projection of dictionary filters onto constraint set
if opt.ZeroMean,
  Pcn = @(x) Pnrm(Pzp(Pzmn(PzpT(x))));
else
  Pcn = @(x) Pnrm(Pzp(PzpT(x)));
end

% Start timer
tstart = tic;

% output info
optinf = struct('itstat', [], 'opt', opt);

% Initialise main working variables
D = Pnrm(D0);
G = Pzp(D);
Gprv = G;
Gf = fft2(G);

% parameters for sparse coding step
lambda_sc = lambda;
opt_sc = [];
opt_sc.Verbose = 0;
opt_sc.MaxMainIter = 500;
opt_sc.AutoRho = 1;
opt_sc.AutoRhoPeriod = 1;
opt_sc.RelaxParam = 1.8;
opt_sc.RelStopTol = 1e-3;

% Use random shuffle order to read images.
epochs = (opt.MaxMainIter) / Nimg;
indices = [];
for ee = 1:epochs
  indices = [indices, randperm(Nimg)];
end

%% Main loop
k = 1;
while k <= opt.MaxMainIter,

  index = indices(k);

  % sparse coding
  SampleS = S(:,:,:,index);
  [X, ~] = cbpdn(PzpT(G), SampleS, lambda_sc, opt_sc);

  Xf = fft2(X);
  Sf = fft2(SampleS);

  % computing the learning rate
  eta = eta_a / (k + eta_b);

  gra = nabla_freq(Xf,Gf,Sf);  % gradient in frequency domain
  Gf = Gf - eta * gra;
  G = Pcn(ifft2(Gf, 'symmetric'));  % projection in spacial domain
  Gf = fft2(G);

  sd = norm(vec(Gprv - G));  % successive differences
  g = norm(gra(:));
  Gprv = G;

  % Record and display iteration details
  tk = toc(tstart);
  optinf.itstat = [optinf.itstat; [k sd g eta tk]];
  if opt.Verbose,
    fprintf('k: %4d  sd: %.4e  index: %3d\n', k, sd, index);
  end

  k = k + 1;

end

D = PzpT(G);

%% Record run time and working variables
optinf.runtime = toc(tstart);
optinf.G = G;
optinf.lambda = lambda;
optinf.lastIndex = index;
optinf.indices = indices;

if opt.Verbose && opt.MaxMainIter > 0,
  disp(char('-' * ones(1,nsep)));
end

return


function u = vec(v)

  u = v(:);

return


function u = zpad(v, sz)

  u = zeros(sz(1), sz(2), size(v,3), size(v,4), class(v));
  u(1:size(v,1), 1:size(v,2),:,:) = v;

return


function u = bcrop(v, sz)

  if numel(sz) <= 2,
    if numel(sz) == 1
      cs = [sz sz];
    else
      cs = sz;
    end
    u = v(1:cs(1), 1:cs(2), :);
  else
    if size(sz,1) < size(sz,2), sz = sz'; end
    cs = max(sz);
    u = zeros(cs(1), cs(2), size(v,3), class(v));
    for k = 1:size(v,3),
      u(1:sz(k,1), 1:sz(k,2), k) = v(1:sz(k,1), 1:sz(k,2), k);
    end
  end

return


function g = nabla_freq(Xf, Df, Sf)

  g = bsxfun( @times, conj(Xf), sum( bsxfun(@times,Xf,Df), 3) - Sf);

return


function opt = defaultopts(opt)

  if ~isfield(opt,'Verbose'),
    opt.Verbose = 1;
  end
  if ~isfield(opt,'MaxMainIter'),
    opt.MaxMainIter = 200;
  end
  if ~isfield(opt,'eta_a'),
    opt.eta_a = 10;
  end
  if ~isfield(opt,'eta_b'),
    opt.eta_b = 5;
  end
  if ~isfield(opt,'DictFilterSizes'),
    opt.DictFilterSizes = [];
  end
  if ~isfield(opt,'ZeroMean'),
    opt.ZeroMean = 0;
  end

return
