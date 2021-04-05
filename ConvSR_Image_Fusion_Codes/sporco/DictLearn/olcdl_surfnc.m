function [D, optinf] = olcdl_surfnc(D0, S, lambda, opt)

% olcdl_surfnc -- Online Convolutional Dictionary Learning
%                 (surrogate function approach)
%
%         argmin_{x_m,d_m} (1/2) \sum_k ||\sum_m d_m * x_k,m - s_k||_2^2 +
%                           lambda \sum_k \sum_m ||x_k,m||_1
%
%         The solution is computed using the surrogate function
%         approach (see liu-2017-online and liu-2017-online2)
%
% Usage:
%       [D, optinf] = olcdl_surfnc(D0, S, lambda, opt);
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
%   tol              The stopping tolerance for the first step
%   SampleN          The size of split training images
%   p                Forgetting exponent
%   DictFilterSizes  Array of size 2 x M where each column specifies the
%                    filter size (rows x columns) of the corresponding
%                    dictionary filter
%   ZeroMean         Force learned dictionary entries to be zero-mean
%
%
% Author: Jialin Liu <danny19921123@gmail.com>
%         Brendt Wohlberg <brendt@lanl.gov>
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.

if nargin < 4,
  opt = [];
end
opt = defaultopts(opt);

SampleN = opt.SampleN;
p = opt.p;
tol = opt.tol;

% Set up status display for verbose operation
nsep = 61;
if opt.Verbose && opt.MaxMainIter > 0,
  disp(char('-' * ones(1,nsep)));
end

Nimg = size(S,3);

if size(S,3) > 1,
  xsz = [SampleN SampleN size(D0,3) size(S,3)];
  S = reshape(S, [size(S,1) size(S,2) 1 size(S,3)]);
else
  xsz = [SampleN SampleN size(D0,3) 1];
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
optinf = struct('opt', opt);

% Initialise main working variables
D = Pnrm(D0);
G = Pzp(D);
Gprv = G;
approx_G = G;

% paramters for the sparse coding steps
lambda_sc = lambda;
opt_sc = [];
opt_sc.Verbose = 0;
opt_sc.MaxMainIter = 500;
opt_sc.AutoRho = 1;
opt_sc.AutoRhoPeriod = 1;
opt_sc.RelaxParam = 1.8;
opt_sc.RelStopTol = 1e-3;
opt_sc.rho = 10;


% A^t and b^t for on-line update
At = zeros(xsz(1), xsz(2), size(D0,3), size(D0,3));
bt = zeros(xsz(1), xsz(2), size(D0,3));

% variables used in inner loop
k = 1;
alpha_sum = 0;
inner_k = 0;

etascalar = SampleN^2 / ((2*dsz(1)*2*dsz(2))*2);  %step size for FISTA

% Use random shuffle order to read images.
epochs = (opt.MaxMainIter) / Nimg;
indices = [];
for ee = 1:(epochs+1)
  indices = [indices, randperm(Nimg)];
end

%% Main loop
while k <= opt.MaxMainIter,

  index = indices(k);

  Nx = floor(size(S,1)/SampleN);
  Ny = floor(size(S,2)/SampleN);
  order_x = randperm(Nx);
  order_y = randperm(Ny);

  for sp_index_x = 1:(Nx),
    for sp_index_y = 1:(Ny),

      % image splitting
      randleft = 1 + (order_x(sp_index_x)-1) * SampleN;
      randtop = 1 + (order_y(sp_index_y)-1) * SampleN;
      randright = randleft + SampleN - 1;
      randbottom = randtop + SampleN - 1;

      % Sparse coding
      SampleS = S(randleft:randright,randtop:randbottom,:,index);
      [X, ~] = cbpdn(PzpT(G), SampleS, lambda_sc, opt_sc);

      % check the coefficient maps
      if (nnz(isnan(X))>0),
        continue;
      end
      if (nnz(X)==0),
        continue;
      end

      Xf = fft2(X);
      XSf = sum(bsxfun(@times, conj(Xf), (fft2(SampleS))),4);

      xsize = [size(At,1) size(At,2) 1 size(At,3)];
      xsize2 = [size(At,1) size(At,2) size(At,3) 1];

      xh = reshape(conj(Xf), xsize2);
      x = reshape(Xf, xsize);

      % update A and b
      inner_k = inner_k + 1;
      alpha = (1 - 1/(inner_k))^p;
      alpha_sum = alpha_sum * alpha + 1;
      At = alpha * At + bsxfun(@times, xh, x);
      bt = alpha * bt + XSf;

      % step size for FISTA
      eta = etascalar / ( norm(vec(At / alpha_sum)));

      t = 1;
      InnerLoop = 300;

      % frequency domain FISTA
      for inner_loop_d  = 1:InnerLoop,

        % G is the main iterate; approx_G is the auxillary iterate.
        approx_Gf = fft2(approx_G);
        gra = sum(bsxfun(@times, At / alpha_sum, ...
                         reshape(approx_Gf, xsize)), 4) - bt / alpha_sum;
        Gf = approx_Gf - eta * gra;
        G = Pcn(ifft2(Gf, 'symmetric'));

        fpr = norm(vec(approx_G - G)); % fixed point residual for FISTA

        % Nesterov acceleration
        t_next = ( 1 + sqrt(1+4*t^2) ) / 2;
        approx_G = G + (t-1)/t_next * (G - Gprv);

        Gprv = G;
        t = t_next;

        if fpr <= tol/(1+k), % stopping condition
          break;
        end
      end
    end
  end

  if opt.Verbose,
    fprintf('k: %4d  split_region: %4d  inner_loop: %3d  fpr: %.4e\n', ...
            k, inner_k, inner_loop_d, fpr);
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


function opt = defaultopts(opt)

  if ~isfield(opt,'Verbose'),
    opt.Verbose = 1;
  end
  if ~isfield(opt,'MaxMainIter'),
    opt.MaxMainIter = 80;
  end
  if ~isfield(opt,'tol'),
    opt.tol = 1e-2;
  end
  if ~isfield(opt,'SampleN'),
    opt.SampleN = 64;
  end
  if ~isfield(opt,'p'),
    opt.p = 10;
  end
  if ~isfield(opt,'DictFilterSizes'),
    opt.DictFilterSizes = [];
  end
  if ~isfield(opt,'ZeroMean'),
    opt.ZeroMean = 0;
  end

return
