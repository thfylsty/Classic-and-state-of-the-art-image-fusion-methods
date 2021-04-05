function [Y, optinf] = celnet_gpu(D, S, lambda, mu, opt)

% celnet_gpu -- Convolutional Elastic Net (GPU version)
%
%         argmin_{x_m} (1/2)||\sum_m d_m * x_m - s||_2^2 +
%                      lambda \sum_m ||x_m||_1 + (mu/2) \sum_m ||x_m||_2^2
%
%         The solution is computed using an ADMM approach (see
%         boyd-2010-distributed) with efficient solution of the main
%         linear systems (see wohlberg-2016-efficient).
%
% Usage:
%       [Y, optinf] = celnet_gpu(D, S, lambda, mu, opt)
%
% Input:
%       D           Dictionary filter set (3D array)
%       s           Input image
%       lambda      Regularization parameter (l1)
%       mu          Regularization parameter (l2)
%       opt         Algorithm parameters structure
%
% Output:
%       Y           Dictionary coefficient map set (3D array)
%       optinf      Details of optimisation
%
%
% Options structure fields:
%   Verbose          Flag determining whether iteration status is displayed.
%                    Fields are iteration number, functional value,
%                    data fidelity term, l1 regularisation term, l2
%                    regularisation term, and primal and dual residuals
%                    (see Sec. 3.3 of boyd-2010-distributed). The value of
%                    rho is also displayed if options request that it is
%                    automatically adjusted.
%   MaxMainIter      Maximum main iterations
%   AbsStopTol       Absolute convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)
%   RelStopTol       Relative convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)
%   L1Weight         Weighting array for coefficients in l1 norm of X.
%                    Array should have the same dimensions as X, but the
%                    first two dimensions may be of unit size, corresponding
%                    to a weighting that varies with filter index but is
%                    spatially constant.
%   L2Weight         Weighting array for l2 norm of X. Array should have
%                    dimensions corresponding to the non-spatial dimensions
%                    of X since spatial weighting is no possible (i.e.
%                    weighting varies only with filter and sample index).
%   Y0               Initial value for Y
%   U0               Initial value for U
%   rho              ADMM penalty parameter
%   AutoRho          Flag determining whether rho is automatically updated
%                    (see Sec. 3.4.1 of boyd-2010-distributed)
%   AutoRhoPeriod    Iteration period on which rho is updated
%   RhoRsdlRatio     Primal/dual residual ratio in rho update test
%   RhoScaling       Multiplier applied to rho when updated
%   AutoRhoScaling   Flag determining whether RhoScaling value is
%                    adaptively determined (see wohlberg-2015-adaptive). If
%                    enabled, RhoScaling specifies a maximum allowed
%                    multiplier instead of a fixed multiplier.
%   RhoRsdlTarget    Residual ratio targeted by auto rho update policy.
%   StdResiduals     Flag determining whether standard residual definitions
%                    (see Sec 3.3 of boyd-2010-distributed) are used instead
%                    of normalised residuals (see wohlberg-2015-adaptive)
%   RelaxParam       Relaxation parameter (see Sec. 3.4.3 of
%                    boyd-2010-distributed)
%   NoBndryCross     Flag indicating whether all solution coefficients
%                    corresponding to filters crossing the image boundary
%                    should be forced to zero.
%   AuxVarObj        Flag determining whether objective function is computed
%                    using the auxiliary (split) variable
%   HighMemSolve     Use more memory for a slightly faster solution
%
%
% Authors: Brendt Wohlberg <brendt@lanl.gov>
%          Ping-Keng Jao <jpk7656@gmail.com>
% Modified: 2015-12-18
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.

gS = gpuArray(S);
gD = gpuArray(D);
glambda = gpuArray(lambda);

if nargin < 5,
  opt = [];
end
if nargin < 4,
  gmu = gpuArray(0);
else
  gmu = gpuArray(mu);
end
checkopt(opt, defaultopts([]));
opt = defaultopts(opt);

% Set up status display for verbose operation
hstr = 'Itn   Fnc       DFid      l1        l2        r         s      ';
sfms = '%4d %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e';
nsep = 64;
if opt.AutoRho,
  hstr = [hstr '   rho  '];
  sfms = [sfms ' %9.2e'];
  nsep = nsep + 10;
end
if opt.Verbose && opt.MaxMainIter > 0,
  disp(hstr);
  disp(char('-' * ones(1,nsep)));
end

% Start timer
tstart = tic;

% Collapsing of trailing singleton dimensions greatly complicates
% handling of both SMV and MMV cases. The simplest approach would be
% if S could always be reshaped to 4d, with dimensions consisting of
% image rows, image cols, a single dimensional placeholder for number
% of filters, and number of measurements, but in the single
% measurement case the third dimension is collapsed so that the array
% is only 3d.
if size(S,3) > 1,
  xsz = [size(S,1) size(S,2) size(D,3) size(S,3)];
  % Insert singleton 3rd dimension (for number of filters) so that
  % 4th dimension is number of images in input s volume
  S = reshape(S, [size(S,1) size(S,2) 1 size(S,3)]);
else
  xsz = [size(S,1) size(S,2) size(D,3) 1];
end
% Compute filters in DFT domain
gDf = fft2(gD, size(S,1), size(S,2));
% Convolve-sum and its Hermitian transpose
gDop = @(x) sum(bsxfun(@times, gDf, x), 3);
gDHop = @(x) bsxfun(@times, conj(gDf), x);
% Compute signal in DFT domain
gSf = fft2(gS);
% S convolved with all filters in DFT domain
gDSf = gDHop(gSf);

% Set up l2 weight array
if isscalar(opt.L2Weight),
  gwl2 = gpuArray(opt.L2Weight);
else
  gwl2 = gpuArray(reshape(opt.L2Weight, [1 1 size(opt.L2Weight,1) ...
                      size(opt.L2Weight,2)]));
end

% Default lambda is 1/10 times the lambda value beyond which the
% solution is a zero vector
if nargin < 3 | isempty(glambda),
  gb = ifft2(gDHop(gSf), 'symmetric');
  glambda = 0.1*max(vec(abs(gb)));
end
% Set up algorithm parameters and initialise variables
grho = gpuArray(opt.rho);
if isempty(grho), grho = 50*glambda+1; end;
gmwr = gmu*gwl2 + grho;
if isempty(opt.RhoRsdlTarget),
  if opt.StdResiduals,
    opt.RhoRsdlTarget = 1;
  else
    opt.RhoRsdlTarget = 1 + (18.3).^(log10(glambda) + 1);
  end
end
if opt.HighMemSolve,
  gcn = bsxfun(@rdivide, gDf, gmwr);
  gcd = sum(gDf.*bsxfun(@rdivide, conj(gDf), gmu*gwl2 + grho), 3) + 1.0;
  gC = bsxfun(@rdivide, gcn, gcd);
  clear cn cd;
else
  C = [];
end
gNx = prod(gpuArray(xsz));
optinf = struct('itstat', [], 'opt', opt);
gr = gpuArray(Inf);
gs = gpuArray(Inf);
gepri = gpuArray(0);
gedua = gpuArray(0);

% Initialise main working variables
% X = [];
if isempty(opt.Y0),
  gY = gpuArray.zeros(xsz);
else
  gY = gpuArray(opt.Y0);
end
gYprv = gY;
if isempty(opt.U0),
  if isempty(opt.Y0),
    gU = gpuArray.zeros(xsz);
  else
    gU = (glambda/grho)*sign(gY);
  end
else
  gU = gpuArray(opt.U0);
end

% Main loop
k = 1;
while k <= opt.MaxMainIter & (gr > gepri | gs > gedua),

  % Solve X subproblem
  gXf = solvedbd_sm(gDf, gmwr, gDSf + grho*fft2(gY - gU), gC);
  gX = ifft2(gXf, 'symmetric');

  % See pg. 21 of boyd-2010-distributed
  if opt.RelaxParam == 1,
    gXr = gX;
  else
    gXr = opt.RelaxParam*gX + (1-opt.RelaxParam)*gY;
  end

  % Solve Y subproblem
  gY = shrink(gXr + gU, (glambda/grho)*opt.L1Weight);
  if opt.NoBndryCross,
    gY((end-size(gD,1)+2):end,:,:,:) = 0;
    gY(:,(end-size(gD,1)+2):end,:,:) = 0;
  end

  % Update dual variable
  gU = gU + gXr - gY;

  % Compute data fidelity term in Fourier domain (note normalisation)
  if opt.AuxVarObj,
    gYf = fft2(gY); % This represents unnecessary computational cost
    gJdf = sum(vec(abs(sum(bsxfun(@times,gDf,gYf),3)-gSf).^2)) / ...
           (2*xsz(1)*xsz(2));
    gJl1 = sum(abs(vec(bsxfun(@times, opt.L1Weight, gY))));
    gJl2 = sum(vec(gwl2.*sum(sum(gY.^2, 1),2)))/2;
  else
    gJdf = sum(vec(abs(sum(bsxfun(@times,gDf,gXf),3)-gSf).^2)) / ...
           (2*xsz(1)*xsz(2));
    gJl1 = sum(abs(vec(bsxfun(@times, opt.L1Weight, gX))));
    gJl2 = sum(vec(gwl2.*sum(sum(gX.^2, 1),2)))/2;
  end
  gJfn = gJdf + glambda*gJl1 + gmu*gJl2;

  gnX = norm(gX(:)); gnY = norm(gY(:)); gnU = norm(gU(:));
  if opt.StdResiduals,
    % See pp. 19-20 of boyd-2010-distributed
    gr = norm(vec(gX - gY));
    gs = norm(vec(grho*(gYprv - gY)));
    gepri = sqrt(gNx)*opt.AbsStopTol+max(gnX,gnY)*opt.RelStopTol;
    gedua = sqrt(gNx)*opt.AbsStopTol+grho*gnU*opt.RelStopTol;
  else
    % See wohlberg-2015-adaptive
    gr = norm(vec(gX - gY))/max(gnX,gnY);
    gs = norm(vec(gYprv - gY))/gnU;
    gepri = sqrt(gNx)*opt.AbsStopTol/max(gnX,gnY)+opt.RelStopTol;
    gedua = sqrt(gNx)*opt.AbsStopTol/(grho*gnU)+opt.RelStopTol;
  end

  % Record and display iteration details
  tk = toc(tstart);
  optinf.itstat = [optinf.itstat; [k gather(gJfn) gather(gJdf) gather(gJl1) ...
                   gather(gJl2) gather(gr) gather(gs) gather(gepri) ...
                      gather(gedua) gather(grho) tk]];
  if opt.Verbose,
    if opt.AutoRho,
      disp(sprintf(sfms, k, gather(gJfn), gather(gJdf), gather(gJl1), ...
                   gather(gJl2), gather(gr), gather(gs), gather(grho)));
    else
      disp(sprintf(sfms, k, gather(gJfn), gather(gJdf), gather(gJl1), ...
                   gather(gJl2), gather(gr), gather(gs)));
    end
  end


  % See wohlberg-2015-adaptive and pp. 20-21 of boyd-2010-distributed
  if opt.AutoRho,
    if k ~= 1 && mod(k, opt.AutoRhoPeriod) == 0,
      if opt.AutoRhoScaling,
        grhomlt = sqrt(gr/(gs*opt.RhoRsdlTarget));
        if grhomlt < 1, grhomlt = 1/grhomlt; end
        if grhomlt > opt.RhoScaling, grhomlt = opt.RhoScaling; end
      else
        rhomlt = opt.RhoScaling;
      end
      grsf = 1;
      if gr > opt.RhoRsdlTarget*opt.RhoRsdlRatio*gs, grsf = grhomlt; end
      if gs > (opt.RhoRsdlRatio/opt.RhoRsdlTarget)*gr, grsf = 1/grhomlt; end
      grho = grsf*grho;
      gU = gU/grsf;
      if opt.HighMemSolve && grsf ~= 1,
        gmwr = gmu*gwl2 + grho;
        gcn = bsxfun(@rdivide, gDf, gmwr);
        gcd = sum(gDf.*bsxfun(@rdivide, conj(gDf), gmwr), 3) + 1.0;
        gC = bsxfun(@rdivide, gcn, gcd);
        clear gcn gcd;
      end
    end
  end

  gYprv = gY;
  k = k + 1;

end

% Record run time and working variables
optinf.runtime = toc(tstart);
optinf.X = gather(gX);
optinf.Xf = gather(gXf);
optinf.Y = gather(gY);
optinf.U = gather(gU);
optinf.lambda = gather(glambda);
optinf.mu = gather(mu);
optinf.rho = gather(grho);
Y = gather(gY);
if opt.Verbose && opt.MaxMainIter > 0,
  disp(char('-' * ones(1,nsep)));
end

return


function u = vec(v)

  u = v(:);

return


function u = shrink(v, lambda)

  if isscalar(lambda),
    u = sign(v).*max(0, abs(v) - lambda);
  else
    u = sign(v).*max(0, bsxfun(@minus, abs(v), lambda));
  end

return


function opt = defaultopts(opt)

  if ~isfield(opt,'Verbose'),
    opt.Verbose = 0;
  end
  if ~isfield(opt,'MaxMainIter'),
    opt.MaxMainIter = 1000;
  end
  if ~isfield(opt,'AbsStopTol'),
    opt.AbsStopTol = 0;
  end
  if ~isfield(opt,'RelStopTol'),
    opt.RelStopTol = 1e-4;
  end
  if ~isfield(opt,'L1Weight'),
    opt.L1Weight = 1;
  end
  if ~isfield(opt,'L2Weight'),
    opt.L2Weight = 1;
  end
  if ~isfield(opt,'Y0'),
    opt.Y0 = [];
  end
  if ~isfield(opt,'U0'),
    opt.U0 = [];
  end
  if ~isfield(opt,'rho'),
    opt.rho = [];
  end
  if ~isfield(opt,'AutoRho'),
    opt.AutoRho = 1;
  end
  if ~isfield(opt,'AutoRhoPeriod'),
    opt.AutoRhoPeriod = 1;
  end
  if ~isfield(opt,'RhoRsdlRatio'),
    opt.RhoRsdlRatio = 1.2;
  end
  if ~isfield(opt,'RhoScaling'),
    opt.RhoScaling = 100;
  end
  if ~isfield(opt,'AutoRhoScaling'),
    opt.AutoRhoScaling = 1;
  end
  if ~isfield(opt,'RhoRsdlTarget'),
    opt.RhoRsdlTarget = [];
  end
  if ~isfield(opt,'StdResiduals'),
    opt.StdResiduals = 0;
  end
  if ~isfield(opt,'RelaxParam'),
    opt.RelaxParam = 1.8;
  end
  if ~isfield(opt,'NoBndryCross'),
    opt.NoBndryCross = 0;
  end
  if ~isfield(opt,'AuxVarObj'),
    opt.AuxVarObj = 0;
  end
  if ~isfield(opt,'HighMemSolve'),
    opt.HighMemSolve = 0;
  end

return
