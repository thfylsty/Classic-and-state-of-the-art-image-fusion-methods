function [D, optinf] = ccmod_gpu(X, S, dsz, opt)

% ccmod_gpu -- Convolutional Constrained Method of Optimal Directions
%              (MOD) (GPU version)
%
%         argmin_{d_m} (1/2) \sum_k ||\sum_m x_k,m * d_m - s_k||_2^2
%                      such that ||d_m||_2 = 1
%
%         The solution of the Convolutional Constrained MOD problem
%         (see wohlberg-2016-efficient) is computed using the ADMM
%         approach (see boyd-2010-distributed).
%
% Usage:
%       [D, optinf] = ccmod_gpu(X, S, dsz, opt)
%
% Input:
%       X           Coefficient maps (3D array)
%       S           Input images
%       dsz         Dictionary size
%       opt         Algorithm parameters structure
%
% Output:
%       D           Dictionary filter set (3D array)
%       optinf      Details of optimisation
%
%
% Options structure fields:
%   Verbose           Flag determining whether iteration status is displayed.
%                     Fields are iteration number, functional value,
%                     data fidelity term, l1 regularisation term, and
%                     primal and dual residuals (see Sec. 3.3 of
%                     boyd-2010-distributed). The values of rho and sigma
%                     are also displayed if options request that they are
%                     automatically adjusted.
%   MaxMainIter       Maximum main iterations
%   AbsStopTol        Absolute convergence tolerance (see Sec. 3.3.1 of
%                     boyd-2010-distributed)
%   RelStopTol        Relative convergence tolerance (see Sec. 3.3.1 of
%                     boyd-2010-distributed)
%   G0                Initial value for G
%   H0                Initial value for H
%   sigma             ADMM penalty parameter
%   AutoSigma         Flag determining whether sigma is automatically updated
%                     (see Sec. 3.4.1 of boyd-2010-distributed)
%   AutoSigmaPeriod   Iteration period on which sigma is updated
%   SigmaRsdlRatio    Primal/dual residual ratio in sigma update test
%   SigmaScaling      Multiplier applied to sigma when updated
%   AutoSigmaScaling  Flag determining whether SigmaScaling value is
%                     adaptively determined (see wohlberg-2015-adaptive). If
%                     enabled, SigmaScaling specifies a maximum allowed
%                     multiplier instead of a fixed multiplier.
%   StdResiduals      Flag determining whether standard residual definitions
%                     (see Sec 3.3 of boyd-2010-distributed) are used instead
%                     of normalised residuals (see wohlberg-2015-adaptive)
%   RelaxParam        Relaxation parameter (see Sec. 3.4.3 of
%                     boyd-2010-distributed)
%   LinSolve          Linear solver for main problem: 'SM' or 'CG'
%   MaxCGIter         Maximum CG iterations when using CG solver
%   CGTol             CG tolerance when using CG solver
%   CGTolAuto         Flag determining use of automatic CG tolerance
%   CGTolFactor       Factor by which primal residual is divided to obtain CG
%                     tolerance, when automatic tolerance is active
%   ZeroMean          Force learned dictionary entries to be zero-mean
%   AuxVarObj         Flag determining whether objective function is computed
%                     using the auxiliary (split) variable
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
gX = gpuArray(X);

if nargin < 4,
  opt = [];
end
checkopt(opt, defaultopts([]));
opt = defaultopts(opt);

% Set up status display for verbose operation
hstr = 'Itn   Obj       Cnst      r         s      ';
sfms = '%4d %9.2e %9.2e %9.2e %9.2e';
nsep = 44;
if opt.AutoSigma,
  hstr = [hstr '   sigma  '];
  sfms = [sfms ' %9.2e'];
  nsep = nsep + 10;
end
if opt.Verbose && opt.MaxMainIter > 0,
  disp(hstr);
  disp(char('-' * ones(1,nsep)));
end

% Collapsing of trailing singleton dimensions greatly complicates
% handling of both SMV and MMV cases. The simplest approach would be
% if S could always be reshaped to 4d, with dimensions consisting of
% image rows, image cols, a single dimensional placeholder for number
% of filters, and number of measurements, but in the single
% measurement case the third dimension is collapsed so that the array
% is only 3d.
if size(S,3) > 1,
  xsz = size(X);
  % Insert singleton 3rd dimension (for number of filters) so that
  % 4th dimension is number of images in input s volume
  gS = reshape(gS, [size(S,1) size(S,2) 1 size(S,3)]);
else
  xsz = [size(X) 1];
end

% Set dsz to correct form
if numel(dsz) == 3, dsz = dsz(1:2); end

% Mean removal and normalisation projections
Pzmn = @(x) bsxfun(@minus, x, mean(mean(x,1),2));
Pnrm = @(x) bsxfun(@rdivide, x, sqrt(sum(sum(x.^2, 1), 2)));

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

% Compute coefficients in DFT domain
gXf = fft2(gX, size(gS,1), size(gS,2));
% Compute signal in DFT domain
gSf = fft2(gS);
% S convolved with all coefficients in DFT domain
gXSf = sum(bsxfun(@times, conj(gXf), gSf), 4);

% Set up algorithm parameters and initialise variables
gsigma = gpuArray(opt.sigma);
if isempty(gsigma), gsigma = size(gS,3); end;
gNd = gpuArray(prod(xsz(1:3)));
gcgt = gpuArray(opt.CGTol);
optinf = struct('itstat', [], 'opt', opt);
gr = gpuArray(Inf);
gs = gpuArray(Inf);
gepri = gpuArray(0);
gedua = gpuArray(0);

% Initialise main working variables
D = []; Df = [];
if isempty(opt.G0),
  gG = gpuArray.zeros([xsz(1) xsz(2) xsz(3)]);
else
  gG = gpuArray(opt.G0);
end
gGprv = gG;
if isempty(opt.H0),
  if isempty(opt.G0),
    gH = gpuArray.zeros([xsz(1) xsz(2) xsz(3)]);
  else
    gH = gG;
  end
else
  gH = gpuArray(opt.H0);
end


% Main loop
k = 1;
while k <= opt.MaxMainIter & (gr > gepri | gs > gedua),

  % Solve subproblems and update dual variable
  if strcmp(opt.LinSolve, 'SM'),
    gDf = solvemdbi_ism_gpu(gXf, gsigma, gXSf + gsigma*fft2(gG - gH));
  else
    [gDf, gcgst] = solvemdbi_cg(gXf, gsigma, gXSf + gsigma*fft2(gG - gH), ...
                              gcgt, opt.MaxCGIter, gDf(:));
  end
  gD = ifft2(gDf, 'symmetric');

  % See pg. 21 of boyd-2010-distributed
  if opt.RelaxParam == 1,
    gDr = gD;
  else
    gDr = opt.RelaxParam*gD + (1-opt.RelaxParam)*gG;
  end

  gG = Pcn(gDr + gH);
  gH = gH + gDr - gG;

  % Compute data fidelity term in Fourier domain (note normalisation)
  if opt.AuxVarObj,
    gGf = fft2(gG); % This represents unnecessary computational cost
    gJob = sum(vec(abs(sum(bsxfun(@times,gGf,gXf),3)-gSf).^2)) / ...
           (2*xsz(1)*xsz(2));
    gJcn = 0;
  else
    gJob = sum(vec(abs(sum(bsxfun(@times,gDf,gXf),3)-gSf).^2)) / ...
           (2*xsz(1)*xsz(2));
    gJcn = norm(vec(Pcn(gD) - gD));
  end

  gnD = norm(gD(:)); gnG = norm(gG(:)); gnH = norm(gH(:));
  if opt.StdResiduals,
    % See pp. 19-20 of boyd-2010-distributed
    gr = norm(vec(gD - gG));
    gs = norm(vec(gsigma*(gGprv - gG)));
    gepri = sqrt(gNd)*opt.AbsStopTol+max(gnD,gnG)*opt.RelStopTol;
    gedua = sqrt(gNd)*opt.AbsStopTol+gsigma*gnH*opt.RelStopTol;
  else
    % See wohlberg-2015-adaptive
    gr = norm(vec(gD - gG))/max(gnD,gnG);
    gs = norm(vec(gGprv - gG))/gnH;
    gepri = sqrt(gNd)*opt.AbsStopTol/max(gnD,gnG)+opt.RelStopTol;
    gedua = sqrt(gNd)*opt.AbsStopTol/(gsigma*gnH)+opt.RelStopTol;
  end

  if opt.CGTolAuto && (gr/opt.CGTolFactor) < gcgt,
    gcgt = gr/opt.CGTolFactor;
  end

  % Record and display iteration details
  tk = toc(tstart);
  optinf.itstat = [optinf.itstat; [k gather(gJob) gather(gJcn) gather(gr) ...
      gather(gs) gather(gepri) gather(gedua) gather(gsigma) tk]];
  if opt.Verbose,
    if opt.AutoSigma,
      disp(sprintf(sfms, k, gather(gJob), gather(gJcn), gather(gr), ...
                   gather(gs), gather(gsigma)));
    else
      disp(sprintf(sfms, k, gather(gJob), gather(gJcn), gather(gr), ...
                   gather(gs)));
    end
  end

  % See wohlberg-2015-adaptive and pp. 20-21 of boyd-2010-distributed
  if opt.AutoSigma,
    if k ~= 1 && mod(k, opt.AutoSigmaPeriod) == 0,
      if opt.AutoSigmaScaling,
        gsigmlt = sqrt(gr/gs);
        if gsigmlt < 1, gsigmlt = 1/gsigmlt; end
        if gsigmlt > opt.SigmaScaling, gsigmlt = gpuArray(opt.SigmaScaling); end
      else
        gsigmlt = gpuArray(opt.SigmaScaling);
      end
      gssf = 1;
      if gr > opt.SigmaRsdlRatio*gs, gssf = gsigmlt; end
      if gs > opt.SigmaRsdlRatio*gr, gssf = 1/gsigmlt; end
      gsigma = gssf*gsigma;
      gH = gH/gssf;
    end
  end

  gGprv = gG;
  k = k + 1;

end

% Record run time and working variables
optinf.runtime = toc(tstart);
optinf.D = gather(gD);
optinf.G = gather(gG);
optinf.H = gather(gH);
optinf.sigma = gather(gsigma);
optinf.cgt = gather(gcgt);
if exist('gcgst'), optinf.cgst = gather(gcgst); end

D = gather(PzpT(gG));

if opt.Verbose && opt.MaxMainIter > 0,
  disp(char('-' * ones(1,nsep)));
end

return


function u = vec(v)

  u = v(:);

return


function u = zpad(v, sz)

  u = gpuArray.zeros(sz(1), sz(2), size(v,3), size(v,4));
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
    u = gpuArray.zeros(cs(1), cs(2), size(v,3));
    for k = 1:size(v,3),
      u(1:sz(k,1), 1:sz(k,2), k) = v(1:sz(k,1), 1:sz(k,2), k);
    end
  end

return


function opt = defaultopts(opt)

  if ~isfield(opt,'Verbose'),
    opt.Verbose = 0;
  end
  if ~isfield(opt,'MaxMainIter'),
    opt.MaxMainIter = 200;
  end
  if ~isfield(opt,'AbsStopTol'),
    opt.AbsStopTol = 1e-6;
  end
  if ~isfield(opt,'RelStopTol'),
    opt.RelStopTol = 1e-4;
  end
  if ~isfield(opt,'G0'),
    opt.G0 = [];
  end
  if ~isfield(opt,'H0'),
    opt.H0 = [];
  end
  if ~isfield(opt,'sigma'),
    opt.sigma = [];
  end
  if ~isfield(opt,'AutoSigma'),
    opt.AutoSigma = 0;
  end
  if ~isfield(opt,'AutoSigmaPeriod'),
    opt.AutoSigmaPeriod = 10;
  end
  if ~isfield(opt,'SigmaRsdlRatio'),
    opt.SigmaRsdlRatio = 10;
  end
  if ~isfield(opt,'SigmaScaling'),
    opt.SigmaScaling = 2;
  end
  if ~isfield(opt,'AutoSigmaScaling'),
    opt.AutoSigmaScaling = 0;
  end
  if ~isfield(opt,'StdResiduals'),
    opt.StdResiduals = 0;
  end
  if ~isfield(opt,'RelaxParam'),
    opt.RelaxParam = 1;
  end
  if ~isfield(opt,'LinSolve'),
    opt.LinSolve = 'SM';
  end
  if ~isfield(opt,'MaxCGIter'),
    opt.MaxCGIter = 1000;
  end
  if ~isfield(opt,'CGTol'),
    opt.CGTol = 1e-3;
  end
  if ~isfield(opt,'CGTolAuto'),
    opt.CGTolAuto = 0;
  end
  if ~isfield(opt,'CGTolAutoFactor'),
    opt.CGTolFactor = 50;
  end
  if ~isfield(opt,'ZeroMean'),
    opt.ZeroMean = 0;
  end
  if ~isfield(opt,'AuxVarObj'),
    opt.AuxVarObj = 0;
  end

return
