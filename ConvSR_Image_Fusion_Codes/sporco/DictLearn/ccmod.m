function [D, optinf] = ccmod(X, S, dsz, opt)

% ccmod -- Convolutional Constrained Method of Optimal Directions (MOD)
%
%         argmin_{d_m} (1/2) \sum_k ||\sum_m x_k,m * d_m - s_k||_2^2
%                      such that ||d_m||_2 = 1
%
%         The solution of the Convolutional Constrained MOD problem
%         (see wohlberg-2016-efficient) is computed using the ADMM
%         approach (see boyd-2010-distributed).
%
% Usage:
%       [D, optinf] = ccmod(X, S, dsz, opt)
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
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-12-18
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


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
  S = reshape(S, [size(S,1) size(S,2) 1 size(S,3)]);
else
  xsz = [size(X) 1];
end

% Set dsz to correct form
if numel(dsz) == 3, dsz = dsz(1:2); end

% Mean removal and normalisation projections
Pzmn = @(x) bsxfun(@minus, x, mean(mean(x,1),2));
Pnrm = @(x) normalise(x);

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
Xf = fft2(X, size(S,1), size(S,2));
% Compute signal in DFT domain
Sf = fft2(S);
% S convolved with all coefficients in DFT domain
XSf = sum(bsxfun(@times, conj(Xf), Sf), 4);

% Set up algorithm parameters and initialise variables
sigma = opt.sigma;
if isempty(sigma), sigma = size(S,3); end;
Nd = prod(xsz(1:3));
cgt = opt.CGTol;
optinf = struct('itstat', [], 'opt', opt);
r = Inf;
s = Inf;
epri = 0;
edua = 0;

% Initialise main working variables
D = []; Df = [];
if isempty(opt.G0),
  G = zeros([xsz(1) xsz(2) xsz(3)], class(S));
else
  G = opt.G0;
end
Gprv = G;
if isempty(opt.H0),
  if isempty(opt.G0),
    H = zeros([xsz(1) xsz(2) xsz(3)], class(S));
  else
    H = G;
  end
else
  H = opt.H0;
end


% Main loop
k = 1;
while k <= opt.MaxMainIter && (r > epri | s > edua),

  % Solve subproblems and update dual variable
  if strcmp(opt.LinSolve, 'SM'),
    Df = solvemdbi_ism(Xf, sigma, XSf + sigma*fft2(G - H));
  else
    [Df, cgst] = solvemdbi_cg(Xf, sigma, XSf + sigma*fft2(G - H), ...
                              cgt, opt.MaxCGIter, Df(:));
  end
  D = ifft2(Df, 'symmetric');

  % See pg. 21 of boyd-2010-distributed
  if opt.RelaxParam == 1,
    Dr = D;
  else
    Dr = opt.RelaxParam*D + (1-opt.RelaxParam)*G;
  end

  G = Pcn(Dr + H);
  H = H + Dr - G;

  % Compute data fidelity term in Fourier domain (note normalisation)
  if opt.AuxVarObj,
    Gf = fft2(G); % This represents unnecessary computational cost
    Job = sum(vec(abs(sum(bsxfun(@times,Gf,Xf),3)-Sf).^2))/(2*xsz(1)*xsz(2));
    Jcn = 0;
  else
    Job = sum(vec(abs(sum(bsxfun(@times,Df,Xf),3)-Sf).^2))/(2*xsz(1)*xsz(2));
    Jcn = norm(vec(Pcn(D) - D));
  end

  nD = norm(D(:)); nG = norm(G(:)); nH = norm(H(:));
  if opt.StdResiduals,
    % See pp. 19-20 of boyd-2010-distributed
    r = norm(vec(D - G));
    s = norm(vec(sigma*(Gprv - G)));
    epri = sqrt(Nd)*opt.AbsStopTol+max(nD,nG)*opt.RelStopTol;
    edua = sqrt(Nd)*opt.AbsStopTol+sigma*nH*opt.RelStopTol;
  else
    % See wohlberg-2015-adaptive
    r = norm(vec(D - G))/max(nD,nG);
    s = norm(vec(Gprv - G))/nH;
    epri = sqrt(Nd)*opt.AbsStopTol/max(nD,nG)+opt.RelStopTol;
    edua = sqrt(Nd)*opt.AbsStopTol/(sigma*nH)+opt.RelStopTol;
  end

  if opt.CGTolAuto && (r/opt.CGTolFactor) < cgt,
    cgt = r/opt.CGTolFactor;
  end

  % Record and display iteration details
  tk = toc(tstart);
  optinf.itstat = [optinf.itstat; [k Job Jcn r s epri edua sigma tk]];
  if opt.Verbose,
    if opt.AutoSigma,
      disp(sprintf(sfms, k, Job, Jcn, r, s, sigma));
    else
      disp(sprintf(sfms, k, Job, Jcn, r, s));
    end
  end

  % See wohlberg-2015-adaptive and pp. 20-21 of boyd-2010-distributed
  if opt.AutoSigma,
    if k ~= 1 && mod(k, opt.AutoSigmaPeriod) == 0,
      if opt.AutoSigmaScaling,
        sigmlt = sqrt(r/s);
        if sigmlt < 1, sigmlt = 1/sigmlt; end
        if sigmlt > opt.SigmaScaling, sigmlt = opt.SigmaScaling; end
      else
        sigmlt = opt.SigmaScaling;
      end
      ssf = 1;
      if r > opt.SigmaRsdlRatio*s, ssf = sigmlt; end
      if s > opt.SigmaRsdlRatio*r, ssf = 1/sigmlt; end
      sigma = ssf*sigma;
      H = H/ssf;
    end
  end

  Gprv = G;
  k = k + 1;

end

% Record run time and working variables
optinf.runtime = toc(tstart);
optinf.D = D;
optinf.G = G;
optinf.H = H;
optinf.sigma = sigma;
optinf.cgt = cgt;
if exist('cgst'), optinf.cgst = cgst; end

D = PzpT(G);

if opt.Verbose && opt.MaxMainIter > 0,
  disp(char('-' * ones(1,nsep)));
end

return


function u = vec(v)

  u = v(:);

return


function u = normalise(v)

  nrm = sqrt(sum(sum(v.^2, 1), 2));
  nrm(nrm == 0) = 1;
  u = bsxfun(@rdivide, v, nrm);

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
    u = zeros(cs(1), cs(2), size(v,3));
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
