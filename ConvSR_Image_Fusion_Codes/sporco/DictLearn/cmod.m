function [G, optinf] = cmod(X, S, opt)

% cmod -- Constrained Method of Optimal Directions (MOD)
%
%           argmin_D (1/2)||D X - S||_2^2 such that ||d_k||_2 = 1
%                                              where d_k are columns of D
%
%         The solution of the MOD problem (see engan-1999-method) is
%         computed using the ADMM approach (see boyd-2010-distributed).
%
% Usage:
%       [G, optinf] = cmod(X, S, opt)
%
% Input:
%       X           Coefficients
%       S           Input images
%       opt         Algorithm parameters structure
%
% Output:
%       G           Dictionary
%       optinf      Details of optimisation
%
%
% Options structure fields:
%   Verbose           Flag determining whether iteration status is displayed.
%                     Fields are iteration number, functional value,
%                     data fidelity term, and
%                     primal and dual residuals (see Sec. 3.3 of
%                     boyd-2010-distributed). The value of sigma
%                     is also displayed if options request that they are
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
%   SigmaRsdlTarget   Residual ratio targeted by auto sigma update policy.
%   StdResiduals      Flag determining whether standard residual definitions
%                     (see Sec 3.3 of boyd-2010-distributed) are used instead
%                     of normalised residuals (see wohlberg-2015-adaptive)
%   RelaxParam        Relaxation parameter (see Sec. 3.4.3 of
%                     boyd-2010-distributed)
%   ZeroMean          Force learned dictionary entries to be zero-mean
%   AuxVarObj         Flag determining whether objective function is computed
%                     using the auxiliary (split) variable
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-07-10
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


if nargin < 3,
  opt = [];
end
checkopt(opt, defaultopts([]));
opt = defaultopts(opt);

% Set up status display for verbose operation
hstr = 'Itn   Obj       Cnst      r         s      ';
sfms = '%4d %9.2e %9.2e %9.2e %9.2e';
nsep = 44;
if opt.AutoSigma,
  hstr = [hstr '   sigma'];
  sfms = [sfms ' %9.2e'];
  nsep = nsep + 10;
end
if opt.Verbose && opt.MaxMainIter > 0,
  disp(hstr);
  disp(char('-' * ones(1,nsep)));
end

% Mean removal and normalisation projections
Pzmn = @(x) bsxfun(@minus, x, mean(x,1));
Pnrm = @(x) normalise(x);

% Projection of dictionary filters onto constraint set
if opt.ZeroMean,
  Pcn = @(x) Pnrm(Pzmn(x));
else
  Pcn = @(x) Pnrm(x);
end

% Start timer
tstart = tic;

% Set up algorithm parameters and initialise variables
sigma = opt.sigma;
if isempty(sigma), sigma = size(S,2)/200; end;
Nc = size(S,1);
Nm = size(X,1);
Nd = Nc*Nm;
SX = S*X';
[luL, luU] = factorise(X, sigma);
optinf = struct('itstat', [], 'opt', opt);
r = Inf;
s = Inf;
epri = 0;
edua = 0;


% Initialise main working variables
D = [];
if isempty(opt.G0),
  G = zeros(Nc,Nm);
else
  G = opt.G0;
end
Gprv = G;
if isempty(opt.H0),
  if isempty(opt.G0),
    H = zeros(Nc,Nm);
  else
    H = G;
  end
else
  H = opt.H0;
end


% Main loop
k = 1;
while k <= opt.MaxMainIter && (r > epri | s > edua),

  D = linsolve(X, sigma, luL, luU, SX + sigma*(G - H));
  %rrs( D*(X*X' + sigma*eye(size(X,1))),  SX + sigma*(G - H))

  % See pg. 21 of boyd-2010-distributed
  if opt.RelaxParam == 1,
    Dr = D;
  else
    Dr = opt.RelaxParam*D + (1-opt.RelaxParam)*G;
  end

  G = Pcn(Dr + H);
  H = H + Dr - G;

  % Objective function and convergence measures
  if opt.AuxVarObj
    Job = sum(vec(abs(G*X - S).^2))/2;
    Jcn = 0;
  else
    Job = sum(vec(abs(D*X - S).^2))/2;
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
        sigmlt = sqrt(r/(s*opt.SigmaRsdlTarget));
        if sigmlt < 1, sigmlt = 1/sigmlt; end
        if sigmlt > opt.SigmaScaling, sigmlt = opt.SigmaScaling; end
      else
        sigmlt = opt.SigmaScaling;
      end
      ssf = 1;
      if r > opt.SigmaRsdlTarget*opt.SigmaRsdlRatio*s, ssf = sigmlt; end
      if s > (opt.SigmaRsdlRatio/opt.SigmaRsdlTarget)*r, ssf = 1/sigmlt; end
      sigma = ssf*sigma;
      H = H/ssf;
      if ssf ~= 1,
        [luL, luU] = factorise(X, sigma);
      end
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

if opt.Verbose && opt.MaxMainIter > 0,
  disp(char('-' * ones(1,nsep)));
end

return


function u = vec(v)

  u = v(:);

return


function u = normalise(v)

  vn = sqrt(sum(v.^2, 1));
  vn(vn == 0) = 1;
  u = bsxfun(@rdivide, v, vn);

return


function [L,U] = factorise(A, c)

  [N,M] = size(A);
  % If N < M it is cheaper to factorise A*A' + cI and then use the
  % matrix inversion lemma to compute the inverse of A'*A + cI
  if N >= M,
    [L,U] = lu(A'*A + c*eye(M,M));
  else
    [L,U] = lu(A*A' + c*eye(N,N));
  end

return


function x = linsolve(A, c, L, U, b)

  [N,M] = size(A);
  if N >= M,
    x = (b - (((b*A) / U) / L)*A')/c;
  else
    x = (b / U) / L;
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
  if ~isfield(opt,'SigmaRsdlTarget'),
    opt.SigmaRsdlTarget = 1;
  end
  if ~isfield(opt,'StdResiduals'),
    opt.StdResiduals = 0;
  end
  if ~isfield(opt,'RelaxParam'),
    opt.RelaxParam = 1;
  end
  if ~isfield(opt,'ZeroMean'),
    opt.ZeroMean = 0;
  end
  if ~isfield(opt,'AuxVarObj'),
    opt.AuxVarObj = 0;
  end


return
