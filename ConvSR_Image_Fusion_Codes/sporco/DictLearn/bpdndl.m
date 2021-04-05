function [G, Y, optinf] = bpdndl(D0, S, lambda, opt)

% bpdndl -- BPDN Dictionary Learning
%
%         argmin_{D,X} (1/2)||D X - S||_2^2 + lambda ||X||_1
%
%         Dictionary learning consists of interleaved alternation of
%         the ADMM (see boyd-2010-distributed) steps for the BPDN (see
%         chen-1998-atomic) and MOD (see engan-1999-method) problems.
%
% Usage:
%       [D, X, optinf] = bpdndl(D0, S, lambda, opt)
%
% Input:
%       D0          Initial dictionary
%       S           Input image
%       lambda      Regularization parameter
%       opt         Options/algorithm parameters structure (see below)
%
% Output:
%       D           Dictionary
%       X           Coefficients
%       optinf      Details of optimisation
%
%
% Options structure fields:
%   Verbose          Flag determining whether iteration status is displayed.
%                    Fields are iteration number, functional value,
%                    data fidelity term, l1 regularisation term, and
%                    primal and dual residuals (see Sec. 3.3 of
%                    boyd-2010-distributed). The values of rho and sigma
%                    are also displayed if options request that they are
%                    automatically adjusted.
%   MaxMainIter      Maximum main iterations
%   AbsStopTol       Absolute convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)
%   RelStopTol       Relative convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)
%   L1Weight         Weight matrix for L1 norm
%   Y0               Initial value for Y
%   U0               Initial value for U
%   G0               Initial value for G (overrides D0 if specified)
%   H0               Initial value for H
%   rho              Augmented Lagrangian penalty parameter
%   AutoRho          Flag determining whether rho is automatically updated
%                    (see Sec. 3.4.1 of boyd-2010-distributed)
%   AutoRhoPeriod    Iteration period on which rho is updated
%   RhoRsdlRatio     Primal/dual residual ratio in rho update test
%   RhoScaling       Multiplier applied to rho when updated
%   AutoRhoScaling   Flag determining whether RhoScaling value is
%                    adaptively determined (see wohlberg-2015-adaptive). If
%                    enabled, RhoScaling specifies a maximum allowed
%                    multiplier instead of a fixed multiplier
%   sigma            Augmented Lagrangian penalty parameter
%   AutoSigma        Flag determining whether sigma is automatically
%                    updated (see Sec. 3.4.1 of boyd-2010-distributed)
%   AutoSigmaPeriod  Iteration period on which sigma is updated
%   SigmaRsdlRatio   Primal/dual residual ratio in sigma update test
%   SigmaScaling     Multiplier applied to sigma when updated
%   AutoSigmaScaling Flag determining whether SigmaScaling value is
%                    adaptively determined (see wohlberg-2015-adaptive). If
%                    enabled, SigmaScaling specifies a maximum allowed
%                    multiplier instead of a fixed multiplier.
%   StdResiduals     Flag determining whether standard residual definitions
%                    (see Sec 3.3 of boyd-2010-distributed) are used instead
%                    of normalised residuals (see wohlberg-2015-adaptive)
%   XRelaxParam      Relaxation parameter (see Sec. 3.4.3 of
%                    boyd-2010-distributed) for X update
%   DRelaxParam      Relaxation parameter (see Sec. 3.4.3 of
%                    boyd-2010-distributed) for D update
%   NonNegCoef       Flag indicating whether solution should be forced to
%                    be non-negative
%   AuxVarObj        Flag determining whether objective function is computed
%                    using the auxiliary (split) variable
%   ZeroMean         Force learned dictionary entries to be zero-mean
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-07-30
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


if nargin < 4,
  opt = [];
end
checkopt(opt, defaultopts([]));
opt = defaultopts(opt);
Nx = size(D0,2)*size(S,2);
Nd = prod(size(D0));

% Set up status display for verbose operation
hstr = ['Itn   Fnc       DFid      l1        Cnstr     '...
        'r(X)      s(X)      r(D)      s(D) '];
sfms = '%4d %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e';
nsep = 84;
if opt.AutoRho,
  hstr = [hstr '     rho  '];
  sfms = [sfms ' %9.2e'];
  nsep = nsep + 10;
end
if opt.AutoSigma,
  hstr = [hstr '     sigma  '];
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

% Project initial dictionary onto constraint set
D = Pnrm(D0);

% Set up algorithm parameters and initialise variables
rho = opt.rho;
if isempty(rho), rho = 50*lambda+1; end;
sigma = opt.sigma;
if isempty(sigma), sigma = size(S,2)/200; end;
optinf = struct('itstat', [], 'opt', opt);
rx = Inf;
sx = Inf;
rd = Inf;
sd = Inf;
eprix = 0;
eduax = 0;
eprid = 0;
eduad = 0;

% Initialise main working variables
X = [];
if isempty(opt.Y0),
  Y = zeros(size(D,2), size(S,2));
else
  Y = opt.Y0;
end
Yprv = Y;
if isempty(opt.U0),
  if isempty(opt.Y0),
    U = zeros(size(D,2), size(S,2), class(S));
  else
    U = (lambda/rho)*sign(Y);
  end
else
  U = opt.U0;
end
if isempty(opt.G0),
  G = D;
else
  G = opt.G0;
end
Gprv = G;
if isempty(opt.H0),
  if isempty(opt.G0),
    H = zeros(size(G), class(S));
  else
    H = G;
  end
else
  H = opt.H0;
end
GS = G'*S;


% Main loop
k = 1;
while k <= opt.MaxMainIter && (rx > eprix|sx > eduax|rd > eprid|sd >eduad),

  % Solve X subproblem, using G as the dictionary for improved stability
  [luLx, luUx] = factorise(G, rho);
  X = linsolveX(G, rho, luLx, luUx, GS + rho*(Y - U));

  % See pg. 21 of boyd-2010-distributed
  if opt.XRelaxParam == 1,
    Xr = X;
  else
    Xr = opt.XRelaxParam*X + (1-opt.XRelaxParam)*Y;
  end

  % Solve Y subproblem
  Y = shrink(Xr + U, (lambda/rho)*opt.L1Weight);
  if opt.NonNegCoef,
    Y(Y < 0) = 0;
  end
  SY = S*Y';

  % Update dual variable corresponding to X, Y
  U = U + Xr - Y;

  % Compute primal and dual residuals and stopping thresholds for X update
  nX = norm(X(:)); nY = norm(Y(:)); nU = norm(U(:));
  if opt.StdResiduals,
    % See pp. 19-20 of boyd-2010-distributed
    rx = norm(vec(X - Y));
    sx = norm(vec(rho*(Yprv - Y)));
    eprix = sqrt(Nx)*opt.AbsStopTol+max(nX,nY)*opt.RelStopTol;
    eduax = sqrt(Nx)*opt.AbsStopTol+rho*nU*opt.RelStopTol;
  else
    % See wohlberg-2015-adaptive
    rx = norm(vec(X - Y))/max(nX,nY);
    sx = norm(vec(Yprv - Y))/nU;
    eprix = sqrt(Nx)*opt.AbsStopTol/max(nX,nY)+opt.RelStopTol;
    eduax = sqrt(Nx)*opt.AbsStopTol/(rho*nU)+opt.RelStopTol;
  end

  % Solve D subproblem, using Y as the coefficients for improved stability
  [luLd, luUd] = factorise(Y, sigma);
  D = linsolveD(Y, sigma, luLd, luUd, SY + sigma*(G - H));

  % See pg. 21 of boyd-2010-distributed
  if opt.DRelaxParam == 1,
    Dr = D;
  else
    Dr = opt.DRelaxParam*D + (1-opt.DRelaxParam)*G;
  end

  % Solve G subproblem
  G = Pcn(Dr + H);
  GS = G'*S;

  % Update dual variable corresponding to D, G
  H = H + Dr - G;

  % Compute primal and dual residuals and stopping thresholds for D update
  nD = norm(D(:)); nG = norm(G(:)); nH = norm(H(:));
  if opt.StdResiduals,
    % See pp. 19-20 of boyd-2010-distributed
    rd = norm(vec(D - G));
    sd = norm(vec(sigma*(Gprv - G)));
    eprid = sqrt(Nd)*opt.AbsStopTol+max(nD,nG)*opt.RelStopTol;
    eduad = sqrt(Nd)*opt.AbsStopTol+sigma*nH*opt.RelStopTol;
  else
    % See wohlberg-2015-adaptive
    rd = norm(vec(D - G))/max(nD,nG);
    sd = norm(vec(Gprv - G))/nH;
    eprid = sqrt(Nd)*opt.AbsStopTol/max(nD,nG)+opt.RelStopTol;
    eduad = sqrt(Nd)*opt.AbsStopTol/(sigma*nH)+opt.RelStopTol;
  end

  % Objective function
  if opt.AuxVarObj,
    Jdf = sum(vec(abs(G*Y - S).^2))/2;
    Jl1 = sum(abs(vec(opt.L1Weight .* Y)));
  else
    Jdf = sum(vec(abs(D*X - S).^2))/2;
    Jl1 = sum(abs(vec(opt.L1Weight .* X)));
  end
  Jfn = Jdf + lambda*Jl1;
  Jcn = norm(vec(Pcn(D) - D));


  % Record and display iteration details
  tk = toc(tstart);
  optinf.itstat = [optinf.itstat; ...
        [k Jfn Jdf Jl1 rx sx rd sd eprix eduax eprid eduad rho sigma tk]];
  if opt.Verbose,
    dvc = [k Jfn Jdf Jl1 Jcn rx sx rd sd];
    if opt.AutoRho,
      dvc = [dvc rho];
    end
    if opt.AutoSigma,
      dvc = [dvc sigma];
    end
    disp(sprintf(sfms, dvc));
  end

  % See wohlberg-2015-adaptive and pp. 20-21 of boyd-2010-distributed
  if opt.AutoRho,
    if k ~= 1 && mod(k, opt.AutoRhoPeriod) == 0,
      if opt.AutoRhoScaling,
        rhomlt = sqrt(rx/sx);
        if rhomlt < 1, rhomlt = 1/rhomlt; end
        if rhomlt > opt.RhoScaling, rhomlt = opt.RhoScaling; end
      else
        rhomlt = opt.RhoScaling;
      end
      rsf = 1;
      if rx > opt.RhoRsdlRatio*sx, rsf = rhomlt; end
      if sx > opt.RhoRsdlRatio*rx, rsf = 1/rhomlt; end
      rho = rsf*rho;
      U = U/rsf;
    end
  end
  if opt.AutoSigma,
    if k ~= 1 && mod(k, opt.AutoSigmaPeriod) == 0,
      if opt.AutoSigmaScaling,
        sigmlt = sqrt(rd/sd);
        if sigmlt < 1, sigmlt = 1/sigmlt; end
        if sigmlt > opt.SigmaScaling, sigmlt = opt.SigmaScaling; end
      else
        sigmlt = opt.SigmaScaling;
      end
      ssf = 1;
      if rd > opt.SigmaRsdlRatio*sd, ssf = sigmlt; end
      if sd > opt.SigmaRsdlRatio*rd, ssf = 1/sigmlt; end
      sigma = ssf*sigma;
      H = H/ssf;
    end
  end


  Yprv = Y;
  Gprv = G;
  k = k + 1;

end

% Record run time and working variables
optinf.runtime = toc(tstart);
optinf.X = X;
optinf.Y = Y;
optinf.U = U;
optinf.D = D;
optinf.G = G;
optinf.H = H;
optinf.lambda = lambda;
optinf.rho = rho;
optinf.sigma = sigma;

if opt.Verbose && opt.MaxMainIter > 0,
  disp(char('-' * ones(1,nsep)));
end

return


function u = vec(v)

  u = v(:);

return


function u = shrink(v, lambda)

  u = sign(v).*max(0, abs(v) - lambda);

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


function x = linsolveX(A, c, L, U, b)

  [N,M] = size(A);
  if N >= M,
    x = U \ (L \ b);
  else
    x = (b - A'*(U \ (L \ (A*b))))/c;
  end

return


function x = linsolveD(A, c, L, U, b)

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
    opt.MaxMainIter = 1000;
  end
  if ~isfield(opt,'AbsStopTol'),
    opt.AbsStopTol = 1e-6;
  end
  if ~isfield(opt,'RelStopTol'),
    opt.RelStopTol = 1e-4;
  end
  if ~isfield(opt,'L1Weight'),
    opt.L1Weight = 1;
  end
  if ~isfield(opt,'Y0'),
    opt.Y0 = [];
  end
  if ~isfield(opt,'U0'),
    opt.U0 = [];
  end
  if ~isfield(opt,'G0'),
    opt.G0 = [];
  end
  if ~isfield(opt,'H0'),
    opt.H0 = [];
  end
  if ~isfield(opt,'rho'),
    opt.rho = [];
  end
  if ~isfield(opt,'AutoRho'),
    opt.AutoRho = 0;
  end
  if ~isfield(opt,'AutoRhoPeriod'),
    opt.AutoRhoPeriod = 10;
  end
  if ~isfield(opt,'RhoRsdlRatio'),
    opt.RhoRsdlRatio = 10;
  end
  if ~isfield(opt,'RhoScaling'),
    opt.RhoScaling = 2;
  end
  if ~isfield(opt,'AutoRhoScaling'),
    opt.AutoRhoScaling = 0;
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
  if ~isfield(opt,'XRelaxParam'),
    opt.XRelaxParam = 1;
  end
  if ~isfield(opt,'DRelaxParam'),
    opt.DRelaxParam = 1;
  end
  if ~isfield(opt,'NonNegCoef'),
    opt.NonNegCoef = 0;
  end
  if ~isfield(opt,'AuxVarObj'),
    opt.AuxVarObj = 1;
  end
  if ~isfield(opt,'ZeroMean'),
    opt.ZeroMean = 0;
  end

return
