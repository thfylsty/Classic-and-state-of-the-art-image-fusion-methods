function [Y, optinf] = bpdngrp(D, S, lambda, mu, g, opt)

% bpdngrp -- Basis Pursuit DeNoising with l2,1 group sparsity
%
%         argmin_x (1/2)||D*x - s||_2^2 + lambda*||x||_1 +
%                  mu * \sum_l ||G_l(x)||_2
%
%         The solution is computed using the ADMM approach (see
%         boyd-2010-distributed for details).
%
% Usage:
%       [Y, optinf] = bpdngrp(D, S, lambda, mu, g, opt)
%
% Input:
%       D           Dictionary matrix
%       S           Signal vector (or matrix)
%       lambda      Regularization parameter
%       mu          l2,1 regularization parameter
%       g           Vector containing index values indicating the
%                   group number for each dictionary element. The
%                   first group index is 1 (0 indicates no group).
%                   Numbers must be contiguous. Overlapping groups
%                   are not supported.
%       opt         Options/algorithm parameters structure (see below)
%
% Output:
%       Y           Dictionary coefficient vector (or matrix)
%       optinf      Details of optimisation
%
%
% Options structure fields:
%   Verbose           Flag determining whether iteration status is displayed.
%                     Fields are iteration number, functional value,
%                     data fidelity term, l1 regularisation term, l2,1
%                     regularisation term, and primal and dual residuals
%                     (see Sec. 3.3 of boyd-2010-distributed). The value of
%                     rho is also displayed if options request that it is
%                     automatically adjusted.
%   MaxMainIter       Maximum main iterations
%   AbsStopTol        Absolute convergence tolerance (see Sec. 3.3.1 of
%                     boyd-2010-distributed)
%   RelStopTol        Relative convergence tolerance (see Sec. 3.3.1 of
%                     boyd-2010-distributed)
%   Y0                Initial value for Y
%   U0                Initial value for U
%   rho               ADMM penalty parameter
%   AutoRho           Flag determining whether rho is automatically updated
%                     (see Sec. 3.4.1 of boyd-2010-distributed)
%   AutoRhoPeriod     Iteration period on which rho is updated
%   RhoRsdlRatio      Primal/dual residual ratio in rho update test
%   RhoScaling        Multiplier applied to rho when updated
%   AutoRhoScaling    Flag determining whether RhoScaling value is
%                     adaptively determined (see wohlberg-2015-adaptive). If
%                     enabled, RhoScaling specifies a maximum allowed
%                     multiplier instead of a fixed multiplier.
%   RhoRsdlTarget     Residual ratio targeted by auto rho update policy.
%   StdResiduals      Flag determining whether standard residual definitions
%                     (see Sec 3.3 of boyd-2010-distributed) are used instead
%                     of normalised residuals (see wohlberg-2015-adaptive)
%   RelaxParam        Relaxation parameter (see Sec. 3.4.3 of
%                     boyd-2010-distributed)
%   AuxVarObj         Flag determining whether objective function is computed
%                     using the auxiliary (split) variable
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-07-10
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


if nargin < 6,
  opt = [];
end
checkopt(opt, defaultopts([]));
opt = defaultopts(opt);

% Set up status display for verbose operation
hstr = 'Itn   Fnc       DFid      l1        l2,1      r         s      ';
sfms = '%4d %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e';
nsep = 64;
if opt.AutoRho,
  hstr = [hstr '   rho   '];
  sfms = [sfms ' %9.2e'];
  nsep = nsep + 10;
end
if opt.Verbose && opt.MaxMainIter > 0,
  disp(hstr);
  disp(char('-' * ones(1,nsep)));
end

% Start timer
tstart = tic;

% Set up algorithm parameters and initialise variables
rho = opt.rho;
if isempty(rho), rho = 50*lambda+1; end;
[Nr, Nc] = size(D);
Nm = size(S,2);
Nx = Nc*Nm;
Ng = max(g);
DTS = D'*S;
[luL, luU] = factorise(D, rho);
optinf = struct('itstat', [], 'opt', opt);
r = Inf;
s = Inf;
epri = 0;
edua = 0;

% Initialise main working variables
X = [];
if isempty(opt.Y0),
  Y = zeros(Nc,Nm);
else
  Y = opt.Y0;
end
Yprv = Y;
if isempty(opt.U0),
  if isempty(opt.Y0),
    U = zeros(Nc,Nm);
  else
    U = (lambda/rho)*sign(Y);
  end
else
  U = opt.U0;
end

% Main loop
k = 1;
while k <= opt.MaxMainIter && (r > epri | s > edua),

  % Solve X subproblem
  X = linsolve(D, rho, luL, luU, DTS + rho*(Y - U));

  % See pg. 21 of boyd-2010-distributed
  if opt.RelaxParam == 1,
    Xr = X;
  else
    Xr = opt.RelaxParam*X + (1-opt.RelaxParam)*Y;
  end

  % Solve Y subproblem
  Y = shrink_groups(Xr + U, g, Ng, lambda/rho, mu/rho);

  % Update dual variable
  U = U + Xr - Y;

  % Objective function and convergence measures
  if opt.AuxVarObj,
    Jdf = sum(vec(abs(D*Y - S).^2))/2;
    Jl1 = sum(abs(vec(Y)));
    Jl21 = norm21(Y, g, Ng);
  else
    Jdf = sum(vec(abs(D*X - S).^2))/2;
    Jl1 = sum(abs(vec(X)));
    Jl21 = norm21(X, g, Ng);
  end
  Jfn = Jdf + lambda*Jl1 + mu*Jl21;

  nX = norm(X(:)); nY = norm(Y(:)); nU = norm(U(:));
  if opt.StdResiduals,
    % See pp. 19-20 of boyd-2010-distributed
    r = norm(vec(X - Y));
    s = norm(vec(rho*(Yprv - Y)));
    epri = sqrt(Nx)*opt.AbsStopTol+max(nX,nY)*opt.RelStopTol;
    edua = sqrt(Nx)*opt.AbsStopTol+rho*nU*opt.RelStopTol;
  else
    % See wohlberg-2015-adaptive
    r = norm(vec(X - Y))/max(nX,nY);
    s = norm(vec(Yprv - Y))/nU;
    epri = sqrt(Nx)*opt.AbsStopTol/max(nX,nY)+opt.RelStopTol;
    edua = sqrt(Nx)*opt.AbsStopTol/(rho*nU)+opt.RelStopTol;
  end

  % Record and display iteration details
  tk = toc(tstart);
  optinf.itstat = [optinf.itstat;[k Jfn Jdf Jl1 Jl21 r s epri edua rho tk]];
  if opt.Verbose,
    if opt.AutoRho,
      disp(sprintf(sfms, k, Jfn, Jdf, Jl1, Jl21, r, s, rho));
    else
      disp(sprintf(sfms, k, Jfn, Jdf, Jl1, Jl21, r, s));
    end
  end

  % See wohlberg-2015-adaptive and pp. 20-21 of boyd-2010-distributed
  if opt.AutoRho,
    if k ~= 1 && mod(k, opt.AutoRhoPeriod) == 0,
      if opt.AutoRhoScaling,
        rhomlt = sqrt(r/(s*opt.RhoRsdlTarget));
        if rhomlt < 1, rhomlt = 1/rhomlt; end
        if rhomlt > opt.RhoScaling, rhomlt = opt.RhoScaling; end
      else
        rhomlt = opt.RhoScaling;
      end
      rsf = 1;
      if r > opt.RhoRsdlTarget*opt.RhoRsdlRatio*s, rsf = rhomlt; end
      if s > (opt.RhoRsdlRatio/opt.RhoRsdlTarget)*r, rsf = 1/rhomlt; end
      rho = rsf*rho;
      U = U/rsf;
      if rsf ~= 1,
        [luL, luU] = factorise(D, rho);
      end
    end
  end

  Yprv = Y;
  k = k + 1;

end

% Record run time and working variables
optinf.runtime = toc(tstart);
optinf.X = X;
optinf.Y = Y;
optinf.U = U;
optinf.lambda = lambda;
optinf.rho = rho;

% End status display for verbose operation
if opt.Verbose && opt.MaxMainIter > 0,
  disp(char('-' * ones(1,nsep)));
end

return


function u = vec(v)

  u = v(:);

return


function u = shrink1(v, a)

  u = sign(v).*max(0, abs(v) - a);

return


function U = shrink2_col_vec(V, a)

  % Additional complexity here allows simultaenous shrinkage of a
  % set of column vectors
  n2v = sqrt(sum(V.^2,1));
  n2v(n2v == 0) = 1;
  U = bsxfun(@times, V, max(0, n2v - a)./n2v);

return


function U = shrink21(V, a, b)

  % See wohlberg-2012-local and chartrand-2013-nonconvex
  U = shrink2_col_vec(shrink1(V,  a), b);

return


function U = shrink_groups(V, g, Ng, a, b)

  U = zeros(size(V));
  U(g==0,:) = shrink1(V(g==0,:), a);
  for l = 1:Ng,
    U(g==l,:) = shrink21(V(g==l,:), a, b);
  end

return


function x = norm21(u, g, Ng)

  x = 0;
  for l = 1:Ng,
    x = x + sqrt(sum(u(g==l,:).^2, 1));
  end
  x = sum(x); % In case u is a matrix (i.e. not a column vector)

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
    x = U \ (L \ b);
  else
    x = (b - A'*(U \ (L \ (A*b))))/c;
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
    opt.AutoRhoPeriod = 10;
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
    opt.RhoRsdlTarget = 1;
  end
  if ~isfield(opt,'StdResiduals'),
    opt.StdResiduals = 0;
  end
  if ~isfield(opt,'RelaxParam'),
    opt.RelaxParam = 1.8;
  end
  if ~isfield(opt,'AuxVarObj'),
    opt.AuxVarObj = 1;
  end

return
