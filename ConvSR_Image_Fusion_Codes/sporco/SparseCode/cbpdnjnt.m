function [Y, optinf] = cbpdnjnt(D, S, lambda, mu, opt)

% cbpdnjnt -- Convolutional Basis Pursuit DeNoising with Joint Sparsity
%
%         argmin_{x_k} (1/2)||\sum_k d_k * x_k - s||_2^2 +
%                           lambda \sum_k ||x_k||_1 +
%                           mu ||{x_k}||_{2,1}
%
%         The solution is computed using an ADMM approach (see
%         boyd-2010-distributed) with efficient solution of the main
%         linear systems (see wohlberg-2016-efficient and
%         wohlberg-2016-convolutional).
%
%         Note: Multi-channel images are represented by stacking the
%         channels on the 3rd dimension of input array S. Since this
%         is also the dimension used for stacking multiple images,
%         multiple multi-channel images are are not handled properly
%         with respect to the l2,1 norm.
%
% Usage:
%       [Y, optinf] = cbpdnjnt(D, S, lambda, mu, opt)
%
% Input:
%       D           Dictionary filter set (3D array)
%       S           Input image
%       lambda      l1 regularization parameter
%       mu          l2,1 regularization parameter
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
%                    data fidelity term, l1 regularisation term, l2,1
%                    regularisation term, and primal and dual residuals
%                    (see Sec. 3.3 of boyd-2010-distributed). The value of
%                    rho is also displayed if options request that it is
%                    automatically adjusted.
%   MaxMainIter      Maximum main iterations
%   AbsStopTol       Absolute convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)
%   RelStopTol       Relative convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)
%   L1Weight         Weighting array for coefficients in l1 norm of X
%   L21Weight        Weighting array for coefficients in l2,1 norm of X
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
%   NonNegCoef       Flag indicating whether solution should be forced to
%                    be non-negative
%   NoBndryCross     Flag indicating whether all solution coefficients
%                    corresponding to filters crossing the image boundary
%                    should be forced to zero.
%   AuxVarObj        Flag determining whether objective function is computed
%                    using the auxiliary (split) variable
%   HighMemSolve     Use more memory for a slightly faster solution
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2016-07-01
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


if nargin < 5,
  opt = [];
end
checkopt(opt, defaultopts([]));
opt = defaultopts(opt);

% Set up status display for verbose operation
hstr = 'Itn   Fnc       DFid      l1        l2,1      r         s      ';
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
  hrm = [1 1 1 size(S,3)];
  % Insert singleton 3rd dimension (for number of filters) so that
  % 4th dimension is number of images in input s volume
  S = reshape(S, [size(S,1) size(S,2) 1 size(S,3)]);
else
  xsz = [size(S,1) size(S,2) size(D,3) 1];
  hrm = 1;
end
xrm = [1 1 size(D,3)];
% Compute filters in DFT domain
Df = fft2(D, size(S,1), size(S,2));
% Convolve-sum and its Hermitian transpose
Dop = @(x) sum(bsxfun(@times, Df, x), 3);
DHop = @(x) bsxfun(@times, conj(Df), x);
% Compute signal in DFT domain
Sf = fft2(S);
% S convolved with all filters in DFT domain
DSf = DHop(Sf);

% Set up algorithm parameters and initialise variables
rho = opt.rho;
if isempty(rho), rho = 50*lambda+1; end;
if isempty(opt.RhoRsdlTarget),
  if opt.StdResiduals,
    opt.RhoRsdlTarget = 1;
  else
    opt.RhoRsdlTarget = 1 + (18.3).^(log10(lambda) + 1);
  end
end
if opt.HighMemSolve,
  C = bsxfun(@rdivide, Df, sum(Df.*conj(Df), 3) + rho);
else
  C = [];
end
Nx = prod(xsz);
optinf = struct('itstat', [], 'opt', opt);
r = Inf;
s = Inf;
epri = 0;
edua = 0;

% Initialise main working variables
X = [];
if isempty(opt.Y0),
  Y = zeros(xsz, class(S));
else
  Y = opt.Y0;
end
Yprv = Y;
if isempty(opt.U0),
  if isempty(opt.Y0),
    U = zeros(xsz, class(S));
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
  Xf = solvedbi_sm(Df, rho, DSf + rho*fft2(Y - U), C);
  X = ifft2(Xf, 'symmetric');

  % See pg. 21 of boyd-2010-distributed
  if opt.RelaxParam == 1,
    Xr = X;
  else
    Xr = opt.RelaxParam*X + (1-opt.RelaxParam)*Y;
  end

  % Solve Y subproblem
  Y = shrink21(Xr + U, lambda/rho, mu/rho, opt.L1Weight, opt.L21Weight);
  if opt.NonNegCoef,
    Y(Y < 0) = 0;
  end
  if opt.NoBndryCross,
    Y((end-size(D,1)+2):end,:,:,:) = 0;
    Y(:,(end-size(D,2)+2):end,:,:) = 0;
  end

  % Update dual variable
  U = U + Xr - Y;

  % Compute data fidelity term in Fourier domain (note normalisation)
  if opt.AuxVarObj,
    Yf = fft2(Y); % This represents unnecessary computational cost
    Jdf = sum(vec(abs(sum(bsxfun(@times,Df,Yf),3)-Sf).^2))/(2*xsz(1)*xsz(2));
    Jl1 = sum(abs(vec(bsxfun(@times, opt.L1Weight, Y))));
    Jl21 = sum(sqrt(vec(sum(bsxfun(@times, opt.L21Weight, Y).^2, 4))));
  else
    Jdf = sum(vec(abs(sum(bsxfun(@times,Df,Xf),3)-Sf).^2))/(2*xsz(1)*xsz(2));
    Jl1 = sum(abs(vec(bsxfun(@times, opt.L1Weight, X))));
    Jl21 = sum(sqrt(vec(sum(bsxfun(@times, opt.L21Weight, X).^2, 4))));
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
  optinf.itstat = [optinf.itstat; [k Jfn Jdf Jl1 Jl21 r s epri edua rho tk]];
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
      if opt.HighMemSolve && rsf ~= 1,
        C = bsxfun(@rdivide, Df, sum(Df.*conj(Df), 3) + rho);
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


function u = shrink1_scalars(v, a)

  if isscalar(a),
    u = sign(v).*max(0, abs(v) - a);
  else
    u = sign(v).*max(0, bsxfun(@minus, abs(v), a));
  end

return


function U = shrink2_row_vectors(V, a)

  n2v = sqrt(sum(V.^2, 4));
  n2v(n2v == 0) = 1;
  if isscalar(a),
    U = bsxfun(@times, V, max(0, n2v - a)./n2v);
  else
    U = bsxfun(@times, V, max(0, bsxfun(@minus, n2v, a))./n2v);
  end

return


function U = shrink21(V, a, b, W1, W2)

  if nargin < 4 || isempty(W1),
    W1 = 1;
  end
  % See wohlberg-2012-local and chartrand-2013-nonconvex
  U = shrink2_row_vectors(shrink1_scalars(V,  W1 .* a), W2 .* b);

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
  if ~isfield(opt,'L21Weight'),
    opt.L21Weight = 1;
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
  if ~isfield(opt,'NonNegCoef'),
    opt.NonNegCoef = 0;
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
