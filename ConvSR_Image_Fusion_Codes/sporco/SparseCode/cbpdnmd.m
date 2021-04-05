function [X, optinf] = cbpdnmd(D, S, lambda, opt)

% cbpdnmd -- Convolutional Basis Pursuit DeNoising (Mask Decoupling)
%
%         argmin_{x_k} (1/2)||W \sum_k d_k * x_k - s||_2^2 +
%                           lambda \sum_k ||x_k||_1
%
%         The solution is computed using an ADMM approach (see
%         boyd-2010-distributed) with efficient solution of the main
%         linear systems (see wohlberg-2016-efficient and
%         wohlberg-2016-boundary).
%
% Usage:
%       [Y, optinf] = cbpdnmd(D, S, lambda, opt)
%
% Input:
%       D           Dictionary filter set (3D array)
%       S           Input image
%       lambda      Regularization parameter
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
%                    data fidelity term, l1 regularisation term, and
%                    primal and dual residuals (see Sec. 3.3 of
%                    boyd-2010-distributed). The value of rho is also
%                    displayed if options request that it is automatically
%                    adjusted.
%   MaxMainIter      Maximum main iterations
%   AbsStopTol       Absolute convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)
%   RelStopTol       Relative convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)
%   L1Weight         Weighting array for coefficients in l1 norm of X
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
%   W                Synthesis spatial weighting matrix
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2016-06-30
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
hstr = 'Itn   Fnc       DFid      l1        r         s      ';
sfms = '%4d %9.2e %9.2e %9.2e %9.2e %9.2e';
nsep = 54;
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
if size(S,3) > 1 && size(S,4) == 1,
  xsz = [size(S,1) size(S,2) size(D,3) size(S,3)];
  % Insert singleton 3rd dimension (for number of filters) so that
  % 4th dimension is number of images in input s volume
  S = reshape(S, [size(S,1) size(S,2) 1 size(S,3)]);
  if ~isscalar(opt.W) & ndims(opt.W) > 2,
    opt.W = reshape(opt.W, [size(opt.W,1) size(opt.W,2) 1 size(opt.W,3)]);
  end
else
  xsz = [size(S,1) size(S,2) size(D,3) size(S,4)];
end
K = size(S,4);
W = opt.W;
WS = bsxfun(@times, W, S);

% Compute filters in DFT domain
Df = fft2(D, size(S,1), size(S,2));
% Convolve-sum and its Hermitian transpose
Dop = @(x) sum(bsxfun(@times, Df, x), 3);
DHop = @(x) bsxfun(@times, conj(Df), x);

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
  C = bsxfun(@rdivide, Df, sum(Df.*conj(Df), 3) + 1.0);
else
  C = [];
end
Nx = prod(xsz);
Ny = prod(xsz + [0 0 1 0]);
optinf = struct('itstat', [], 'opt', opt);
r = Inf;
s = Inf;
epri = 0;
edua = 0;

% Initialise main working variables
X = [];
if isempty(opt.Y0),
  Y = zeros(xsz + [0 0 1 0]);
  Y(:,:,end,:) = S;
else
  Y = opt.Y0;
end
Yprv = Y;
if isempty(opt.U0),
  if isempty(opt.Y0),
    U = zeros(xsz + [0 0 1 0]);
  else
    U(:,:,1:(end-1),:) = (lambda/rho)*sign(Y(:,:,1:(end-1),:));
    U(:,:,end,:) = bsxfun(@times, W, (bsxfun(@times, W, Y(:,:,end,:)) - S))/rho;
  end
else
  U = opt.U0;
end

% Main loop
k = 1;
while k <= opt.MaxMainIter && (r > epri | s > edua),

  % Solve X subproblem
  YUf = fft2(Y - U);
  YU0f = YUf(:,:,1:(end-1),:);
  YU1f = YUf(:,:,end,:);
  Xf = solvedbi_sm(Df, 1.0, DHop(YU1f) + YU0f, C);
  X = ifft2(Xf, 'symmetric');
  DX = ifft2(sum(bsxfun(@times, Df, Xf), 3), 'symmetric');

  % See pg. 21 of boyd-2010-distributed
  AX = cat(3, X, DX);
  if opt.RelaxParam ~= 1.0,
    AX = opt.RelaxParam*AX + (1-opt.RelaxParam)*Y;
  end

  % Solve Y subproblem
  Y(:,:,1:(end-1),:) = shrink(AX(:,:,1:(end-1),:) + U(:,:,1:(end-1),:), ...
                              (lambda/rho)*opt.L1Weight);
  if opt.NoBndryCross,
    Y((end-size(D,1)+2):end,:,1:(end-1),:) = 0;
    Y(:,(end-size(D,2)+2):end,1:(end-1),:) = 0;
  end
  Y(:,:,end,:) = bsxfun(@rdivide,(WS + rho*(DX+U(:,:,end,:))),...
                        ((W.^2) + rho));

  % Update dual variable
  U = U + AX - Y;

  % Compute data fidelity term in Fourier domain (note normalisation)
  if opt.AuxVarObj,
    Jdf = sum(vec(abs(bsxfun(@times, W, Y(:,:,end,:)) - S).^2))/2;
    Jl1 = sum(abs(vec(bsxfun(@times, opt.L1Weight, Y(:,:,1:(end-1),:)))));
  else
    Jdf = sum(vec(abs(bsxfun(@times, W, DX) - S).^2))/2;
    Jl1 = sum(abs(vec(bsxfun(@times, opt.L1Weight, X))));
  end
  Jfn = Jdf + lambda*Jl1;

  % This is computationally expensive for diagnostic information
  U0 = U(:,:,1:(end-1),:);
  U1 = U(:,:,end,:);
  U1f = fft2(U1);
  ATU0 = U0;
  ATU1 = ifft2(DHop(U1f), 'symmetric');
  nX = norm(X(:)); nY = norm(Y(:));
  nU0 = norm(U0(:)); nU1 = norm(U1(:));
  if opt.StdResiduals,
    % See pp. 19-20 of boyd-2010-distributed
    r = norm(vec(AX - Y));
    s = rho*norm(vec(ATU0 + ATU1));
    epri = sqrt(Ny)*opt.AbsStopTol+max(nX,nY)*opt.RelStopTol;
    edua = sqrt(Nx)*opt.AbsStopTol+rho*max(nU0,nU1)*opt.RelStopTol;
  else
    % See wohlberg-2015-adaptive
    r = norm(vec(AX - Y))/max(nX,nY);
    s = norm(vec(ATU0 + ATU1))/max(nU0,nU1);
    epri = sqrt(Ny)*opt.AbsStopTol/max(nX,nY)+opt.RelStopTol;
    edua = sqrt(Nx)*opt.AbsStopTol/(rho*max(nU0,nU1))+opt.RelStopTol;
  end

  % Record and display iteration details
  tk = toc(tstart);
  optinf.itstat = [optinf.itstat; [k Jfn Jdf Jl1 r s epri edua rho tk]];
  if opt.Verbose,
    if opt.AutoRho,
      disp(sprintf(sfms, k, Jfn, Jdf, Jl1, r, s, rho));
    else
      disp(sprintf(sfms, k, Jfn, Jdf, Jl1, r, s));
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
        C = bsxfun(@rdivide, Df, sum(Df.*conj(Df), 3) + 1.0);
      end
    end
  end

  Yprv = Y;
  k = k + 1;

end

% Record run time and working variables
optinf.runtime = toc(tstart);
optinf.X = X;
optinf.Xf = Xf;
optinf.Y = Y;
optinf.U = U;
optinf.lambda = lambda;
optinf.rho = rho;

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
    opt.StdResiduals = 1;
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
  if ~isfield(opt,'W'),
    opt.W = 1.0;
  end

return
