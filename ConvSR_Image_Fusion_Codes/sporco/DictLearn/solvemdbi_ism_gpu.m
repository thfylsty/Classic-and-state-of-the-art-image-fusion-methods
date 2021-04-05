function x = solvemdbi_ism_gpu(ah, rho, b)

% solvemdbi_ism_gpu -- Solve a multiple diagonal block linear system with a
%                  scaled identity term by iterated application of the
%                  Sherman-Morrison equation (GPU version)
%
%         The solution is obtained by independently solving a set of linear
%         systems of the form (see wohlberg-2016-efficient)
%
%                  (rho I + a_0 a_0^H + a_1 a_1^H + ...) x = b
%
%         In this equation inner products and matrix products are taken along
%         the 3rd dimension of the corresponding multi-dimensional arrays; the
%         solutions are independent over the 1st and 2nd (and 4th, if
%         non-singleton) dimensions.
%
% Usage:
%       x = solvedbi_sm_gpu(ah, rho, b);
%
% Input:
%       ah          Multi-dimensional array containing a^H
%       rho         Scalar rho
%       b           Multi-dimensional array containing b
%
% Output:
%       x           Multi-dimensional array containing linear system solution
%
%
% Authors: Brendt Wohlberg <brendt@lanl.gov>
%          Ping-Keng Jao <jpk7656@gmail.com>
% Modified: 2015-07-23
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.

a = conj(ah);
K = size(ah,4);
gamma = gpuArray.zeros(size(a));
delta = gpuArray.zeros([size(a,1) size(a,2) 1 size(a,4)]);
alpha = gpuArray(a(:,:,:,1))/rho;
beta = gpuArray(b)/rho;
clear b;
for k = 1:K,

  gamma(:,:,:,k) = alpha;
  delta(:,:,1,k) =  1 + sum(bsxfun(@times, ah(:,:,:,k), gamma(:,:,:,k)), 3);

  c = sum(ah(:,:,:,k) .* beta, 3);
  d = bsxfun(@times, c, gamma(:,:,:,k));
  beta = beta - bsxfun(@rdivide, d, delta(:,:,1,k));

  if k <= K-1,
    alpha = a(:,:,:,k+1)/rho;
    for l = 1:k,
      c = sum(ah(:,:,:,l) .* alpha, 3);
      d = bsxfun(@times, c, gamma(:,:,:,l));
      alpha = alpha - bsxfun(@rdivide, d, delta(:,:,1,l));
    end
  end

end

x = beta;

return
