function [B,S,C] = pca(U, cntr)

% pca -- PCA for data set
%
% Usage:
%       [B,S,C] = pca(U)
%
% Input:
%       U           Data matrix; each column is a sample
%       cntr        Flag indicating whether to centre data (default 0)
%
% Output:
%       B           PCA basis (projection onto PCA basis is B')
%       S           Eigenvalues
%       C           Data centre vector
%
% The projection onto the PCA subspace is B', and the projection into
% the subspace embedded in the full space is B*B'.
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2014-05-29
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


if nargin < 2,
  cntr = 0;
end

if cntr,
  C = mean(U, 2);
  U = U - repmat(C, 1, size(U,2));
else
  C = [];
end

[B, S] = svd(U, 0);
S = diag(S).^2;

return
