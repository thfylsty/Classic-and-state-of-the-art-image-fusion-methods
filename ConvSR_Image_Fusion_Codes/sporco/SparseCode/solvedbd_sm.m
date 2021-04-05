function x = solvedbd_sm(ah, d, b, c)

% solvedbd_sm -- Solve a diagonal block linear system with a diagonal term
%                using the Sherman-Morrison equation
%
%         The solution is obtained by independently solving a set of linear
%         systems of the form (see wohlberg-2016-efficient)
%
%                  (a a^H + d) x = b
%
%         In this equation inner products and matrix products are taken along
%         the 3rd dimension of the corresponding multi-dimensional arrays; the
%         solutions are independent over the 1st and 2nd (and 4th, if
%         non-singleton) dimensions.
%
% Usage:
%       x = solvedbd_sm(ah, d, b, c);
%
% Input:
%       ah          Multi-dimensional array containing a^H
%       d           Multi-dimensional array containing d
%       b           Multi-dimensional array containing b
%       c           Multi-dimensional array containing pre-computed quantities
%                   a^H d^(-1)/ (a^H d^(-1) a + 1)
%
% Output:
%       x           Multi-dimensional array containing linear system solution
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2016-03-15
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


a = conj(ah);

if nargin < 4 || isempty(c),
  cn = bsxfun(@rdivide, ah, d);
  cd = sum(bsxfun(@times, ah, bsxfun(@rdivide, a, d)), 3) + 1.0;
  c = bsxfun(@rdivide, cn, cd);
  clear cn cd;
end

cb = sum(bsxfun(@times, c, b), 3);
clear ah c;
cba = bsxfun(@times, cb, a);
clear a cb;
x = bsxfun(@rdivide, b - cba, d);

return
