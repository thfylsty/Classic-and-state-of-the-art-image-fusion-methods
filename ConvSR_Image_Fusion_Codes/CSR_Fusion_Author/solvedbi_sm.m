function x = solvedbi_sm(ah, rho, b, c)

% solvedbi_sm -- Solve a diagonal block linear system with a scaled identity 
%                term using the Sherman-Morrison equation
%
%         The solution is obtained by independently solving a set of linear 
%         systems of the form (see wohlberg-2016-efficient)
%
%                  (a a^H + rho I) x = b
%
%         In this equation inner products and matrix products are taken along
%         the 3rd dimension of the corresponding multi-dimensional arrays; the
%         solutions are independent over the 1st and 2nd (and 4th, if 
%         non-singleton) dimensions.
%   
% Usage:
%       x = solvedbi_sm(ah, rho, b, c);
%
% Input:
%       ah          Multi-dimensional array containing a^H
%       rho         Scalar rho
%       b           Multi-dimensional array containing b
%       c           Multi-dimensional array containing pre-computed quantities
%                   a^H / (a^H a + rho)
%
% Output:
%       x           Multi-dimensional array containing linear system solution
%
%   
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2014-12-18
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


a = conj(ah);

if nargin < 4 || isempty(c),
  c = bsxfun(@rdivide, ah, sum(ah.*a, 3) + rho);
end

cb = sum(bsxfun(@times, c, b), 3);
clear ah c;
cba = bsxfun(@times, cb, a);
clear a cb;
x = (b - cba) / rho;

return
