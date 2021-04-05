function r = rrs(ax, b)

% rrs -- Compute relative residual for solution of A*x = b
%
% Usage:
%       x = rrs(ax, b)
%
% Input:
%       ax         A*x part of equation
%       b          b part of equation
%
% Output:
%       x          Relative residual norm(b - A*x)/norm(b)
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2014-12-05
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.

r = norm(ax(:) - b(:))/norm(b(:));

return
