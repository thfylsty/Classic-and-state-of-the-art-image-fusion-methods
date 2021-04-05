function v = randint(ri, nr, nc)

% randint -- Construct an array of uniformly distributed random integers
%
% Usage:
%       v = randint(ri, nr, nc)
%
% Input:
%       ri          Two-vector specifying range of random integers. If a
%                   scalar is specified the range is from 1 to that scalar.
%       nr          Number of rows in output
%       nc          Number of columns in output
%
% Output:
%       v           Array of random integers
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2014-10-10
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


if nargin < 3,
  nc = 1;
end
if nargin < 2,
  nr = 1;
end
if prod(size(ri)) == 1,
  ri = [1 ri];
end
n = ri(2)-ri(1)+1;
v = ceil(n.*rand(nr,nc)) + ri(1) - 1;

return
