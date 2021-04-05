function u = vec(v)

% vec -- Vectorise image
%
% Usage:
%       u = vec(v)
%
% Input:
%       v           Input image
%
% Output:
%       u           Vectorised image
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2014-05-29
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


u = v(:);

return
