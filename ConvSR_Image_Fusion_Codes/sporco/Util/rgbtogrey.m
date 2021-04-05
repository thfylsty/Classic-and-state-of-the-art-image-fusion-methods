function u = rgbtogrey(v)

% rgbtogrey -- Create greyscale image from RGB image
%
% Usage:
%       u = rgbtogrey(v)
%
% Input:
%       v           Input RGB image
%
% Output:
%       u           Output greyscale image
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2014-05-29
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


if size(v,3) == 1,
  u = v;
elseif size(v,3) == 3,
  C = [0.30 0.59 0.11];
  u = C(1)*v(:,:,1) +  C(2)*v(:,:,2) + C(3)*v(:,:,3);
else
  warning('rgbtogrey: input image must have 1 or 3 bands');
  u = [];
end

return
