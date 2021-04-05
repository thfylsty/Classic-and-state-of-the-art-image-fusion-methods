function x = mse(ref, sig)

% mse -- Compute Mean Squared Error for images
%
% Usage:
%       x = mse(ref, sig)
%
% Input:
%       ref         Reference image
%       sig         Modified image
%
% Output:
%       x           MSE value
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2014-06-03
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


x = mean(abs(ref(:)-sig(:)).^2);

return
