function x = psnr(ref, sig)

% snr -- Compute Peak Signal-to-Noise Ratio for images
%
% Usage:
%       x = psnr(ref, sig)
%
% Input:
%       ref         Reference image
%       sig         Modified image
%
% Output:
%       x           PSNR value
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-07-10
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


mse = mean(abs(ref(:)-sig(:)).^2);
dv = (max(ref(:)) - min(ref(:)))^2;
x = 10*log10(dv/mse);

return
