function x = snr(ref, sig)

% snr -- Compute Signal-to-Noise Ratio for images
%
% Usage:
%       x = snr(ref, sig)
%
% Input:
%       ref         Reference image
%       sig         Modified image
%
% Output:
%       x           SNR value
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2014-06-03
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


mse = mean(abs(ref(:)-sig(:)).^2);
dv = var(ref(:),1);
x = 10*log10(dv/mse);

return
