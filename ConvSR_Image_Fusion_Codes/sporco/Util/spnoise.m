function [sn, msk] = spnoise(s, frc)

% spnoise -- Apply salt & pepper noise to image
%
% Usage:
%       [sn, msk] = spnoise(s, frc)
%
% Input:
%       s         Input image or 3d array of images
%       frc       Desired fraction of pixels corrupted by noise
%
% Output:
%       sn        Noisy image
%       msk       Mask indicating corrupted pixel locations
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.

spm = 2.0 * rand(size(s)) - 1.0;
sn = s;
sn(spm < frc - 1.0) = 0;
sn(spm > 1.0 - frc) = 1;
msk = zeros(size(spm));
msk(spm < frc - 1.0) = 1;
msk(spm > 1.0 - frc) = 1;

return
