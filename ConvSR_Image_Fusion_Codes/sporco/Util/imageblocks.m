function B = imageblocks(img, sz, st)

% imageblocks -- Extract blocks of specified size from image or
%                array of images
%
% Usage:
%       B = imageblocks(img, sz, st)
%
% Input:
%       img         Image from which to extract blocks
%       sz          Block size vector
%       st          Block step vector
%
% Output:
%       B           Array of image blocks
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2014-10-20
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


if nargin < 3,
  st = [1 1];
end

Nbr = sz(1);
Nbc = sz(2);
[Nir Nic Nih] = size(img);
Nib = length(1:st(1):(Nir-Nbr+1))*length(1:st(2):(Nic-Nbc+1))*Nih;
B = zeros(Nbr, Nbc, Nib);

% Loop over all blocks in images, adding each one to block set
n = 1;
for k = 1:Nih,
  for l=1:st(1):(Nir-Nbr+1),
    for m=1:st(2):(Nic-Nbc+1),
      B(:,:,n) = img(l:(l+Nbr-1), m:(m+Nbc-1), k);
      n = n + 1;
    end
  end
end


return
