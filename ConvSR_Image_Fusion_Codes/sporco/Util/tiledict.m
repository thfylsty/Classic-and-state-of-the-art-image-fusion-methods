function im = tiledict(D, sz)

% tiledict -- Construct an image allowing visualization of dictionary
%             content
%
% Usage:
%       im = tiledict(D, sz)
%
% Input:
%       D           Dictionary matrix
%       sz          Size of each block in dictionary
%
% Output:
%       im          Image tiled with dictionary entries
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-02-12
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.

% Handle standard 2D (non-convolutional) dictionary
if ndims(D) == 2,
  D = reshape(D, [sz size(D,2)]);
  sz = [];
end
dsz = size(D);

% Construct dictionary atom size vector if not provided
if nargin < 2 || isempty(sz),
  sz = repmat(dsz(1:2)', [1 size(D,3)]);
end
% Compute the maximum atom dimensions
mxsz = max(sz');

% Shift and scale values to [0, 1]
D = D - min(D(:));
D = D / max(D(:));

% Construct tiled image
N = dsz(3);
Vr = floor(sqrt(N));
Vc = ceil(N/Vr);
im = ones(Vr*mxsz(1) + Vr-1, Vc*mxsz(2) + Vc-1, 1);
k = 1;
for l = 0:(Vr-1),
  for m = 0:(Vc-1),
    r = mxsz(1)*l + l + 1;
    c = mxsz(2)*m + m + 1;
    im(r:(r+sz(1,k)-1),c:(c+sz(2,k)-1),:) = D(1:sz(1,k), 1:sz(2,k), k);
    k = k + 1;
    if k > N,
      break;
    end
  end
  if k > N,
    break;
  end
end

return
