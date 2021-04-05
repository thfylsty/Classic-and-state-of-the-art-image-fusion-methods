function [Y, Uf, Vf] = fftconv2(U, V, shape)

% fftconv2 -- Perform 2D convolution via FFT
%
% Usage:
%       Y = fftconv2(U, V, shape)
%
% Input:
%       U           Image to convolve
%       V           Image to convolve
%       shape       Specify the size of the computed
%                   convolution. Valid values are 'full' and
%                   'same', having the same meanings as in the
%                   conv2 function
%
% Output:
%       Y           2D convolution product
%       Uf          Appropriately padded FFT of U
%       Vf          Appropriately padded FFT of V
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2014-05-29
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


if nargin < 3,
  shape = 'full';
end

if ~iscell(U) && ~iscell(V),
  su = size(U);
  sv = size(V);
  sp = su + sv - 1;
  ru = isreal(U);
  rv = isreal(V);
else
  if iscell(U),
    sp = size(U{1});
    su = U{2};
    sv = sp - su + 1;
    ru = U{3};
    rv = isreal(V);
  else
    sp = size(V{1});
    sv = V{2};
    su = sp - sv + 1;
    ru = isreal(U);
    rv = V{3};
  end
end

if iscell(U),
  Uf = U{1};
else
  Uf = fft2(U, sp(1), sp(2));
end
if iscell(V),
  Vf = V{1};
else
  Vf = fft2(V, sp(1), sp(2));
end
% Perform the (zero-padded) convolution
Y = ifft2(Uf .* Vf);

% Ensure real output for real input
if ru && rv, Y = real(Y); end

switch lower(shape)
 case 'full'
  % Nothing to do
 case 'same'
  bl = ceil((sv-1)/2) + 1;
  bu = su + bl - 1;
  Y = Y(bl(1):bu(1), bl(2):bu(2),:);
 case 'valid'
  error(sprintf('Shape value \"valid\" is not implemented', shape));
 otherwise
  error(sprintf('Invalid shape value \"%s\"', shape));
end

return
