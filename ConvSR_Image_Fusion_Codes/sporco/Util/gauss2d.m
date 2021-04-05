function [G, dyG, dxG, Y, X] = gauss2d(sz, sv, theta)

% gauss2d -- Compute 2D Gaussian distribution, and its partial derivatives
%
% Usage:
%       [G, dxG, dyG, X, Y] = gauss2d(sz, sv, theta)
%
% Input:
%       sz          The size of the Gaussian kernel; either a
%                   scalar for a square block, or a 2-vector giving
%                   height and width
%       sv          The standard deviation of the Gaussian; either
%                   a scalar for an isotropic Gaussian, or a
%                   2-vector giving standard deviations in x- and
%                   y- directions
%       theta       Orientation angle for non-isotropic Gaussian
%
% Output:
%       G           The Gaussian
%       dyG         The y-derivative of the Gaussian
%       dxG         The x-derivative of the Gaussian
%       Y           The y-direction coordinate grid
%       X           The x-direction coordinate grid
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2014-05-29
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


if nargin < 3,
  theta = 0;
end
if nargin < 2 || isempty(sv),
  sv = [1.0 1.0];
end
if nargin < 1 || isempty(sz)
  sz = [5 5];
end
if length(sv) == 1,
  sv = [sv sv];
end
if length(sz) == 1,
  sz = [sz sz];
end

% Effective grid spacing
dx = 1; dy = 1;
% Construct the support for the surface
my = floor(sz(1)/2) - (1-mod(sz(1),2))/2;
mx = floor(sz(2)/2) - (1-mod(sz(2),2))/2;
[X, Y] = meshgrid((-mx:mx)*dx, (-my:my)*dy);

% Compute Gaussian
a = cos(theta)^2/2/sv(2)^2 + sin(theta)^2/2/sv(1)^2;
b = -sin(2*theta)/4/sv(2)^2 + sin(2*theta)/4/sv(1)^2 ;
c = sin(theta)^2/2/sv(2)^2 + cos(theta)^2/2/sv(1)^2;
G = exp(-(a*X.^2 + 2*b*X.*Y + c*Y.^2));
% Ensure correct normalisation
G = G/sum(G(:));

if nargout > 1,
  % Compute the x-derivative of the Gaussian
  dxG = -(sv(2)^(-2))*X.*G;
  % Compute the y-derivative of the Gaussian
  dyG = -(sv(1)^(-2))*Y.*G;
end

return
