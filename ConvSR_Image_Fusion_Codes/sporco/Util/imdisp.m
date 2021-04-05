function h = imdisp(im, opt)

% imdisp -- Display an image
%
% Usage:
%       h = imdisp(im, opt)
%
% Input:
%       im          Image to display
%       opt         Display options
%
% Output:
%       h           Handle to image object
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2014-05-29
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


if nargin < 2,
  opt = [];
end

% Check for box plot region options
brgn = [];
if isfield(opt,'bregion'),
  brgn = opt.bregion;
end

% Check for zoom region options
zrgn = [];
if isfield(opt,'zregion'),
  zrgn = opt.zregion;
elseif isfield(opt,'zgap') && isfield(opt,'bregion'),
  zrgn = opt.bregion + opt.zgap*[-1 -1 1 1];
end
if ~isempty(zrgn),
  im = im(zrgn(1):zrgn(3),zrgn(2):zrgn(4),:);
  if ~isempty(brgn),
    brgn = brgn - [zrgn(1:2) zrgn(1:2)];
  end
end

% Handle image offset and scaling
if 0,
if isfloat(im) && min(im(:)) < 0,
  im = im - min(im(:));
end
if isfloat(im) && max(im(:)) > 1,
    im = im / max(im(:));
end
else
  if isfloat(im),
    im = im - min(im(:));
    im = im / max(im(:));
  end
end

% Different behaviour for greyscale and RGB images
if size(im,3) == 1,
  hh = imagesc(im);
  colormap(gray);
else
  hh = image(im);
end
axis image; axis off;

% Plot box region if option supplied
if ~isempty(brgn),
  bx = brgn([2 2 4 4 2]) + [-0.5 -0.5 0.5  0.5 -0.5] + 1;
  by = brgn([1 3 3 1 1]) + [-0.5  0.5 0.5 -0.5 -0.5] + 1;
  bl = 'r-';
  if isfield(opt,'bline'),
    bl = opt.bline;
  end
  hold on;
  plot(bx, by, bl);
  hold off;
end

% Check for title option
if isfield(opt, 'title'),
  title(opt.title);
end

if nargout > 0
    h = hh;
end

return
