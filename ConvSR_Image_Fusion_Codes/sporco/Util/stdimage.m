function im = stdimage(imnm, rflg, sbdr)

% stdimage -- Get a named standard image
%
% Usage:
%       im = stdimage(imnm, rflg, sbdr)
%
% Input:
%       imnm        String containing the image name
%       rflg        If true (default), read image and return data, otherwise
%                   return full path to image file
%       sbdr        Cell array of subdirectories to search (this
%                   argument should usually not be specified)
%
% Output:
%       im          Image data or image file path
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-07-22
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


if nargin < 3.
  sbdr = {'Std', 'Kodak'};
end
if nargin < 2,
  rflg = 1;
end

% Determine base path to image data
p0 = which('sporco');
K = strfind(p0, filesep);
p1 = p0(1:K(end)-1);
bp = [p1 filesep 'Data'];

% Try to find specified file
ip = [];
for k = 1:length(sbdr),
  ipt = [bp filesep sbdr{k} filesep imnm '.png'];
  if exist(ipt,'file'),
    ip = ipt;
    break;
  end
end

if rflg && ~isempty(ip),
  im = imread(ip);
else
  im = ip;
end

return
