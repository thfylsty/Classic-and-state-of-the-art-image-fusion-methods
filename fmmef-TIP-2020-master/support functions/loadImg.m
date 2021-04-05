% This procedure loads a sequence of images
%
% Arguments:
%   'path', refers to a directory which contains a sequence of JPEG or PPM
%   images
%   'reduce' is an optional parameter that controls downsampling, e.g., reduce = .5
%   downsamples all images by a factor of 2.
%
% tom.mertens@gmail.com, August 2007
%

function I = loadImg(path, reduce)

if ~exist('reduce', 'var')
    reduce = 1;
end

if (reduce > 1 || reduce <= 0)
    error('reduce must fulfill: 0 < reduce <= 1');
end

% find all JPEG or PPM files in directory
files = dir([path '/*.tif']);
N = length(files);
if (N == 0)
    files = dir([path '/*.jpg']);
    N = length(files);
    if (N == 0)
        files = dir([path '/*.gif']);
        N = length(files);
        if (N == 0)
            files = dir([path '/*.bmp']);
            N = length(files);
            if (N == 0)
                files = dir([path '/*.png']);
                N = length(files);
                if (N == 0)
                    error('no files found');
                end
            end
        end
    end
end

% allocate memory
sz = size(imread([path '/' files(1).name]));
r = floor(sz(1)*reduce);
c = floor(sz(2)*reduce);
I = zeros(r,c,3,N);

% read all files
for i = 1:N
    
    % load image
    filename = [path '/' files(i).name];
    im = im2double(imread(filename));
    if (size(im,1) ~= sz(1) || size(im,2) ~= sz(2))
        error('images must all have the same size');
    end
    
    % optional downsampling step
    if (reduce < 1)
        im = imresize(im,[r c],'bicubic');
    end
    if size(im,3)==1
    I(:,:,:,i) = cat(3,im,im,im);
    else
    I(:,:,:,i) = im;
    end
end
