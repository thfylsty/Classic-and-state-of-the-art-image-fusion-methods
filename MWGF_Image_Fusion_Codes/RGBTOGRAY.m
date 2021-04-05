% RGBTOGRAY.m
% -------------------------------------------------------------------
%
% Date:    27/04/2013
% Last modified: 27/04/2013
% -------------------------------------------------------------------

function gray = RGBTOGRAY(img)

    if size(img, 3) ~= 3,
        error('The input should be RGB');
    end
    
    if ~isinteger(img) && max(img(:)) > 1,
        gray = rgb2gray(uint8(img));
        gray = double(gray);
    else
        gray = rgb2gray(img);
    end
    
end