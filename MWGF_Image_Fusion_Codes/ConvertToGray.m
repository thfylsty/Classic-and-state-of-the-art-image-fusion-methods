% ConvertToGray.m
% -------------------------------------------------------------------
%
% Authors: Sun Li
% Date:    05/05/2013
% Last modified: 05/05/2013
% -------------------------------------------------------------------

function imgGray = ConvertToGray(img)
    if size(img, 3) == 3,
        imgGray = RGBTOGRAY(img);
    else 
        imgGray = img;
    end
end