% GradientMethod.m
% -------------------------------------------------------------------
% 
% Authors: Sun Li
% Date:    14/03/2013
% Last modified: 2/04/2013
% -------------------------------------------------------------------

function [dx, dy] = GradientMethod(img, meth)
    if ~exist('meth', 'var'),
        meth = 'defaute';
    end
    if strcmp(meth, 'sobel'),
        disp('Sobel Gradient');
        Sv = [-1 -2 -1;...
               0  0  0;...
               1  2  1];
        Sh = [-1  0  1;
              -2  0  2;
              -1  0  1];
        dx=conv2(img, -Sh/8,'same'); % 卷积加个负号
        dy=conv2(img, -Sv/8,'same');
    elseif strcmp(meth, 'zhou')
        fx=[ 0,  0, 0;...
            -1,  1, 0;...
             0,  0, 0];
        fy=[0,  -1, 0;...
            0,  1, 0;...
            0,  0 0];
        dx = imfilter(img, fx, 0, 'corr');
        dy = imfilter(img, fy, 0, 'corr');
    else  
        disp('The default gradient ...');
        [dx, dy] = gradient(img);
    end
    
    step = 1;
    dx(1:step, :) = 0;
    dx(end-step+1:end, :) = 0;
    dx(:, 1:step) = 0;
    dx(:, end-step+1:end) = 0;
    
    dy(1:step, :) = 0;
    dy(end-step+1:end, :) = 0;
    dy(:, 1:step) = 0;
    dy(:, end-step+1:end) = 0;
    
end