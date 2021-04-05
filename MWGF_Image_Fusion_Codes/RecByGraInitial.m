% RecoverByGraInitial.m 
% RecoverByGra.m
% -------------------------------------------------------------------
% This function is just the joint of RecoverByGra and initial f0
% Date:    17/03/2013
% Last modified: 16/04/2015
% -------------------------------------------------------------------

function [imgRec, rms, gObj] = RecByGraInitial(img1, img2, ww1, ww2, dxdy, iter, res, alpha, iniMode)

    % ------------ Check parameter --------------
%     narginchk(9, 9);
%     if size(img1, 3) ~=1 || size(img2, 3) ~=1,
%         error('The image should be gray');
%     end
    % -------------------------------------------
    switch lower(iniMode),
        case 'avg',
            f0 = (img1+img2)/2;
        case 'weight',
            f0 = ww1.*img1+ww2.*img2;
        otherwise
            error('There only two mode');
    end
    
    [imgRec, rms, gObj] = RecoverByGra(f0, dxdy, iter, res, alpha);
    
end

%%
function [imgRec, rms, gObj] = RecoverByGra(imgOri, Obj, iter, res, alpha)

    lp = [0 1 0;1 -4 1;0 1 0];
    if isreal(Obj),
%         disp('The original is the Laplace');
        gObj = Obj;
    else
%         disp('The original is the Gradient');
        gObj = LaplaceZ(Obj);
    end
    
    rms = [];
%     f0 = Boundary(imgOri, dxObj, dyObj);
    f0 = imgOri;
    for ii = 1:iter,
%         f0 = Boundary(f0, dxObj, dyObj);
        deltaF = imfilter(int16(f0), lp, 'replicate', 'corr');
        deltaF = double(deltaF);
        delta = alpha * (deltaF-gObj);

        f1 = f0 + delta;

        f1 = max(f1, 0);
        f1 = min(f1, 255);


        f0 = f1;  
    end
    
    disp(['The ' num2str(ii) ' iteration is complete.']);
    imgRec = f1;
end

%%
function lap = LaplaceZ(IMGORGRA)

    if isreal(IMGORGRA),
%         disp('The input should be IMAGE')
        lh=[0,  1, 0;...
            1, -4, 1;...
            0,  1, 0];
        lap = imfilter(IMGORGRA, lh, 'replicate', 'corr');
    else
%         disp('The input should be GRADIENT');
        fx=[0, 0, 0; 0, -1, 1; 0, 0, 0];
        fy=[0, 0, 0;  0, -1, 0; 0, 1, 0];

        dx = real(IMGORGRA);
        dy = imag(IMGORGRA);

        ddx = imfilter(dx, fx, 0, 'corr');
        ddy = imfilter(dy, fy, 0, 'corr');

        lap=ddx + ddy;
    end
end