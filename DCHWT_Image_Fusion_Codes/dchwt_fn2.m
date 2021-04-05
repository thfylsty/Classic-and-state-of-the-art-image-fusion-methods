function [C c]=dchwt_fn2(x,nlevel)

%%% dchwt_fn2: Multi-level 2-D Discrete Cosine Harmonic Wavelet Transform
%%% (DCHWT)
%%% Note: Approximate Subband is stored in Last Cell.
%%% c=dchwt_fn2(x,nlevel) does the follows according to the value of nlevel:
%%%   nlevel > 0:   decomposes image x up to nlevel level;
%%%   nlevel < 0:   does the inverse transform to nlevel level;
%%%   nlevel = 0:   sets c equal to x;
%%%
%%% Example:
%%%     Y = dchwt_fn2(X, 5);    % Decompose X up to 5 level
%%%     R = dchwt_fn2(Y, -5);   % Reconstruct from Y
%%%
%%% Author : B.K. SHREYAMSHA KUMAR 
%%% Created on 20-11-2009.
%%% Updated on 02-12-2009.


[p,q]=size(x);
arow=floor(p/2^abs(nlevel));
acol=floor(q/2^abs(nlevel));
k=1;

%----------------  remain unchanged when nlevel = 0  -------------------%
if nlevel==0
    c=x;
    C=x;
%--------------------  decomposition,  if nlevel > 0  ------------------%
%%% Using one DCT and many IDCTs.
elseif nlevel > 0
    Id=dct2(x);
    temp{k}=idct2(Id(1:arow,1:acol)); %% Approximate Coeff. is stored in First Cell.
    xwt=temp{k};
    k=k+1;
    for i = 1:nlevel
        rr=arow*2^(i-1);
        cc=acol*2^(i-1);
        temp{k}=idct2(Id(1:rr,cc+1:2*cc)); %% Vertical
        temp{k+1}=idct2(Id(rr+1:2*rr,1:cc)); %% Horizontal
        temp{k+2}=idct2(Id(rr+1:2*rr,cc+1:2*cc)); %% Diagonal
        xwt=[xwt temp{k}; temp{k+1} temp{k+2}];
        k=k+3;
    end
    c=xwt;
    len=size(temp,2);
    for i=1:len
        C{i}=temp{len-i+1}; %% Now Approximate Coeff. is stored in Last Cell.
    end

%--------------------  reconstruction,  if nlevel < 0  -----------------%
else
    k=size(x,2);
    xtemp=dct2(x{k});
    k=k-1;
    for i=1:abs(nlevel)
        xtemp=[xtemp dct2(x{k});dct2(x{k-1}) dct2(x{k-2})]; % [App Vert;Horz Diag];
        k=k-3;
    end
    C=idct2(xtemp);
end