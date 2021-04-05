%   The code was writed by Jicai Teng, Kun Zhan
%   $Revision: 1.0.0.0 $  $Date: 2013/09/16 $ 12:25:47 $

%   Reference:
%   C. Xydeas and V. Petrovic, 
%   "Objective image fusion performance measure," 
%   Electronics Letters, vol. 36, no. 4, pp. 308-309, 2000.

function Q=Qp_ABF(A,B,F)
    [Q_AF, g_A] = EdgePreservation(A,F);
    [Q_BF, g_B] = EdgePreservation(B,F);

    L=1;
    w_A=g_A.^L;
    w_B=g_B.^L;

    Q=sum(sum(Q_AF.*w_A+Q_BF.*w_B))/sum(sum(w_A+w_B));
end
function [g_X, alpha_X]=g_alpha(x)
    X=padarray(x,[1 1],'symmetric');
    X=im2double(X);
    sobel_x=[-1 -2 -1;0 0 0;1 2 1];
    S_X_x=filter2(sobel_x,X,'valid');
    S_X_y=filter2(sobel_x',X,'valid');

    g_X=sqrt(S_X_x.^2+S_X_y.^2);
    alpha_X = atan2(S_X_y,S_X_x);
end
function [Q_XF, g_X] = EdgePreservation(X,F)
    [g_X, alpha_X]=g_alpha(X);
    [g_F, alpha_F]=g_alpha(F);

    G_XF = min(g_X,g_F)./(max(g_F,g_X)+eps);

    A_XF=1-abs(alpha_X-alpha_F)./(pi/2);

    Gamma_g=0.9994;K_g=-15;sigma_g=0.5;
    Q_g_XF=Gamma_g./(1+exp(K_g*(G_XF-sigma_g)));
    Gamma_alpha=0.9879;K_alpha=-22;sigma_alpha=0.8;
    Q_alpha_XF=Gamma_alpha./(1+exp(K_alpha*(A_XF-sigma_alpha)));

    Q_XF=Q_g_XF.*Q_alpha_XF;
end