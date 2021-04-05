function out = calcFocusMeasure(img, bb, fmeasure)

switch fmeasure
                  
    case 'SAG'
        H = fspecial('sobel');
        img = mirror_extend(img,1,1);
        Gx = conv2(img,H,'valid');
        Gy = conv2(img,H','valid');
        grad = abs(Gx) + abs(Gy);
        out = imfilter(grad,fspecial('average', bb),'symmetric');        
        
    case 'VOI'
        [M, N] = size(img);
        I_ext = mirror_extend(img,(bb-1)/2,(bb-1)/2);
        fun = @(x) std(x);
        tmp = colfilt(I_ext,[bb bb],'sliding',fun);
        out = tmp((bb+1)/2:M+(bb-1)/2,(bb+1)/2:N+(bb-1)/2);                    
        
    case 'SF'
        [M, N] = size(img);
        I_ext = mirror_extend(img,1,1);
        h=[0 1 -1];
        diffh = conv2c(I_ext,h);
        diffh = diffh(2:M+1,2:N+1);
        h=[0 1 -1]';
        diffv = conv2c(I_ext,h);
        diffv = diffv(2:M+1,2:N+1);
        RF2 = imfilter(diffv.^2,fspecial('average', bb),'symmetric');
        CF2 = imfilter(diffh.^2,fspecial('average', bb),'symmetric');
        out = sqrt(RF2+CF2);
        
   case 'EOL'
        H = fspecial('laplacian');
        img = mirror_extend(img,1,1);
        L = conv2(img,H,'valid');
        if (bb>0)
            out = imfilter(L.^2,fspecial('gaussian', bb,bb/6),'symmetric');
        else
            out = L.^2;
        end
        
   case 'AOL'
        H = fspecial('laplacian');
        img = mirror_extend(img,1,1);
        L = conv2(img,H,'valid');
        if (bb>0)
            out = imfilter(abs(L),fspecial('gaussian', bb,bb/6),'symmetric');
        else
            out = abs(L);
        end
end
return

