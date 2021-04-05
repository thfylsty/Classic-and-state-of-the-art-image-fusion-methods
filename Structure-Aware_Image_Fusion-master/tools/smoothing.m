function q = smoothing(I, r,eps,N)
    mean_I = boxfilter(I, r) ./ N;
    mean_II = boxfilter(I.*I, r) ./ N;
    var_I = mean_II - mean_I .* mean_I;
    a = var_I ./ (var_I + eps); 
    q = mean_I+ a.*(I-mean_I); 
end