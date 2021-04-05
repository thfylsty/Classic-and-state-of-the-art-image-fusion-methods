function f = fuison_base_parts(b1, b2)
% ZCA + l1 norm
% featrue1 = whitening_norm(b1, 1, 0);
% featrue2 = whitening_norm(b2, 1, 0);
% figure;imshow(featrue1);
% figure;imshow(featrue2);

% f_sum = featrue1+featrue2;
% f_sum(f_sum==0) = 1;

% w1 = featrue1./f_sum;
% w2 = featrue2./f_sum;
% figure;imshow(w1);
% figure;imshow(w2);

w1 = 0.5;
w2 = 0.5;
f = w1.*b1+w2.*b2;
% figure;imshow(f);
end
