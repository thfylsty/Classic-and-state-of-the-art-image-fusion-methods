function out = mirror_extend(in,bx,by)
% Pad array with mirror reflections of itself.
% bx and by specify the amount of padding to add.
% ***********************************************

[h,w] = size(in);

%First flip up and down
u = flipud(in(2:1+bx,:));
d = flipud(in(h-bx:h-1,:));

in2 = [u' in' d']';

%Next flip left and right
l = fliplr(in2(:, 2:1+by));
r = fliplr(in2(:,w-by:w-1));

%set the 'mirrored' image to out.
out = [l in2 r];

return

%test
% A = [1 2 3 4;5 6 7 8;9 10 11 12]
% B = mirror_extend(A,2,2)

