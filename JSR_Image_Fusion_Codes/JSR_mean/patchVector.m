function V = patchVector(I, unit, v_num)
% 将图像分块，并组成列向量矩阵
[m,n] = size(I);
m1 = floor(m/unit);
n1 = floor(n/unit);

count = 0;
for i=1:m1
    for j=1:n1
        count = count+1;
        p = I((i-1)*unit+1:i*unit,(j-1)*unit+1:j*unit);
        V_all(:, count) = p(:);
    end
end

index = randperm(count);
index_num = index(1:v_num);
V_p = V_all(:,index_num);
V = V_p;

end