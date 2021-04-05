
function I = reconstruction(temp_vector, unit, t1,t2, count1)
I_d_temp = zeros(t1,t2);
for ii=1:t1-unit+1
    for jj=1:t2-unit+1
        temp = temp_vector(:,(ii-1)*(t2-unit+1)+jj);
        I_d_temp(ii:(unit+ii-1), jj:(unit+jj-1)) = I_d_temp(ii:(unit+ii-1), jj:(unit+jj-1)) + reshape(temp, [unit unit]);
    end
end
I = I_d_temp./count1;
end

