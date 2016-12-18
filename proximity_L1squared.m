function x = proximity_L1squared(x0, lambda)
d = length(x0);
s = sign(x0);
x0 = abs(x0);
[y0,ind_sort] = sort(x0, 'descend');
ycum = cumsum(y0);
val = (lambda ./ (1 + (1:d)*lambda)) .* ycum;
ind = find(y0 > val);
rho = ind(end);
tau = val(rho);
y0(end);
y = y0 - tau;
ind = find(y < 0);
y(ind) = 0;
x(ind_sort) = y;
x = s .* x;