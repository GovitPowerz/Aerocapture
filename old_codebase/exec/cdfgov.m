function [xcdf,ycdf] = cdfgov(x)

n = length(x);
x = sort(x');
y = (1:n)'/n;
notdup = ([diff(x); 1] > 0);
x = x(notdup);
y = [0; y(notdup)];
k = length(x);
l = reshape(repmat(1:k, 2, 1), 2*k, 1);

xcdf = [-Inf; x(l); Inf];
ycdf = [0; 0; y(1+l)];

return;
