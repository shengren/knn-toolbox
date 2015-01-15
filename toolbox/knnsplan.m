function [plan] = knnsplan(n, q, d, k)
% knnsplan: plan the execution of knns in GPU
% plan = knnsplan(n, q, k);
% n: number of vectors in the corpus 
% q: number of queries
% d: number of dimensions
% k: number of neighbors
% plan: the planer
% AUTHOR: Nikos Sismanis
% Date: Mar 2012


iter = 10;

streams = 1;

data = single(rand(d, n));
queries = single(rand(d, q));

nx = log2(n);

for i=1:iter

    %1
    [distBS idxBS timeBS(i)] = gpuknnBitonic(queries, data, k, 1);

    %2
    [distHS idxHS timeHS(i)] = gpuknnHeap(queries, data, k, 1);

end

t = mean([timeBS(:) timeHS(:)]);

[mm ii] = min(t);

switch (ii)

case 1
plan.function = @gpuknnBitonic;

case 2
plan.function = @gpuknnHeap;


otherwise

warning('plan failed');
plan.function = 0;

end

plan.k = k;
plan.streams = streams;

end
