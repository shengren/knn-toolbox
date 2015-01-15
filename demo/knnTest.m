% AUTHOR: Nikos Sismanis
% Date: Mar 2012

clear all
close all



d = 43;
N = 2^15;
q = 256;

k = 16;
streams = 1;

data = single(rand(d, N));
query = single(rand(d, q));

sprintf('Testing with %d dimensional random vectors, %s precision', d, class(data))

[distBF idxBF] = knn(query, data, k);

[distBS idxBS timeBS] = gpuknnBitonic(query, data, k, streams);

[distHS idxHS timeHS] = gpuknnHeap(query, data, k, streams);

ns = abs(max(distBF(:)));
er(1) = norm(distBS(:) - distBF(:)) / ns;
er(2) = norm(distHS(:) - distBF(:)) / ns;

if all(er < 1.0e-02)
  sprintf('PASS\n')
 else
   sprintf('FAIL\n')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

d = 128;
data = single(rand(d, N));
query = single(rand(d, q));

sprintf('Testing with %d dimensional random vectors, %s precision', d, class(data))

[distBF idxBF] = knn(query, data, k);

[distBS idxBS timeBS] = gpuknnBitonic(query, data, k, streams);

[distHS idxHS timeHS] = gpuknnHeap(query, data, k, streams);

ns = abs(max(distBF(:)));
er(1) = norm(distBS(:) - distBF(:)) / ns;
er(2) = norm(distHS(:) - distBF(:)) / ns;

if all(er < 1.0e-02)
sprintf('PASS\n')
else
sprintf('FAIL\n')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

d = 1024;
data = single(rand(d, N));
query = single(rand(d, q));

sprintf('Testing with %d dimensional random vectors, %s precision', d, class(data))
 
[distBF idxBF] = knn(query, data, k);
 
[distBS idxBS timeBS] = gpuknnBitonic(query, data, k, streams);
 
[distHS idxHS timeHS] = gpuknnHeap(query, data, k, streams);
 
ns = abs(max(distBF(:)));
er(1) = norm(distBS(:) - distBF(:)) / ns;
er(2) = norm(distHS(:) - distBF(:)) / ns;
 
if all(er < 1.0e-02)
sprintf('PASS\n')
else
sprintf('FAIL\n')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

d = 2048;
data = single(rand(d, N));
query = single(rand(d, q));

sprintf('Testing with %d dimensional random vectors, %s precision', d, class(data))

[distBF idxBF] = knn(query, data, k);

[distBS idxBS timeBS] = gpuknnBitonic(query, data, k, streams);


[distHS idxHS timeHS] = gpuknnHeap(query, data, k, streams);
 
ns = abs(max(distBF(:)));
er(1) = norm(distBS(:) - distBF(:)) / ns;
er(2) = norm(distHS(:) - distBF(:)) / ns;
 
if all(er < 1.0e-02)
sprintf('PASS\n')
else
sprintf('FAIL\n')
end

