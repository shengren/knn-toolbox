% AUTHOR: Nikos Sismanis
% Date: Mar 2012

clear all
close all

N = 2^15;
d = 43;
q = 64;
k = 128;
streams = 1;

data = rand(d, N);
query = rand(d, q);

sprintf('Testing with %d dimensional random vectors, %s precision', d, class(data))

[distBF idxBF] = knn(query, data, k);

%'Truncated Heap sort'
[distHS idxHS timeHS] = gpuknnHeap(query, data, k, streams);

%'Truncated Bitonic sort'
[distBS idxBS timeBS] = gpuknnBitonic(query, data, k, streams);

ns = abs(max(distBF(:)));
er(1) = norm(distBF(:) - distHS(:)) / ns;
er(2) = norm(distBF(:) - distBS(:)) / ns;

if all(er < 1.0e-07)
  sprintf('PASS\n')
 else
   sprintf('FAIL\n')
end


d = 128;
data = rand(d, N);
query = rand(d, q);

sprintf('Testing with %d dimensional random vectors, %s precision', d, class(data))

[distBF idxBF] = knn(query, data, k);

%'Truncated Heap sort'
[distHS idxHS timeHS] = gpuknnHeap(query, data, k, streams);

%'Truncated Bitonic sort'
[distBS idxBS timeBS] = gpuknnBitonic(query, data, k, streams);


ns = abs(max(distBF(:)));
er(1) = norm(distBF(:) - distHS(:)) / ns;
er(2) = norm(distBF(:) - distBS(:)) / ns;


if all(er < 1.0e-07)
sprintf('PASS\n')
else
sprintf('FAIL\n')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


d = 1024;
data = rand(d, N);
query = rand(d, q);

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


d = 2048;
data = rand(d, N);
query = rand(d, q);

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

