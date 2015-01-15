clear all
close all


d = 47;
N = 2^14;
q = N;
k = 16;
streams = 1;

data = single(rand(d, N));

query = data;

sprintf('Testing all kNN %d dimensional random vectors, %s precision', d, class(data))

[distBF idxBF] = knn(query, data, k);

[distHS idxHS timeHS] = gpuknnHeap(query, data, k, streams);

ns = abs(max(distBF(:)));
er = norm(distHS(:) - distBF(:)) / ns;

if all(er < 1.0e-02)
  sprintf('PASS\n')
 else
   sprintf('FAIL\n')
end

dists = distHS(2:end,:);
idx = idxHS(2:end, :);
