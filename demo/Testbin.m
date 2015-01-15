% Simple test of the C interface using streaming
% AUTHOR: Nikos Sismanis
% Date: Mar 2012

clear all
close all

system(sprintf('rm -f KNNdist.bin KNNidx.bin'));

N = 2^12;
D = 128;
Q = 2^12;
d = 128;
k = 128;

sprintf('Testing Single precision')

corpusfile = '../data/TestRandomCorpusSingle.bin';
queryfile = '../data/TestRandomQueriesSingle.bin';

data = single(rand(d, N));
queries = single(rand(d, Q));

fid = fopen(corpusfile, 'w');
fwrite(fid, data, 'single');
fclose(fid);

fid = fopen(queryfile, 'w');
fwrite(fid, queries, 'single');
fclose(fid);

system(sprintf('./knnTest %s %s %d %d %d %d %d', corpusfile, queryfile, N, Q, D, k));

[distBS idx] = importData(Q, k, 'single');
[distBS I] = sort(distBS);

%fid = fopen(sprintf('%s', corpusfile));
%data = single(fread(fid, [D, N], 'single'));
%fclose(fid);

%fid = fopen(sprintf('%s', queryfile));
%queries = single(fread(fid, [D, Q], 'single'));
%fclose(fid);

[distRef idxRef] = knn(queries, data, k);

ns = abs(max(distRef(:)));
err = norm(distRef(:) - distBS(:)) / ns;


if all(err <1e-02)
'PASS'
else
'FAIL'
end


sprintf('Testing Double precision');

corpusfile = '../data/TestRandomCorpusDouble.bin';
queryfile = '../data/TestRandomQueriesDouble.bin';

data = rand(d, N);
queries = rand(d, Q);


fid = fopen(corpusfile, 'w');
fwrite(fid, data, 'double');
fclose(fid);

fid = fopen(queryfile, 'w');
fwrite(fid, queries, 'double');
fclose(fid);

system(sprintf('./knnTestDouble %s %s %d %d %d %d %d', corpusfile, queryfile, N, Q, D, k));

[distBS idx] = importData(Q, k, 'double');
[distBS I] = sort(distBS);


[distRef idxRef] = knn(queries, data, k);

ns = abs(max(distRef(:)));
err = norm(distRef(:) - distBS(:)) / ns;

if all(err <1e-06)
'PASS'
  else
'FAIL'
end


