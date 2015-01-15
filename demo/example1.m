% Simple example of the use of MATLAB interface
% AUTHOR: Nikos Sismanis
% Date: Mar 2012

clear all
close all

N = 2^15;
d = 128;
q = 2^7;
k = 128;

corpusfile = '../data/TestRandomCorpusDouble.bin';
queryfile = '../data/TestRandomQueriesDouble.bin';

data = single(rand(d, N));
query = single(rand(d, q));


fid = fopen(corpusfile, 'w');
fwrite(fid, data, 'single');
fclose(fid);

fid = fopen(queryfile, 'w');
fwrite(fid, query, 'single');
fclose(fid);

sprintf('Planning...')
plan = knnsplan(N, q, d, k);


sprintf('executing...')
[dist idx time] = gpuknns(plan, query, data);
dist = sort(dist);


[distRef idxref] = knn(query, data, k);

sprintf('time elapsed: %f sec\n', time)

er = norm(distRef(:) - dist(:)) / max(abs(distRef(:)));


if er < 1e-04
sprintf('PASS')
 else
sprintf('FAIL')
end


