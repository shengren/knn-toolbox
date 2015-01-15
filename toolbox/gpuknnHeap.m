function [dist idx time dTime sTime] = gpuknnHeap(query, pdata, k, numStreams)
% gpuknnHeap: MATLAB interface of GPU knns using the Heap Search algorithm
%[dist idx time] = gpuknnHeap(query, pdata objects, k, numStreams)
% query: matrix containing the queries. The COLUMNS correspond to the query
% vectors
% pdata: matrix containing the corpus. The COLUMNS correspond to the coprus
% vectors
% dotp: vector containing the precalculated dot product of the corpus
% objects: The number of vectors in the corpus
% k: The number of neighbors
% numStreams: The number of CUDA streams to be used. Only emulation in
% MATLAB
% dist: matrix containing the distances of the k-nearest neighbors to the queries.
% idx: matrix containing the index of each neighbor
% time: The execution time of the knns in seconds 
% AUTHOR: Nikos Sismanis
% Date: Mar 2012


cls = class(pdata);

if cls == 'single'
  [dist idx time dTime sTime] = mexknnsHeap(single(query), single(pdata), k, numStreams);
 else
   [dist idx time dTime sTime] = mexknnsDHeap(query, pdata, k, numStreams);
end

%[dist I] = sort(dist');
[dist I] = sort(dist);

idx = idx + 1;
for i=1:size(idx,2)
idx(:,i) = idx(I(:,i),i);
end

dist = dist(1:k,:);
idx = idx(1:k,:);

end
