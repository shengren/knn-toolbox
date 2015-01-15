function [dist idx time dTime sTime] = gpuknnBitonic(query, pdata, k, numStreams)
% gpuknnBitonic: MATLAB interface of GPU knns using the Bitonic Search algorithm
%[dist idx time] = gpuknnBitonic(query, pdata, objects, k, numStreams)
% query: matrix containing the queries. The COLUMNS correspond to the query
% vectors
% pdata: matrix containing the corpus. The COLUMNS correspond to the corpus
% vectors
% objects: The number of vectors in the corpus
% k: The number of neighbors
% numStreams: The number of CUDA streams to be used. Only emulation in
% MATLAB
% AUTHOR: Nikos Sismanis
% Date: Mar 2012

cls = class(pdata);

if cls == 'single'
  [dist idx time dTime sTime] = mexknnsBitonic(single(query), single(pdata), k, numStreams);
 else
   [dist idx time dTime sTime] = mexknnsDBitonic(query, pdata, k, numStreams);
end
  
  
idx = idx + 1;

[dist I] = sort(dist);

for i=1:size(dist,2)
	idx(:,i) = idx(I(:,i),i);
end

dist = dist(1:k,:);
idx = idx(1:k,:);

end
