function [dist idx time] = gpuknns(plan, queries, data)
% gpuknns: execution of knns in GPU
% [dist idx time] = gpuknns(plan, queries, data);
% plan: planer of the execution
% queries: matrix containing the queries. The COLUMNS of the matrix
% correspond to the query vectors
% data: matrix containing the corpus. The COLUMNS of the matrix correspond
% to the corpus vectors
% dist: matrix containing the distances of the k-nearest neighbors to the queries.
% idx: matrix containing the index of each neighbor
% time: The execution time of the knns in seconds 
% AUTHOR: Nikos Sismanis
% Date: Mar 2012


N = size(data, 2);

  [dist idx time] = plan.function(queries, data, plan.k, plan.streams);

end
