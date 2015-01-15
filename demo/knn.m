function [D, N] = knn(X, Y, k)
% This function is a mtlab verion of the knn algorithm to be used as reference


 [D, N] = sort(bsxfun(@plus,(-2)*(Y'*X),dot(Y,Y,1)'));
	       
 N = N(1:k,:);
 D = D(1:k,:);
 D = bsxfun(@plus,D,dot(X,X,1));


