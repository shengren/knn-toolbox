function [] = exportData(data, query, type)

if type ~= 'double'
sdata = single(data);
squery = single(query);
 else
sdata = data;
squery = query;
end

nt = 2^22;

%dp = dot(data, data);


N = size(data, 2);

N

if N <= nt
dp = dot(data, data);
 else

for i=1:nt:N
        dp(i:i+nt-1) = dot(data(:,i:i+nt-1), data(:,i:i+nt-1));
end

end


fid = fopen('TestData.bin', 'w');

fwrite(fid, sdata, type);

fclose(fid);


fid = fopen('QueryData.bin', 'w');

fwrite(fid, squery, type);

fclose(fid);

fid = fopen('dp.bin', 'w');

fwrite(fid, dp, type);

fclose(fid)

end
