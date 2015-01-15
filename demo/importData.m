function [KNNdist KNNidx] = importData(Q, k, type)

  fid = fopen('KNNdist.bin');

  KNNdist = fread(fid, [k Q], type);

fclose(fid);


fid = fopen('KNNidx.bin');

KNNidx = fread(fid, [k Q], type);

fclose(fid);

end
