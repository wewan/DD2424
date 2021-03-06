function[X,Y,y] = loadBatch(filename)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Imagedata = load(filename);
% X (d x N) d:dimensionality of each image N:image number
X = double(Imagedata.data')./255;
% y (N x 1) is the vector label of images 
y = Imagedata.labels+1;
% Y (k x N) k is class number
Y = zeros(length(unique(y)),length(y));
for i = 1: length(y)
    Y(y(i),i) = 1;
end