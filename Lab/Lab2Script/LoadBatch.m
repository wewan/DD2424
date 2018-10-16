function[X,Y,y] = LoadBatch(filename)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Imagedata = load(filename);
% X (d x N) d:dimensionality of each image N:image number
X = double(Imagedata.data')./255;
% y (N x 1) is the vector label of images 
y = double(Imagedata.labels+1);
% Y (k x N) k is class number
Y = double(zeros(length(unique(y)),length(y)));
for i = 1: length(y)
    Y(y(i),i) = 1;
end