function  [ind_to_char,char_to_ind,book_data] = Read_Data(book_fname)
% book_fname = 'data/Goblet.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);
book_chars = unique(book_data);
char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');
char_len = size(book_chars,2);
for i = 1:char_len
    char_to_ind(book_chars(i)) = i;
    ind_to_char(i) = book_chars(i);
end