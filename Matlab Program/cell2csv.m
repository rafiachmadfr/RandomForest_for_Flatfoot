function cell2csv(fileName, cellArray)
    fid = fopen(fileName, 'w');
    [rows, cols] = size(cellArray);
    for i = 1:rows
        for j = 1:cols
            var = cellArray{i, j};
            if isnumeric(var)
                fprintf(fid, '%f', var);
            else
                fprintf(fid, '%s', var);
            end
            if j ~= cols
                fprintf(fid, ',');
            end
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end