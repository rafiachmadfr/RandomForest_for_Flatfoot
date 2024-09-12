% Tentukan direktori input dan output
inputDir = 'D:\SIKIL\Dataset Flat';
outputDir = 'D:\SIKIL\HOG Flat';

% Buat direktori output jika belum ada
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

jpgFiles = dir(fullfile(inputDir, '*.jpg'));

% Inisialisasi tabel untuk menyimpan fitur HOG
hogTable = cell(length(jpgFiles) + 1, 61); % Header csv
hogTable(1,:) = [{'nama file'}, arrayfun(@(x) sprintf('bin %d', x), 1:60, 'UniformOutput', false)];

for i = 1:length(jpgFiles)
    img = imread(fullfile(inputDir, jpgFiles(i).name));
    Img_gray = im2gray(img);
    Img_resized = imresize(Img_gray, [128 64]);
   
    %% HOG
    % Ekstraksi fitur HOG
    cellSize = [8 8];
    numBins = 60;
    [hogFeature, visualization] = extractHOGFeatures(Img_resized, 'CellSize', cellSize,'NumBins',numBins);

    % Visualisasi HOG
    figure;
    set(gcf, 'Position', [10, 10, 1000, 500]);
    subplot(1, 2, 1);
    set(gca, 'Position', [0.01, 0.1, 0.25, 0.8]);
    imshow(Img_resized); hold on;
    plot(visualization);
    title('Visualisasi Fitur HOG');

    % Membuat histogram dari fitur HOG
    binEdges = linspace(0, 180, numBins+1); % Rentang sudut dari 0 hingga 180 derajat
    hogFeatureReshaped = reshape(hogFeature, [], numBins); % Bentuk ulang fitur HOG menjadi matriks
    hogHistogram = sum(hogFeatureReshaped, 1); % Jumlahkan gradien di setiap bin

    % Visualisasi histogram
    subplot(1, 2, 2);
    set(gca, 'Position', [0.35, 0.1, 0.6, 0.8]);
    bar(binEdges(1:end-1), hogHistogram, 'histc');
    xlim([0 180]);
    xlabel('Sudut Gradien (derajat)');
    ylabel('Frekuensi');
    title('Histogram Bins Fitur HOG');

    % Mengganti .jpg dengan string kosong dalam pembuatan nama file
    saveas(gcf, fullfile(outputDir, [strrep(jpgFiles(i).name, '.jpg', '') '_HOG_hist.jpg']));
    close(gcf);
    
    % Menambahkan data fitur HOG ke dalam tabel
    hogTable(i + 1, :) = [{jpgFiles(i).name}, num2cell(hogHistogram)];
end

% Menyimpan tabel ke file CSV
cell2csv(fullfile(outputDir, 'HOG_features.csv'), hogTable);

% Fungsi untuk menyimpan cell array ke file CSV
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
