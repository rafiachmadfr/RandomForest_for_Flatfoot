% Setting 
cellSize    = [8 8];
numBins     = 60;
fiturBOF    = 5;

% Tentukan direktori input dan output
inputDir  = "D:\SIKIL\DatasetFull\";
outputHOG = "D:\SIKIL\HOG";
outputBOF = "D:\SIKIL\BOF";
outputcsv = "D:\SIKIL\Variabel\";

% Buat direktori output jika belum ada
if ~exist(outputHOG, 'dir')
    mkdir(outputHOG);
elseif ~exist(outputBOF, 'dir')
    mkdir(outputBOF);
end

jpgFiles = dir(fullfile(inputDir, '*.jpg'));

% Inisialisasi tabel untuk menyimpan fitur HOG
hogTable = cell(length(jpgFiles) + 1, numBins+1); % Header csv
hogTable(1,:) = [arrayfun(@(x) sprintf('Bin %d', x), 1:numBins, 'UniformOutput', false), {'Label'}];
bofTable = cell(length(jpgFiles) + 1, fiturBOF+1); % Header csv
bofTable(1,:) = [{'Curvature'}, {'Luas'}, {'Kepadatan'}, {'Bin Tertinggi'}, {'Posisi Bin Tertinggi'}, {'Label'}];

for i = 1:length(jpgFiles)
    fprintf(jpgFiles(i).name + "\n");

    img = imread(fullfile(inputDir, jpgFiles(i).name));
    Img_gray = im2gray(img);
    Img_resized = imresize(Img_gray, [128 64]);
   
    %% HOG
    % Ekstraksi fitur HOG
    [hogFeature, visualization] = extractHOGFeatures(Img_resized, 'CellSize', cellSize,'NumBins',numBins);

    % Visualisasi HOG
    HisHOG = figure('Visible', "off");
    set(gcf, 'Position', [10, 10, 1000, 500]);
    subplot(1, 2, 1);
    set(gca, 'Position', [0.01, 0.1, 0.25, 0.8]);
    imshow(Img_resized); hold on;
    plot(visualization);
    title('Visualisasi Fitur HOG');

    % Membuat histogram dari fitur HOG
    binEdges = linspace(0, 180, numBins+1);
    hogFeatureReshaped = reshape(hogFeature, [], numBins);
    hogHistogram = sum(hogFeatureReshaped, 1);

    % Visualisasi histogram
    subplot(1, 2, 2);
    set(gca, 'Position', [0.35, 0.1, 0.6, 0.8]);
    bar(binEdges(1:end-1), hogHistogram, 'histc');
    xlim([0 180]);
    xlabel('Sudut Gradien (derajat)');
    ylabel('Frekuensi');
    title('Histogram Bins Fitur HOG');
    
    %% BOF
    % Menghitung kelengkungan (curvature)
    curvature = diff(hogHistogram, 2); % Turunan kedua dari histogram
    meanCurvature = mean(abs(curvature));

    % Mencari puncak dari histogram
    [~, peakIndices] = findpeaks(hogHistogram);
    peakDistances = diff(peakIndices);
    meanPeakDistance = mean(peakDistances);
    
    % Luas kepadatan
    areaFootprint = sum(hogHistogram); % Luas dari histogram HOG
    perimeterFootprint = sum(abs(diff(hogHistogram))); % Keliling dari histogram HOG
    compactnessFootprint = areaFootprint / (perimeterFootprint^2); % Kompak
    
    % Bin tertinggi dan Nilai bin tertinggi berada
    [maxBinValue, maxBinIndex] = max(hogHistogram);
    maxBinPosition = binEdges(maxBinIndex);

    % Membuat histogram BoF dengan tambahan fitur geometris
    bofHistogram = [meanCurvature, areaFootprint, compactnessFootprint, maxBinValue, maxBinPosition];

    % Visualisasi histogram BoF
    HisBOF = figure('Visible', 'off');
    bar(bofHistogram);
    xticklabels({'Curvature', 'Luas', 'Kepadatan', 'Bin Tertinggi', 'Posisi Bin Tertinggi'});
    xlabel('Fitur');
    ylabel('Nilai');
    title('Histogram BoF');

    %% Save File jpg dan csv
    % Menyimpan grafik
    saveas(HisHOG, fullfile(outputHOG, [strrep(jpgFiles(i).name, '.jpg', '') '_HOG.jpg']));
    %close(gcf);
    saveas(HisBOF, fullfile(outputBOF, [strrep(jpgFiles(i).name, '.jpg', '') '_BOF.jpg']));
    %close(gcf);

    % Menambahkan data fitur ke dalam tabel csv
    hogTable(i + 1, :) = [num2cell(hogHistogram), strtok(jpgFiles(i).name, ' ')];
    bofTable(i + 1, :) = [num2cell(bofHistogram), strtok(jpgFiles(i).name, ' ')];

end

% Menyimpan tabel ke file CSV
cell2csv(fullfile(outputcsv, "HOG features" + ".csv"), hogTable);
cell2csv(fullfile(outputcsv, "BOF features" + ".csv"), bofTable);

% Simpan variabel penting untuk random forest
save(fullfile('D:\SIKIL\Variabel\', 'data.mat'), 'fiturBOF', 'numBins');

clc;fprintf('Alhamdulillahirabbilalamin\n');
