% Tentukan direktori input dan output
inputDir = 'D:\SIKIL\Dataset Gray Flat';
outputDirHist = 'D:\SIKIL\';
outputDirBoF = 'D:\SIKIL\BOF Flat';

% Buat direktori output jika belum ada
if ~exist(outputDirHist, 'dir')
    mkdir(outputDirHist);
end

if ~exist(outputDirBoF, 'dir')
    mkdir(outputDirBoF);
end

jpgFiles = dir(fullfile(inputDir, '*.jpg'));

for i = 1:length(jpgFiles)
    % Konversi ke skala abu-abu
    img = imread(fullfile(inputDir, jpgFiles(i).name));
    
    % Pastikan gambar dalam format grayscale
    if size(img, 3) == 3
        I_gray = rgb2gray(img); 
    else
        I_gray = img;
    end
   
    % Ekstraksi fitur HOG
    cellSize = [8 8];
    numBins = 60;
    [hogFeature, visualization] = extractHOGFeatures(I_gray, 'CellSize', cellSize, 'NumBins', numBins);

    % Visualisasi HOG
    figure;
    subplot(1, 2, 1);
    imshow(I_gray); hold on;
    plot(visualization);
    title('Visualisasi Fitur HOG');

    % Membuat histogram dari fitur HOG
    binEdges = linspace(0, 180, numBins+1); % Rentang sudut dari 0 hingga 180 derajat
    hogFeatureReshaped = reshape(hogFeature, [], numBins); % Bentuk ulang fitur HOG menjadi matriks
    hogHistogram = sum(hogFeatureReshaped, 1); % Jumlahkan gradien di setiap bin

    % Konversi histogram menjadi double untuk findpeaks
    hogHistogram = double(hogHistogram);

    % Visualisasi histogram
    subplot(1, 2, 2);
    bar(binEdges(1:end-1), hogHistogram, 'histc');
    xlim([0 180]);
    xlabel('Sudut Gradien (derajat)');
    ylabel('Jumlah Gradien');
    title('Histogram Orientasi Gradien');

    % Mengganti .jpg dengan string kosong dalam pembuatan nama file
    saveas(gcf, fullfile(outputDirHist, [strrep(jpgFiles(i).name, '.jpg', '') 'HOG_hist.jpg']));
    close(gcf);
    
    % Menghitung kelengkungan (curvature)
    curvature = diff(hogHistogram, 2); % Turunan kedua dari histogram

    % Mencari puncak dari histogram
    [~, peakIndices] = findpeaks(hogHistogram);

    % Menghitung jarak antar puncak
    peakDistances = diff(peakIndices);

    % Menghitung rata-rata kelengkungan dan jarak antar puncak
    meanCurvature = mean(abs(curvature));
    meanPeakDistance = mean(peakDistances);

    % Membuat histogram BoF dengan dua sumbu X
    BoFHistogram = [meanCurvature, meanPeakDistance];

    % Visualisasi histogram BoF
    figure;
    bar(BoFHistogram);
    xticklabels({'Curvature', 'Peak Distance'});
    xlabel('Fitur');
    ylabel('Nilai Rata-rata');
    title('Histogram BoF (Curvature dan Jarak Antar Puncak)');

    % Menyimpan grafik
    saveas(gcf, fullfile(outputDirBoF, [strrep(jpgFiles(i).name, '.jpg', '') '_BoF.jpg']));
    close(gcf);
end
