clear; clc;
% Setting
numTrees = 100; % Jumlah pohon
k = 10; % Jumlah fold untuk K-Fold cross-validation

% Import data yang dibutuhkan
dataset = "D:\SIKIL\Variabel\";
csvFiles = dir(fullfile(dataset, '*.csv'));
load(fullfile('D:\SIKIL\Variabel\', 'data.mat'));

for i = 1:length(csvFiles)
    % Membuat path lengkap untuk file CSV
    filePath = fullfile(dataset, csvFiles(i).name);
    
    % Mendeteksi opsi impor
    opts = detectImportOptions(filePath);
    opts.VariableNamingRule = 'preserve';
    
    % Membaca tabel data
    data = readtable(filePath, opts);

    % Memisahkan fitur dan label
    if strcmp(strtok(csvFiles(i).name, ' '), "HOG")
        features = data(:, 1:numBins);
        Kategori = "Histogram of Oriented Gradients";
    elseif strcmp(strtok(csvFiles(i).name, ' '), "BOF")
        features = data(:, 1:5);
        Kategori = "Bag of Features";
    end
    labels = data.Label;

    % Membuat objek pembagian data untuk K-Fold cross-validation
    cv = cvpartition(labels, 'KFold', k, 'Stratify', true);
    
    % Inisialisasi metrik kinerja
    akurasi_sum = zeros(k, 1);
    presisi1_sum = zeros(k, 1);
    presisi2_sum = zeros(k, 1);
    recall1_sum = zeros(k, 1);
    recall2_sum = zeros(k, 1);

    for j = 1:k
        % Mendapatkan indeks data pelatihan dan pengujian untuk fold ke-j
        trainIdx = training(cv, j);
        testIdx = test(cv, j);
        trainFeatures = features(trainIdx, :);
        trainLabels = labels(trainIdx);
        testFeatures = features(testIdx, :);
        testLabels = labels(testIdx);

        % Melatih model Random Forest
        model = TreeBagger(numTrees, trainFeatures, trainLabels, 'Method', 'classification', OOBPredictorImportance='on');

        % Prediksi data testing
        predictedLabels = predict(model, testFeatures);

        % Membuat confusion matrix
        confMatrix = confusionmat(testLabels, predictedLabels);

        % Menghitung Evaluasi Model Random Forest
        tp = confMatrix(1,1); % True Positive
        tn = confMatrix(2,2); % True Negative
        fp = confMatrix(2,1); % False Positive
        fn = confMatrix(1,2); % False Negative
        akurasi = ((tp + tn) / (tp + tn + fp + fn)) * 100;

        % Simpan hasil evaluasi untuk fold ini
        tp_sum(j) = tp;
        tn_sum(j) = tn;
        fp_sum(j) = fp;
        fn_sum(j) = fn;
        akurasi_sum(j) = akurasi;
    end

    % Menghitung rata-rata metrik kinerja
    tp_mean = round(mean(tp_sum));
    tn_mean = round(mean(tn_sum));
    fp_mean = round(mean(fp_sum));
    fn_mean = round(mean(fn_sum));
    precision = (tp_mean / (tp_mean + fp_mean)) * 100;
    precision2 = (tn_mean / (tn_mean + fn_mean)) * 100;
    recall = (tp_mean / (tp_mean + fn_mean)) * 100;
    recall2 = (tn_mean / (tn_mean + fp_mean)) * 100;
    meanAccuracy = mean(akurasi);

    % Menampilkan hasil evaluasi confusion matrix rata-rata
    fprintf('Confusion Matrix Table\n');
    fprintf("Random Forest - " + Kategori + "\n" + "K-Fold = "+ k +"\n");
    fprintf('-----------------------------------------------------------\n');
    fprintf(['Akurasi: ', num2str(meanAccuracy),'%%']);
    fprintf('\n\t\t\t\tTrue Flat\tTrue Normal\tClass Precision\n');
    fprintf('Pred. Flat\t\t%d\t\t\t%d\t\t\t%.2f%%\n', tp_mean, fp_mean, precision);
    fprintf('Pred. Normal\t%d\t\t\t%d\t\t\t%.2f%%\n', fn_mean, tn_mean, precision2);
    fprintf('Class Recall\t%.2f%%\t\t%.2f%%\n', recall, recall2);
    fprintf('-----------------------------------------------------------\n');
    fprintf('\n\n');

    % Menghitung feature importance
    featureImportance = model.OOBPermutedVarDeltaError;
    % Menampilkan feature importance dalam bentuk histogram
    figure;
    bar(featureImportance);
    if strcmp(strtok(csvFiles(i).name, ' '), "BOF")
        xticklabels({'Curvature', 'Luas', 'Kepadatan', 'Bin Tertinggi', 'Posisi Bin Tertinggi'});
    end
    xlabel('Fitur');
    ylabel('Feature Importance');
    title('Feature Importance  ' + Kategori);
end
