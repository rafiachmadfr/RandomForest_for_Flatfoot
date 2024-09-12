clear; clc;
% Setting
numTrees = 150; % Jumlah pohon
testdata = 10; % Persen data testing

% Merge dan improt data yang dibutuhkan
%Mergecsv();
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
    if strtok(csvFiles(i).name, ' ') == "HOG"
        features = data(:, 1:numBins);
        Kategori = "Histogram of Oriented Gradients";
    elseif strtok(csvFiles(i).name, ' ') == "BOF"
        features = data(:, 1:fiturBOF);
        Kategori = "Bag of Features";
    end
    labels = data.Label;

    % Pembagian data training dan testing
    if strtok(csvFiles(i).name, ' ') == "BOF"
       cv = cvpartition(labels, 'HoldOut', testdata/100, 'Stratify', true);
    end
    trainIdx = training(cv);
    testIdx = test(cv);
    trainFeatures = features(trainIdx, :);
    trainLabels = labels(trainIdx);
    testFeatures = features(testIdx, :);
    testLabels = labels(testIdx);

    % Melatih model Random Forest
    model = TreeBagger(numTrees, trainFeatures, trainLabels, ...
        'Method', 'classification', 'OOBPredictorImportance', 'on');

    % Prediksi data testing
    predictedLabels = predict(model, testFeatures);

    % Membuat confusion matrix
    confMatrix = confusionmat(testLabels, predictedLabels);

    % Menghitung Evaluasi Model Random Forest
    tp = confMatrix(1,1); % True Positive
    tn = confMatrix(2,2); % True Negative
    fp = confMatrix(2,1); % Type Error 1
    fn = confMatrix(1,2); % Type Error 2
    precision  = (tp / (tp + fp))*100;
    precision2 = (tn / (tn + fn))*100;
    recall  = (tp / (tp + fn))*100;
    recall2 = (tn / (tn + fp)) * 100;
    akurasi=((tp+tn)/(tp+tn+fp+fn))*100;

    % Menampilkan hasil evaluasi confusion matriks
    fprintf('Confusion Matrix Table\n');
    fprintf("Random Forest - " + Kategori +"\n");
    fprintf('-----------------------------------------------------------\n');
    fprintf(['Akurasi: ', num2str(akurasi),'%%']);
    fprintf('\n\t\t\t\tTrue Flat\tTrue Normal\tClass Precision\n');
    fprintf('Pred. Flat\t\t%d\t\t\t%d\t\t\t%.2f%%\n', tp, fp, precision);
    fprintf('Pred. Normal\t%d\t\t\t%d\t\t\t%.2f%%\n', fn, tn, precision2);
    fprintf('Class Recall\t%.2f%%\t\t%.2f%%\n', recall, recall2);
    fprintf('-----------------------------------------------------------\n');
    fprintf('\n\n');

    % Menghitung feature importance
    featureImportance = model.OOBPermutedVarDeltaError;
    % Menampilkan feature importance dalam bentuk histogram
    figure;
    bar(featureImportance);
    xlabel('Fitur');
    ylabel('Feature Importance');
    title('Feature Importance  ' + Kategori);
end
