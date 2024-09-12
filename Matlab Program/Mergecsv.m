function Mergecsv()
    % Path ke file CSV
    HOG1 = 'D:\SIKIL\HOG Flat\HOG Flat features.csv';
    HOG2 = 'D:\SIKIL\HOG Normal\HOG Normal features.csv';
    BOF1 = 'D:\SIKIL\BOF Flat\BOF Flat features.csv';
    BOF2 = 'D:\SIKIL\BOF Normal\BOF Normal features.csv';

    % Membaca file CSV tanpa memperhatikan header
    tableHOG1 = readtable(HOG1);
    tableHOG2 = readtable(HOG2, 'HeaderLines', 0);
    tableBOF1 = readtable(BOF1);
    tableBOF2 = readtable(BOF2, 'HeaderLines', 0);

    % Menggabungkan kedua tabel
    combinedTabHOG = [tableHOG1; tableHOG2];
    combinedTabBOF = [tableBOF1; tableBOF2];
    
    % Menentukan path lengkap untuk file output
    pathHOG = fullfile('D:\SIKIL\', 'HOG Input.csv');
    pathBOF = fullfile('D:\SIKIL\', 'BOF Input.csv');

    % Menyimpan tabel gabungan ke file CSV
    writetable(combinedTabHOG, pathHOG);
    writetable(combinedTabBOF, pathBOF);
    clc;
end
