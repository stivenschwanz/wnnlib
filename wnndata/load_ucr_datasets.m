ucr_data_path = 'C:\Users\20214358\Workspace\UCR_TimeSeriesAnomalyDatasets2021\AnomalyDatasets_2021\UCR_TimeSeriesAnomalyDatasets2021\FilesAreInHere\UCR_Anomaly_FullData\';

ucr_data_output = './ucr_time_series_anomaly_datasets_2021.mat';

ucr_data_files = dir(fullfile(ucr_data_path,'*.txt'));

ucr_data_number_of_files = length(ucr_data_files);

ucr_datasets = cell(ucr_data_number_of_files, 1);

for k = 1:ucr_data_number_of_files
  ucr_dataset_file_name = ucr_data_files(k).name;

  expression = ['(?<dataset_index>\d+)_UCR_Anomaly_'...
                '(?<dataset_mmemonic>\w+)_(?<training_end>\d+)_'...
                '(?<anomaly_begin>\d+)_(?<anomaly_end>\d+).txt'];

   tokens = regexp(ucr_dataset_file_name, expression, 'names');

   dataset.samples = load(strcat(ucr_data_path,ucr_dataset_file_name));

   dataset.index = str2double(tokens.dataset_index);
   dataset.mmemonic = tokens.dataset_mmemonic;
   dataset.training_begin = 1;
   dataset.training_end = str2double(tokens.training_end);
   dataset.testing_begin = dataset.training_end + 1;
   dataset.testing_end = length(dataset.samples);
   dataset.anomaly_begin = str2double(tokens.anomaly_begin);
   dataset.anomaly_end = str2double(tokens.anomaly_end);
   dataset.anomaly_center = (dataset.anomaly_begin + dataset.anomaly_end)/2;

   dataset.training_statistics.mean = mean(dataset.samples(1:dataset.training_end));
   dataset.training_statistics.max = max(dataset.samples(1:dataset.training_end));
   dataset.training_statistics.min = min(dataset.samples(1:dataset.training_end));
   dataset.training_statistics.std = std(dataset.samples(1:dataset.training_end));

   dataset.norm_samples = (dataset.samples - dataset.training_statistics.mean)/dataset.training_statistics.std;

%    plot(dataset.norm_samples);
% 
%    clf;
%    hold on
%    plot(dataset.samples(dataset.anomaly_begin:dataset.anomaly_end));
%    plot(dataset.samples(dataset.anomaly_begin-1000:dataset.anomaly_end-1000));

   ucr_datasets{k} = dataset;
end


save(ucr_data_output, 'ucr_datasets', '-nocompression') 

