% script to create training files used in hyper_svm_experiments

% 3 variables for the user to set
OUT_PATH = '.';
DATASET_PATH = '/scratch0/ilya/locDoc/data/hyperspec/datasets'
ntrials = 10;

% init
groundtruthfiles = { 'Indian_pines_gt', 'PaviaU_gt', 'Pavia_gt', 'Botswana_gt', ...
    'KSC_gt', 'Pavia_center_right_gt', 'Salinas_gt', 'Smith_gt', ...
    'PaviaU_gt'};
datasetsamples = cell(length(groundtruthfiles), 1);

% the user may set the number of samples from each class to get directly, for example:
datasetsamples{1} = [5, 143, 72, 23, 45, 75, 3, 49, 2, 94, 238, 59, 21, 129, 28, 10];
datasetsamples{1} = [5 143 83 23 50 75 3 49 2 97 247 61 21 129 38 10]; % ma
datasetsamples{2} = [658,1828,208,305,135,503,133,368,95];
datasetsamples{2} = [597 1681 189 276 121 453 120 331 85]; % ma
datasetsamples{2} = [200 200 200 200 200 200 200 200 200];
datasetsamples{4} = [14,6,13,11,14,14,13,11,16,13,16,10,14,5];
datasetsamples{5} = [39,13,13,13,9,12,6,22,26,21,21,26,47];
% datasetsamples{6} = [30 30 30 30 30 30 30 30 30];
datasetsamples{7} = [20 37 20 14 27 40 36 113 62 33 11 19 9 11 73 18]; % ma
datasetsamples{9} = floor([50, 50, 50, 50, 50, 50, 50, 50, 50]);

% the user may set the percentage of samples from each class to get directly, for example:
datasetsamples{1} = max(1,floor([46, 1428, 830, 237, 483, 730, 28, 478, 20, 972, 2455, 593, 205, 1265, 386, 93] .* 0.01)); % he
datasetsamples{1} = floor([46, 1428, 830, 237, 483, 730, 28, 478, 20, 972, 2455, 593, 205, 1265, 386, 93] .* 0.01);
datasetsamples{2} = floor([6631,18649,2099,3064,1345,5029,1330,3682,947] .* 0.06);
datasetsamples{3} = floor([65971 7598 3090 2685 6584 9248 7287 42826 2863] .* 0.01);
datasetsamples{6} = floor([65278 6508 2905 2140 6549 7585 7287  3122 2165] .* 0.01);
datasetsamples{7} = floor([2009 3726 1976 1394 2678 3959 3579 11271 6203 3278 1068 1927 916 1070 7268 1807] .* 0.01);
datasetsamples{8} = max(1,floor([196 246 184 66 97 57 32 70 200 90 76 58 166 328 105 159 144 167 18 44 206 34] .* 0.05));
datasetsamples{9} = floor([6631, 18649, 2099, 3064, 1345, 5029, 1330, 3682, 947] .* 0.01);




for datasetidx=1;
    sample = datasetsamples{datasetidx};
    Y = get_hyperdata(DATASET_PATH, groundtruthfiles{datasetidx});
    labels = reshape(Y,[1 prod(size(Y))]);

    outfilenames = cell(length(ntrials),1);
    for trial=1:ntrials;
        train_mask = sample_by_label(labels, sample);
        test_mask = ~train_mask & ~~labels(:); % we are skipping the 0 labels

        id = DataHash([train_mask test_mask]);
        outfilenames{trial} = sprintf('%s_traintest_p01_nozero_%d_%s.mat', groundtruthfiles{datasetidx}, trial, id(end-5:end));
        outfile = fullfile(OUT_PATH, outfilenames{trial});
        save(outfile, 'train_mask', 'test_mask');
    end
    fprintf('traintestfilenames = [ ''%s'' ];\n', strjoin(outfilenames, ''', '''));
end

