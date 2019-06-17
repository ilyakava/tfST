function picks = sample_by_label(labels, samples_per_label)

picks = zeros(length(labels),1);

for i=1:max(unique(labels)); % we're skipping 0, which does exist in the hyperspectral data
    idxs = find(labels == i);
    chosen_idxs = idxs(randsample(length(idxs),samples_per_label(i)));
    picks(chosen_idxs) = 1;
end
