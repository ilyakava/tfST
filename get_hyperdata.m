function [Y, X] = get_hyperdata(dataset_path, Yname, Xname)
% function for loading hyperspectral data

Yfile = sprintf('%s.mat', Yname);
Ypath = fullfile(dataset_path, Yfile);
Ystruct = load(Ypath);
Yfieldnames = fieldnames(Ystruct);
if length(Yfieldnames) > 1;
    warning('more than one field in loaded Y file, loading first...')
end
Y = getfield(Ystruct, Yfieldnames{1});

if nargin > 2
	Xfile = sprintf('%s.mat', Xname);
	Xpath = fullfile(dataset_path, Xfile);
	Xstruct = load(Xpath);
	Xfieldnames = fieldnames(Xstruct);
	if length(Xfieldnames) > 1;
	    warning('more than one field in loaded X file, loading first...')
	end
	X = getfield(Xstruct, Xfieldnames{1});
else
	X = -1;
end
