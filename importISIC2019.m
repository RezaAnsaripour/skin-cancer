function ISIC2019 = importISIC2019(filename, dataLines)
%IMPORTFILE Import data from a text file
%  ISIC2019 = IMPORTFILE(FILENAME) reads data from text file FILENAME
%  for the default selection.  Returns the data as a table.
%
%  ISIC2019 = IMPORTFILE(FILE, DATALINES) reads data for the specified
%  row interval(s) of text file FILENAME. Specify DATALINES as a
%  positive scalar integer or a N-by-2 array of positive scalar integers
%  for dis-contiguous row intervals.
%
%  Example:
%  ISIC2019 = importfile("ISIC_2019.csv", [2, Inf]);


%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [2, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 10);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["image", "MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "image", "TrimNonNumeric", true);
opts = setvaropts(opts, "image", "ThousandsSeparator", ",");

% Import the data
ISIC2019 = readtable(filename, opts);

end