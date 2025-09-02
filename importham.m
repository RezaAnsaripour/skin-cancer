function HAM1 = importham(filename, dataLines)
%IMPORTFILE Import data from a text file
%  HAM1 = IMPORTFILE(FILENAME) reads data from text file FILENAME for
%  the default selection.  Returns the data as a table.
%
%  HAM1 = IMPORTFILE(FILE, DATALINES) reads data for the specified row
%  interval(s) of text file FILENAME. Specify DATALINES as a positive
%  scalar integer or a N-by-2 array of positive scalar integers for
%  dis-contiguous row intervals.
%
%  Example:
%  HAM1 = importfile("F:Code\HAM10000", [2, Inf]);
%

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [2, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 8);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization", "dataset"];
opts.VariableTypes = ["double", "double", "categorical", "categorical", "double", "categorical", "categorical", "categorical"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["dx", "dx_type", "sex", "localization", "dataset"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, ["lesion_id", "image_id"], "TrimNonNumeric", true);
opts = setvaropts(opts, ["lesion_id", "image_id"], "ThousandsSeparator", ",");

% Import the data
HAM1 = readtable(filename, opts);

end