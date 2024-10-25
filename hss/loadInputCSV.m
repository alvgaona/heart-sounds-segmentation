function [signals, labels] = loadInputCSV()
    % Initialize cell arrays to store data
    signals = {};
    labels = {};
    
    % Define the path where your CSV files are stored
    folderPath = 'resources/data/springer_sounds/'; % Update this with your actual path
    
    % Loop through files from 0001 to 0792
    for i = 1:792
        % Create filename with proper formatting (4 digits with leading zeros)
        filename = sprintf('%s%04d.csv', folderPath, i);
        
        try
            % Read CSV file
            data = readtable(filename);
            
            % Extract signals and labels
            signals{i} = data.Signals'; % Transpose to match your original format
            labels{i} = categorical(data.Labels);   % Transpose to match your original format
            
        catch ME
            warning('Problem reading file %s: %s', filename, ME.message);
            continue;
        end
    end
    
    % Make sure all cell arrays are column vectors
    signals = signals(:);
    
    labels = labels(:);
end