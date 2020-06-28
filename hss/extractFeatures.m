function [signalsFsstTrain, signalsFsstVal, signalsFsstTest] = extractFeatures(frames, Fs)
    trainSignals = frames.train(:,1);
    signalsFsstTrain = cell(size(trainSignals));
    meanTrain = cell(1,length(trainSignals));
    stdTrain = cell(1,length(trainSignals));
    for idx = 1:length(trainSignals)
       disp("Feature extraction (training): #" + idx)
       [s,f,~] = fsst(trainSignals{idx},Fs,kaiser(128));
       f_indices = (f > 25) & (f < 200);
       signalsFsstTrain{idx} = [real(s(f_indices,:)); imag(s(f_indices,:))];

       meanTrain{idx} = mean(signalsFsstTrain{idx},2);
       stdTrain{idx} = std(signalsFsstTrain{idx},[],2);
    end

    standardizeFun = @(x) (x - mean(cell2mat(meanTrain),2))./mean(cell2mat(stdTrain),2);
    signalsFsstTrain = cellfun(standardizeFun,signalsFsstTrain,'UniformOutput',false);
    
    valSignals = frames.validation(:,1);
    signalsFsstVal = cell(size(valSignals));
    for idx = 1:length(valSignals)
       disp("Feature extraction (validation): #" + idx)
       [s,f,~] = fsst(valSignals{idx},Fs,kaiser(128));
       f_indices =  (f > 25) & (f < 200);
       signalsFsstVal{idx}= [real(s(f_indices,:)); imag(s(f_indices,:))]; 
    end

    signalsFsstVal = cellfun(standardizeFun,signalsFsstVal,'UniformOutput',false);
    
    testSignals = frames.test(:,1);

    signalsFsstTest = cell(size(testSignals));
    for idx = 1:length(testSignals)
       disp("Feature extraction (evaluation): #" + idx)
       [s,f,~] = fsst(testSignals{idx},Fs,kaiser(128));
       f_indices =  (f > 25) & (f < 200);
       signalsFsstTest{idx}= [real(s(f_indices,:)); imag(s(f_indices,:))]; 
    end

    signalsFsstTest = cellfun(standardizeFun,signalsFsstTest,'UniformOutput',false);
end
