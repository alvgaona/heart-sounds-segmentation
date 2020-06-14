close all

audio_data = springer_dataset.audio_data;
annotations = springer_dataset.annotations;
patient_number = springer_dataset.patient_number;
binary_diagnosis = springer_dataset.binary_diagnosis;
labels = springer_dataset.labels;
features = springer_dataset.features;

for i=1:length(audio_data)
    signal_name = sprintf("sound%03d", i);
    mkdir("data/"+signal_name);
    
    a = audio_data{i};
    l = labels{i};
    f = features{i};
    r_ann = annotations{i,1}';
    t_ann = annotations{i,2}';
    pn = patient_number{i};
    bd = binary_diagnosis{i};
    
    csvwrite(sprintf("data/"+ signal_name+ "/audio.csv", i), a);
    csvwrite(sprintf("data/"+ signal_name+ "/labels.csv", i), a);
    csvwrite(sprintf("data/"+ signal_name+ "/rann.csv", i), r_ann);
    csvwrite(sprintf("data/"+ signal_name+ "/tann.csv", i), t_ann);
    csvwrite(sprintf("data/"+ signal_name+ "/features.csv", i), f);
    
    metadata = fopen(sprintf("data/"+ signal_name+ "/metadata.txt", i),'w');
    fprintf(metadata,'%s\t%s\n', 'SIGNAL_NAME', signal_name);
    fprintf(metadata,'%s\t%d\n', 'PATIENT_NUMBER', pn);
    fprintf(metadata,'%s\t%d\n\n', 'BINARY_DIAGNOSIS', bd);
    fprintf(metadata,'##%s\n', ' Contains R-wave and T-end-wave annotations', ' Also extracted FSST features and labels');
    fclose(metadata);    
end
