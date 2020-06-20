function multiClassRocCurve(labels,scores,leg)
    figure
    grid on
    hold on
    AUCs = zeros(1,size(scores,1));
    for i=1:size(scores,1)
        [X,Y,~,AUC] = perfcurve(labels,scores(i,:), int2str(i));
        fprintf("Class %d AUC: %.3f\n", i, AUC)
        AUCs(i) = AUC;
        plot(X,Y)
    end
    title('Receiver Operation Characteristic Curve')
    xlabel('False positive rate') 
    ylabel('True positive rate')
    legend(leg)

    fprintf("Average AUC: %.3f\n", mean(AUCs))
end