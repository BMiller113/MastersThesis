function plotGenderROC(rocStructs, labels)
    figure;
    hold on;
    for i = 1:length(rocStructs)
        far = rocStructs{i}.far;
        frr = rocStructs{i}.frr;
        if isempty(far) || isempty(frr)
            continue;
        end
        semilogx(far*3600/0.01, frr*100, 'LineWidth', 2);
    end
    xlabel('False Alarms per Hour');
    ylabel('False Reject Rate (%)');
    title('ROC Curves by Gender Filter');
    legend(labels, 'Location', 'best');
    grid on;
end
