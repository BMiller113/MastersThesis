function plotROC(net, XTest, YTest)
    scores = predict(net, XTest);
    [X, Y, T, AUC] = perfcurve(YTest, scores(:, 1), 'positiveClass');
    figure;
    plot(X, Y);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(['ROC Curve, AUC = ', num2str(AUC)]);
end