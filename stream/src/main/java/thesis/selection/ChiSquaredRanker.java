package thesis.selection;

public class ChiSquaredRanker extends AbstractFrequencyRanker {

    public ChiSquaredRanker(int numFeatures, int numBins, int numClasses) {
        super(numFeatures, numBins, numClasses);
    }

    @Override
    protected double score(int featureIdx) {
        long n = featureTotal(featureIdx);
        if (n == 0) return 0.0;
        long[] cm = classMarginal(featureIdx);

        double chi2 = 0.0;
        for (int b = 0; b < numBins; b++) {
            int nb = featureBinTotals[featureIdx][b];
            if (nb == 0) continue;
            for (int c = 0; c < numClasses; c++) {
                if (cm[c] == 0) continue;
                double expected = (nb * (double) cm[c]) / n;
                if (expected <= 0.0) continue;
                double observed = joint[featureIdx][b][c];
                double diff = observed - expected;
                chi2 += (diff * diff) / expected;
            }
        }
        return chi2;
    }
}