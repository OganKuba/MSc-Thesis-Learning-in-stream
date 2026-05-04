package thesis.selection;

public class MutualInformationRanker extends AbstractFrequencyRanker {

    private static final double LOG2 = Math.log(2.0);

    public MutualInformationRanker(int numFeatures, int numBins, int numClasses) {
        super(numFeatures, numBins, numClasses);
    }

    public MutualInformationRanker(int numFeatures, int numBins, int numClasses, int minSamplesReady) {
        super(numFeatures, numBins, numClasses, minSamplesReady);
    }

    @Override
    protected double score(int featureIdx) {
        double n = featureTotal(featureIdx);
        if (n <= 0.0) return 0.0;
        double[] cm = classMarginal(featureIdx);

        double mi = 0.0;
        for (int b = 0; b < numBins; b++) {
            double nb = featureBinTotals[featureIdx][b];
            if (nb <= 0.0) continue;
            for (int c = 0; c < numClasses; c++) {
                double o = joint[featureIdx][b][c];
                if (o <= 0.0 || cm[c] <= 0.0) continue;
                double ratio = (o * n) / (nb * cm[c]);
                if (ratio <= 0.0) continue;
                mi += (o / n) * Math.log(ratio);
            }
        }
        double miBits = mi / LOG2;
        return miBits > 0.0 ? miBits : 0.0;
    }
}