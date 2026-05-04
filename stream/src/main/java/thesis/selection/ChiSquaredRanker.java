package thesis.selection;

public class ChiSquaredRanker extends AbstractFrequencyRanker {

    private final boolean useCramerV;

    public ChiSquaredRanker(int numFeatures, int numBins, int numClasses) {
        this(numFeatures, numBins, numClasses, 50, true);
    }

    public ChiSquaredRanker(int numFeatures, int numBins, int numClasses,
                            int minSamplesReady, boolean useCramerV) {
        super(numFeatures, numBins, numClasses, minSamplesReady);
        this.useCramerV = useCramerV;
    }

    @Override
    protected double score(int featureIdx) {
        double n = featureTotal(featureIdx);
        if (n < minSamplesReady) return 0.0;
        double[] cm = classMarginal(featureIdx);

        double chi2 = 0.0;
        double invN = 1.0 / n;
        int nonEmptyBins = 0;
        for (int b = 0; b < numBins; b++) {
            double nb = featureBinTotals[featureIdx][b];
            if (nb <= 0.0) continue;
            nonEmptyBins++;
            for (int c = 0; c < numClasses; c++) {
                double ck = cm[c];
                if (ck <= 0.0) continue;
                double expected = nb * ck * invN;
                if (expected <= 0.0) continue;
                double observed = joint[featureIdx][b][c];
                double diff = observed - expected;
                chi2 += (diff * diff) / expected;
            }
        }
        if (nonEmptyBins < 2) return 0.0;
        if (!useCramerV) return chi2;
        int dof = Math.max(1, Math.min(nonEmptyBins, numClasses) - 1);
        double v2 = chi2 / (n * dof);
        if (v2 < 0.0) v2 = 0.0;
        if (v2 > 1.0) v2 = 1.0;
        return Math.sqrt(v2);
    }
}