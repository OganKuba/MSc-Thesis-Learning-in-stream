package thesis.selection;

public class MutualInformationRanker extends AbstractFrequencyRanker {

    public MutualInformationRanker(int numFeatures, int numBins, int numClasses) {
        super(numFeatures, numBins, numClasses);
    }

    @Override
    protected double score(int featureIdx) {
        long n = featureTotal(featureIdx);
        if (n == 0) return 0.0;
        long[] cm = classMarginal(featureIdx);

        double mi = 0.0;
        for (int b = 0; b < numBins; b++) {
            int nb = featureBinTotals[featureIdx][b];
            if (nb == 0) continue;
            for (int c = 0; c < numClasses; c++) {
                int o = joint[featureIdx][b][c];
                if (o == 0 || cm[c] == 0) continue;
                double pxy = o / (double) n;
                double px = nb / (double) n;
                double py = cm[c] / (double) n;
                mi += pxy * Math.log(pxy / (px * py));
            }
        }
        return Math.max(0.0, mi);
    }
}