package thesis.selection;

public class InformationGainRanker extends AbstractFrequencyRanker {

    private static final double LOG2 = Math.log(2.0);

    public InformationGainRanker(int numFeatures, int numBins, int numClasses) {
        super(numFeatures, numBins, numClasses);
    }

    @Override
    protected double score(int featureIdx) {
        long n = featureTotal(featureIdx);
        if (n == 0) return 0.0;
        long[] cm = classMarginal(featureIdx);

        double hY = 0.0;
        for (int c = 0; c < numClasses; c++) {
            if (cm[c] > 0) {
                double p = cm[c] / (double) n;
                hY -= p * Math.log(p) / LOG2;
            }
        }

        double hYgivenX = 0.0;
        for (int b = 0; b < numBins; b++) {
            int nb = featureBinTotals[featureIdx][b];
            if (nb == 0) continue;
            double pb = nb / (double) n;
            double hYb = 0.0;
            for (int c = 0; c < numClasses; c++) {
                int o = joint[featureIdx][b][c];
                if (o == 0) continue;
                double pc = o / (double) nb;
                hYb -= pc * Math.log(pc) / LOG2;
            }
            hYgivenX += pb * hYb;
        }

        return Math.max(0.0, hY - hYgivenX);
    }
}