package thesis.selection;

public class InformationGainRanker extends AbstractFrequencyRanker {

    private static final double LOG2 = Math.log(2.0);

    public InformationGainRanker(int numFeatures, int numBins, int numClasses) {
        super(numFeatures, numBins, numClasses);
    }

    public InformationGainRanker(int numFeatures, int numBins, int numClasses, int minSamplesReady) {
        super(numFeatures, numBins, numClasses, minSamplesReady);
    }

    @Override
    protected double score(int featureIdx) {
        double n = featureTotal(featureIdx);
        if (n <= 0.0) return 0.0;
        double[] cm = classMarginal(featureIdx);

        double hY = 0.0;
        double invN = 1.0 / n;
        for (int c = 0; c < numClasses; c++) {
            double ck = cm[c];
            if (ck > 0.0) {
                double p = ck * invN;
                hY -= p * Math.log(p);
            }
        }

        double hYgivenX = 0.0;
        for (int b = 0; b < numBins; b++) {
            double nb = featureBinTotals[featureIdx][b];
            if (nb <= 0.0) continue;
            double pb = nb * invN;
            double invNb = 1.0 / nb;
            double hYb = 0.0;
            for (int c = 0; c < numClasses; c++) {
                double o = joint[featureIdx][b][c];
                if (o <= 0.0) continue;
                double pc = o * invNb;
                hYb -= pc * Math.log(pc);
            }
            hYgivenX += pb * hYb;
        }

        double ig = (hY - hYgivenX) / LOG2;
        return ig > 0.0 ? ig : 0.0;
    }
}