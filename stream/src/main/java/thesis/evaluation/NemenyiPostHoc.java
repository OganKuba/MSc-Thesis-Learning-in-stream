package thesis.evaluation;

import java.util.ArrayList;
import java.util.List;

public class NemenyiPostHoc {

    private static final double[] Q_ALPHA_005 = {
            Double.NaN, Double.NaN,
            1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164,
            3.219, 3.268, 3.313, 3.354, 3.391, 3.426, 3.458, 3.489, 3.517, 3.544
    };

    private static final double[] Q_ALPHA_010 = {
            Double.NaN, Double.NaN,
            1.645, 2.052, 2.291, 2.460, 2.589, 2.693, 2.780, 2.855, 2.920,
            2.978, 3.030, 3.077, 3.120, 3.159, 3.196, 3.230, 3.261, 3.291, 3.319
    };

    private final double alpha;

    public NemenyiPostHoc() { this(0.05); }

    public NemenyiPostHoc(double alpha) {
        if (alpha != 0.05 && alpha != 0.10) {
            throw new IllegalArgumentException("alpha must be 0.05 or 0.10 (table-driven)");
        }
        this.alpha = alpha;
    }

    public Result test(double[] averageRanks, int numDatasets) {
        if (averageRanks == null || averageRanks.length < 2) {
            throw new IllegalArgumentException("need at least 2 methods");
        }
        int k = averageRanks.length;
        if (k >= Q_ALPHA_005.length) throw new IllegalArgumentException("k too large for built-in q-table (max " + (Q_ALPHA_005.length - 1) + ")");
        double q = (alpha == 0.05) ? Q_ALPHA_005[k] : Q_ALPHA_010[k];
        double cd = q * Math.sqrt((k * (k + 1)) / (6.0 * numDatasets));

        List<int[]> sigPairs = new ArrayList<>();
        double[][] diffs = new double[k][k];
        boolean[][] sig = new boolean[k][k];
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < k; j++) {
                double d = Math.abs(averageRanks[i] - averageRanks[j]);
                diffs[i][j] = d;
                diffs[j][i] = d;
                if (d > cd) {
                    sig[i][j] = true;
                    sig[j][i] = true;
                    sigPairs.add(new int[]{i, j});
                }
            }
        }
        return new Result(k, numDatasets, alpha, q, cd, averageRanks.clone(), diffs, sig, sigPairs);
    }

    public static final class Result {
        public final int numMethods;
        public final int numDatasets;
        public final double alpha;
        public final double qAlpha;
        public final double criticalDifference;
        public final double[] averageRanks;
        public final double[][] rankDifferences;
        public final boolean[][] significant;
        public final List<int[]> significantPairs;

        public Result(int numMethods, int numDatasets, double alpha, double qAlpha,
                      double criticalDifference, double[] averageRanks,
                      double[][] rankDifferences, boolean[][] significant, List<int[]> significantPairs) {
            this.numMethods = numMethods;
            this.numDatasets = numDatasets;
            this.alpha = alpha;
            this.qAlpha = qAlpha;
            this.criticalDifference = criticalDifference;
            this.averageRanks = averageRanks;
            this.rankDifferences = rankDifferences;
            this.significant = significant;
            this.significantPairs = significantPairs;
        }
    }
}