package thesis.evaluation;

import org.apache.commons.math3.distribution.ChiSquaredDistribution;
import org.apache.commons.math3.distribution.FDistribution;

import java.util.Arrays;

public class FriedmanTest {

    private final boolean higherIsBetter;

    public FriedmanTest() { this(true); }

    public FriedmanTest(boolean higherIsBetter) {
        this.higherIsBetter = higherIsBetter;
    }

    public Result test(double[][] matrix) {
        if (matrix == null || matrix.length == 0) throw new IllegalArgumentException("matrix is empty");
        int n = matrix.length;
        int k = matrix[0].length;
        if (k < 2) throw new IllegalArgumentException("need at least 2 methods");
        for (double[] row : matrix) {
            if (row.length != k) throw new IllegalArgumentException("inconsistent row width");
        }

        double[][] ranks = new double[n][k];
        for (int i = 0; i < n; i++) {
            ranks[i] = rankRow(matrix[i], higherIsBetter);
        }

        double[] avgRanks = new double[k];
        for (int j = 0; j < k; j++) {
            double s = 0.0;
            for (int i = 0; i < n; i++) s += ranks[i][j];
            avgRanks[j] = s / n;
        }

        double sumR2 = 0.0;
        for (int j = 0; j < k; j++) {
            double rj = avgRanks[j] * n;
            sumR2 += rj * rj;
        }
        double chi2 = (12.0 / (n * k * (k + 1))) * sumR2 - 3.0 * n * (k + 1);

        double pChi = 1.0 - new ChiSquaredDistribution(k - 1).cumulativeProbability(chi2);

        double imanDavenport;
        double pF;
        int df1 = k - 1;
        int df2 = (k - 1) * (n - 1);
        if (n > 1 && (n * (k - 1) - chi2) > 1e-12) {
            imanDavenport = ((n - 1) * chi2) / (n * (k - 1) - chi2);
            pF = 1.0 - new FDistribution(df1, df2).cumulativeProbability(imanDavenport);
        } else {
            imanDavenport = Double.NaN;
            pF = Double.NaN;
        }

        return new Result(k, n, avgRanks, ranks, chi2, pChi, imanDavenport, pF, df1, df2);
    }

    private static double[] rankRow(double[] row, boolean higherIsBetter) {
        int k = row.length;
        Integer[] idx = new Integer[k];
        for (int i = 0; i < k; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> higherIsBetter ? Double.compare(row[b], row[a]) : Double.compare(row[a], row[b]));
        double[] ranks = new double[k];
        int i = 0;
        while (i < k) {
            int j = i + 1;
            while (j < k && Double.compare(row[idx[j]], row[idx[i]]) == 0) j++;
            double avg = ((i + 1) + j) / 2.0;
            for (int t = i; t < j; t++) ranks[idx[t]] = avg;
            i = j;
        }
        return ranks;
    }

    public static final class Result {
        public final int numMethods;
        public final int numDatasets;
        public final double[] averageRanks;
        public final double[][] ranks;
        public final double chiSquared;
        public final double pValueChi;
        public final double imanDavenport;
        public final double pValueF;
        public final int dfChi;
        public final int dfF1, dfF2;

        public Result(int numMethods, int numDatasets, double[] averageRanks, double[][] ranks,
                      double chiSquared, double pValueChi,
                      double imanDavenport, double pValueF,
                      int dfChi, int dfF2) {
            this.numMethods = numMethods;
            this.numDatasets = numDatasets;
            this.averageRanks = averageRanks;
            this.ranks = ranks;
            this.chiSquared = chiSquared;
            this.pValueChi = pValueChi;
            this.imanDavenport = imanDavenport;
            this.pValueF = pValueF;
            this.dfChi = dfChi;
            this.dfF1 = numMethods - 1;
            this.dfF2 = dfF2;
        }

        public boolean rejectsNull(double alpha) {
            return !Double.isNaN(pValueF) ? pValueF < alpha : pValueChi < alpha;
        }
    }
}