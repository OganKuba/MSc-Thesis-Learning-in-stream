package thesis.evaluation;

import org.apache.commons.math3.distribution.ChiSquaredDistribution;
import org.apache.commons.math3.distribution.FDistribution;

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
        if (n < 2) throw new IllegalArgumentException("need at least 2 datasets");
        for (int i = 0; i < n; i++) {
            if (matrix[i] == null || matrix[i].length != k) {
                throw new IllegalArgumentException("inconsistent row width at row " + i);
            }
            for (int j = 0; j < k; j++) {
                if (!Double.isFinite(matrix[i][j])) {
                    throw new IllegalArgumentException("non-finite entry at [" + i + "," + j + "]");
                }
            }
        }

        double[][] ranks = new double[n][k];
        double tieCorrectionSum = 0.0;
        for (int i = 0; i < n; i++) {
            RankedRow rr = rankRow(matrix[i], higherIsBetter);
            ranks[i] = rr.ranks;
            tieCorrectionSum += rr.tieCorrection;
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
        double denom = (double) n * (double) k * (double) (k + 1);
        double chi2Raw = (12.0 / denom) * sumR2 - 3.0 * n * (k + 1);
        if (chi2Raw < 0.0) chi2Raw = 0.0;

        double tieDenom = 1.0 - tieCorrectionSum
                / ((double) n * ((double) k * k * k - (double) k));
        if (tieDenom <= 0.0 || !Double.isFinite(tieDenom)) tieDenom = 1.0;
        double chi2 = chi2Raw / tieDenom;

        int dfChi = k - 1;
        double pChi = (chi2 == 0.0) ? 1.0
                : 1.0 - new ChiSquaredDistribution(dfChi).cumulativeProbability(chi2);

        int df1 = k - 1;
        int df2 = (k - 1) * (n - 1);
        double imanDavenport;
        double pF;
        double residual = (double) n * (k - 1) - chi2;
        if (df2 <= 0) {
            imanDavenport = Double.NaN;
            pF = Double.NaN;
        } else if (chi2 == 0.0) {
            imanDavenport = 0.0;
            pF = 1.0;
        } else if (residual <= 1e-12) {
            imanDavenport = Double.POSITIVE_INFINITY;
            pF = 0.0;
        } else {
            imanDavenport = ((n - 1) * chi2) / residual;
            pF = 1.0 - new FDistribution(df1, df2).cumulativeProbability(imanDavenport);
        }

        return new Result(k, n, avgRanks, ranks, chi2, pChi, imanDavenport, pF, dfChi, df1, df2);
    }

    private static final class RankedRow {
        final double[] ranks;
        final double tieCorrection;
        RankedRow(double[] r, double tc) { this.ranks = r; this.tieCorrection = tc; }
    }

    private static RankedRow rankRow(double[] row, boolean higherIsBetter) {
        int k = row.length;
        Integer[] idx = new Integer[k];
        for (int i = 0; i < k; i++) idx[i] = i;
        java.util.Arrays.sort(idx, (a, b) -> higherIsBetter
                ? Double.compare(row[b], row[a])
                : Double.compare(row[a], row[b]));
        double[] ranks = new double[k];
        double tieCorrection = 0.0;
        int i = 0;
        while (i < k) {
            int j = i + 1;
            while (j < k && Double.compare(row[idx[j]], row[idx[i]]) == 0) j++;
            double avg = ((i + 1) + j) / 2.0;
            int t = j - i;
            if (t > 1) tieCorrection += (double) t * t * t - (double) t;
            for (int p = i; p < j; p++) ranks[idx[p]] = avg;
            i = j;
        }
        return new RankedRow(ranks, tieCorrection);
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
        public final int dfF1;
        public final int dfF2;

        public Result(int numMethods, int numDatasets, double[] averageRanks, double[][] ranks,
                      double chiSquared, double pValueChi,
                      double imanDavenport, double pValueF,
                      int dfChi, int dfF1, int dfF2) {
            this.numMethods = numMethods;
            this.numDatasets = numDatasets;
            this.averageRanks = averageRanks;
            this.ranks = ranks;
            this.chiSquared = chiSquared;
            this.pValueChi = pValueChi;
            this.imanDavenport = imanDavenport;
            this.pValueF = pValueF;
            this.dfChi = dfChi;
            this.dfF1 = dfF1;
            this.dfF2 = dfF2;
        }

        public boolean rejectsNull(double alpha) {
            return !Double.isNaN(pValueF) ? pValueF < alpha : pValueChi < alpha;
        }
    }
}