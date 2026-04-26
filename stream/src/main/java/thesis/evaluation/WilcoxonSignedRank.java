package thesis.evaluation;

import org.apache.commons.math3.stat.inference.WilcoxonSignedRankTest;

public class WilcoxonSignedRank {

    private final boolean exact;

    public WilcoxonSignedRank() { this(true); }

    public WilcoxonSignedRank(boolean exact) {
        this.exact = exact;
    }

    public Result test(double[] a, double[] b) {
        if (a == null || b == null) throw new IllegalArgumentException("inputs are null");
        if (a.length != b.length) throw new IllegalArgumentException("vectors must have equal length");
        if (a.length < 2) throw new IllegalArgumentException("need at least 2 paired samples");

        WilcoxonSignedRankTest w = new WilcoxonSignedRankTest();
        double statistic = w.wilcoxonSignedRank(a, b);
        double pValue;
        boolean usedExact;
        if (exact && a.length <= 30) {
            pValue = w.wilcoxonSignedRankTest(a, b, true);
            usedExact = true;
        } else {
            pValue = w.wilcoxonSignedRankTest(a, b, false);
            usedExact = false;
        }

        int n = a.length;
        int wins = 0, losses = 0, ties = 0;
        for (int i = 0; i < n; i++) {
            double d = a[i] - b[i];
            if (d > 0) wins++;
            else if (d < 0) losses++;
            else ties++;
        }
        return new Result(n, statistic, pValue, usedExact, wins, losses, ties);
    }

    public static final class Result {
        public final int n;
        public final double statistic;
        public final double pValue;
        public final boolean exact;
        public final int wins;
        public final int losses;
        public final int ties;

        public Result(int n, double statistic, double pValue, boolean exact,
                      int wins, int losses, int ties) {
            this.n = n;
            this.statistic = statistic;
            this.pValue = pValue;
            this.exact = exact;
            this.wins = wins;
            this.losses = losses;
            this.ties = ties;
        }

        public boolean rejectsNull(double alpha) { return pValue < alpha; }
    }
}