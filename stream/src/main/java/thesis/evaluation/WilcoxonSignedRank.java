package thesis.evaluation;

import org.apache.commons.math3.stat.inference.WilcoxonSignedRankTest;

public class WilcoxonSignedRank {

    private final boolean exactWhenPossible;
    private final boolean dropZeros;

    public WilcoxonSignedRank() { this(true, true); }

    public WilcoxonSignedRank(boolean exactWhenPossible, boolean dropZeros) {
        this.exactWhenPossible = exactWhenPossible;
        this.dropZeros = dropZeros;
    }

    public Result test(double[] a, double[] b) {
        if (a == null || b == null) throw new IllegalArgumentException("inputs are null");
        if (a.length != b.length) throw new IllegalArgumentException("vectors must have equal length");
        if (a.length < 2) throw new IllegalArgumentException("need at least 2 paired samples");

        int n = a.length;
        int wins = 0, losses = 0, ties = 0;
        for (int i = 0; i < n; i++) {
            if (!Double.isFinite(a[i]) || !Double.isFinite(b[i])) {
                throw new IllegalArgumentException("non-finite value at index " + i);
            }
            double d = a[i] - b[i];
            if (d > 0) wins++;
            else if (d < 0) losses++;
            else ties++;
        }

        if (wins + losses == 0) {
            return new Result(n, 0, 0.0, 1.0, false, wins, losses, ties, true);
        }

        double[] aa, bb;
        if (dropZeros && ties > 0) {
            int m = n - ties;
            aa = new double[m];
            bb = new double[m];
            int p = 0;
            for (int i = 0; i < n; i++) {
                if (a[i] != b[i]) { aa[p] = a[i]; bb[p] = b[i]; p++; }
            }
        } else {
            aa = a; bb = b;
        }
        int effectiveN = aa.length;
        if (effectiveN < 2) {
            return new Result(n, effectiveN, 0.0, 1.0, false, wins, losses, ties, true);
        }

        WilcoxonSignedRankTest w = new WilcoxonSignedRankTest();
        double statistic = w.wilcoxonSignedRank(aa, bb);
        boolean canExact = exactWhenPossible && effectiveN < 30
                && (dropZeros ? true : ties == 0);
        double pValue = w.wilcoxonSignedRankTest(aa, bb, canExact);
        if (!Double.isFinite(pValue)) pValue = 1.0;
        return new Result(n, effectiveN, statistic, pValue, canExact, wins, losses, ties, false);
    }

    public static final class Result {
        public final int n;
        public final int effectiveN;
        public final double statistic;
        public final double pValue;
        public final boolean exact;
        public final int wins;
        public final int losses;
        public final int ties;
        public final boolean degenerate;

        public Result(int n, int effectiveN, double statistic, double pValue, boolean exact,
                      int wins, int losses, int ties, boolean degenerate) {
            this.n = n;
            this.effectiveN = effectiveN;
            this.statistic = statistic;
            this.pValue = pValue;
            this.exact = exact;
            this.wins = wins;
            this.losses = losses;
            this.ties = ties;
            this.degenerate = degenerate;
        }

        public boolean rejectsNull(double alpha) {
            return !degenerate && pValue < alpha;
        }
    }
}