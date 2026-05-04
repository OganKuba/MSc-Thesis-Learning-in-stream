package thesis.discretization;

import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Random;
import java.util.Set;

public class DiscretizationSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("DISCRETIZATION SMOKE TESTS");
        System.out.println("=".repeat(70));

        testLayer1HistogramBinning();
        testLayer1HistogramClipsOutOfRange();
        testLayer1HistogramFromWarmup();
        testLayer1HistogramRejectsBadParams();
        testLayer1HistogramTracksOverflowUnderflow();
        testLayer1HistogramRebinPreservesTotal();
        testLayer1HistogramDecayShrinksCounts();

        testLayer2MergerReducesToB2();
        testLayer2MergerMonotonicMapping();
        testLayer2MergerKeepsTotalCount();
        testLayer2MergerRejectsBadB2();
        testLayer2MergerPrefersSimilarDistributions();

        testFeatureDiscretizerNotReadyDuringWarmup();
        testFeatureDiscretizerReadyAfterWarmup();
        testFeatureDiscretizerProducesValidBins();
        testFeatureDiscretizerUnknownBinForNanAndNotReady();
        testFeatureDiscretizerExpandsRangeOnDrift();
        testFeatureDiscretizerSoftResetForgetsHistory();
        testFeatureDiscretizerRecomputeResetsCounter();
        testFeatureDiscretizerResetClearsState();
        testFeatureDiscretizerRejectsBadParams();

        testPiDDiscretizerUpdatesAllFeatures();
        testPiDDiscretizerDiscretizeAllShape();
        testPiDDiscretizerAutoRecomputeFiresAfterInterval();
        testPiDDiscretizerSeparatesClassesOnInformativeFeature();
        testPiDDiscretizerResetFeatureIsolated();
        testPiDDiscretizerRejectsBadInputs();
        testPiDRankingChangesAfterDriftWhenSoftReset();
        testPiDDoesNotDegenerateUnderTrend();
        testPiDOnDriftAlarmAppliesOnlyToDriftingFeatures();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    private static void testLayer1HistogramBinning() {
        Layer1Histogram h = new Layer1Histogram(10, 2, 0.0, 10.0);
        boolean ok = h.bin(0.0) == 0 && h.bin(0.999) == 0 && h.bin(1.0) == 1
                && h.bin(9.999) == 9 && h.bin(5.0) == 5;
        report("Layer1Histogram bin() maps values correctly", ok);
    }

    private static void testLayer1HistogramClipsOutOfRange() {
        Layer1Histogram h = new Layer1Histogram(8, 2, 0.0, 8.0);
        boolean ok = h.bin(-100.0) == 0 && h.bin(1e9) == 7;
        report("Layer1Histogram clips out-of-range values", ok);
    }

    private static void testLayer1HistogramFromWarmup() {
        Random r = new Random(11);
        int n = 500;
        double[] vals = new double[n];
        int[] cls = new int[n];
        for (int i = 0; i < n; i++) { vals[i] = r.nextDouble() * 100.0; cls[i] = r.nextInt(2); }
        Layer1Histogram h = Layer1Histogram.fromWarmup(vals, cls, 16, 2);
        long total = h.totalCount();
        report("Layer1Histogram.fromWarmup ingests all samples (total=" + total + ")", total == n);
    }

    private static void testLayer1HistogramRejectsBadParams() {
        boolean t1 = false, t2 = false, t3 = false;
        try { new Layer1Histogram(2, 2, 0, 1); } catch (IllegalArgumentException e) { t1 = true; }
        try { new Layer1Histogram(8, 1, 0, 1); } catch (IllegalArgumentException e) { t2 = true; }
        try { new Layer1Histogram(8, 2, 1, 1); } catch (IllegalArgumentException e) { t3 = true; }
        report("Layer1Histogram rejects invalid params", t1 && t2 && t3);
    }

    private static void testLayer1HistogramTracksOverflowUnderflow() {
        Layer1Histogram h = new Layer1Histogram(8, 2, 0.0, 1.0);
        for (int i = 0; i < 10; i++) h.update(-5.0, 0);
        for (int i = 0; i < 7; i++) h.update(99.0, 1);
        report("Layer1Histogram tracks underflow/overflow",
                h.getUnderflow() == 10 && h.getOverflow() == 7);
    }

    private static void testLayer1HistogramRebinPreservesTotal() {
        Layer1Histogram h = new Layer1Histogram(8, 2, 0.0, 8.0);
        Random r = new Random(33);
        for (int i = 0; i < 200; i++) h.update(r.nextDouble() * 8.0, r.nextInt(2));
        long before = h.totalCount();
        h.rebin(-2.0, 12.0);
        long after = h.totalCount();
        report("rebin preserves total count (" + before + " -> " + after + ")", before == after);
    }

    private static void testLayer1HistogramDecayShrinksCounts() {
        Layer1Histogram h = new Layer1Histogram(8, 2, 0.0, 8.0);
        for (int i = 0; i < 1000; i++) h.update(i % 8, i % 2);
        long before = h.totalCount();
        h.decay(0.5);
        long after = h.totalCount();
        report("decay shrinks counts (" + before + " -> " + after + ")",
                after < before && after > before / 4);
    }

    private static void testLayer2MergerReducesToB2() {
        int b1 = 20, b2 = 5, K = 3;
        int[] counts = new int[b1];
        int[][] cc = new int[b1][K];
        Random r = new Random(12);
        for (int i = 0; i < b1; i++) {
            counts[i] = 10 + r.nextInt(20);
            for (int c = 0; c < K; c++) cc[i][c] = r.nextInt(counts[i] + 1);
        }
        int[] map = Layer2Merger.merge(counts, cc, b2, K);
        Set<Integer> groups = new HashSet<>();
        for (int g : map) groups.add(g);
        report("Layer2Merger reduces to b2", map.length == b1 && groups.size() == b2);
    }

    private static void testLayer2MergerMonotonicMapping() {
        int b1 = 30, b2 = 6, K = 2;
        int[] counts = new int[b1]; Arrays.fill(counts, 5);
        int[][] cc = new int[b1][K];
        for (int i = 0; i < b1; i++) { cc[i][0] = 3; cc[i][1] = 2; }
        int[] map = Layer2Merger.merge(counts, cc, b2, K);
        boolean monotonic = true;
        for (int i = 1; i < map.length; i++) if (map[i] < map[i - 1]) { monotonic = false; break; }
        report("Layer2Merger monotonic L1->L2 mapping", monotonic);
    }

    private static void testLayer2MergerKeepsTotalCount() {
        int b1 = 25, b2 = 4, K = 3;
        int[] counts = new int[b1]; int[][] cc = new int[b1][K];
        Random r = new Random(13); int total = 0;
        for (int i = 0; i < b1; i++) {
            counts[i] = 1 + r.nextInt(50); total += counts[i];
            int rem = counts[i];
            for (int c = 0; c < K - 1; c++) { cc[i][c] = r.nextInt(rem + 1); rem -= cc[i][c]; }
            cc[i][K - 1] = rem;
        }
        int[] map = Layer2Merger.merge(counts, cc, b2, K);
        int[] grp = new int[b2]; int sum = 0;
        for (int i = 0; i < b1; i++) grp[map[i]] += counts[i];
        for (int g : grp) sum += g;
        report("Layer2Merger preserves total count", total == sum);
    }

    private static void testLayer2MergerRejectsBadB2() {
        boolean t1 = false, t2 = false;
        try { Layer2Merger.merge(new int[10], new int[10][2], 1, 2); } catch (IllegalArgumentException e) { t1 = true; }
        try { Layer2Merger.merge(new int[10], new int[10][2], 11, 2); } catch (IllegalArgumentException e) { t2 = true; }
        report("Layer2Merger rejects invalid b2", t1 && t2);
    }

    private static void testLayer2MergerPrefersSimilarDistributions() {
        int b1 = 6, b2 = 3, K = 2;
        int[] counts = new int[]{100, 100, 100, 100, 100, 100};
        int[][] cc = new int[][]{{90,10},{88,12},{50,50},{52,48},{10,90},{12,88}};
        int[] map = Layer2Merger.merge(counts, cc, b2, K);
        boolean ok = map[0] == map[1] && map[2] == map[3] && map[4] == map[5]
                && map[1] != map[2] && map[3] != map[4];
        report("Layer2Merger merges similar-distribution neighbors first " + Arrays.toString(map), ok);
    }

    private static void testFeatureDiscretizerNotReadyDuringWarmup() {
        FeatureDiscretizer f = new FeatureDiscretizer(10, 4, 2, 100);
        for (int i = 0; i < 50; i++) f.update(i, i % 2);
        report("FeatureDiscretizer not ready during warmup",
                !f.isReady() && f.discretize(42.0) == FeatureDiscretizer.UNKNOWN_BIN);
    }

    private static void testFeatureDiscretizerReadyAfterWarmup() {
        FeatureDiscretizer f = new FeatureDiscretizer(10, 4, 2, 100);
        Random r = new Random(14);
        for (int i = 0; i < 100; i++) f.update(r.nextDouble(), r.nextInt(2));
        report("FeatureDiscretizer ready after warmup",
                f.isReady() && f.getLayer1() != null && f.getL1ToL2Mapping() != null);
    }

    private static void testFeatureDiscretizerProducesValidBins() {
        FeatureDiscretizer f = new FeatureDiscretizer(16, 5, 2, 200);
        Random r = new Random(15);
        for (int i = 0; i < 200; i++) f.update(r.nextGaussian(), r.nextInt(2));
        boolean ok = true;
        for (int i = 0; i < 1000; i++) {
            int b = f.discretize(r.nextGaussian());
            if (b < 0 || b >= 5) { ok = false; break; }
        }
        report("FeatureDiscretizer outputs bins in [0,b2)", ok);
    }

    private static void testFeatureDiscretizerUnknownBinForNanAndNotReady() {
        FeatureDiscretizer f = new FeatureDiscretizer(10, 4, 2, 100);
        boolean nrUnknown = f.discretize(0.5) == FeatureDiscretizer.UNKNOWN_BIN;
        Random r = new Random(99);
        for (int i = 0; i < 100; i++) f.update(r.nextGaussian(), r.nextInt(2));
        boolean nanUnknown = f.discretize(Double.NaN) == FeatureDiscretizer.UNKNOWN_BIN;
        report("UNKNOWN_BIN returned for not-ready and NaN", nrUnknown && nanUnknown);
    }

    private static void testFeatureDiscretizerExpandsRangeOnDrift() {
        FeatureDiscretizer f = new FeatureDiscretizer(16, 5, 2, 200);
        Random r = new Random(77);
        for (int i = 0; i < 200; i++) f.update(r.nextGaussian(), r.nextInt(2));
        double minBefore = f.getLayer1().getMin();
        double maxBefore = f.getLayer1().getMax();
        for (int i = 0; i < 500; i++) f.update(r.nextGaussian() + 20.0, r.nextInt(2));
        double maxAfter = f.getLayer1().getMax();
        report("range expanded on drift (max " + maxBefore + " -> " + maxAfter + ", expansions=" + f.getExpansions() + ")",
                maxAfter > maxBefore + 5.0 && f.getExpansions() >= 1
                        && Math.abs(f.getLayer1().getMin() - minBefore) < Math.abs(maxAfter));
    }

    private static void testFeatureDiscretizerSoftResetForgetsHistory() {
        FeatureDiscretizer f = new FeatureDiscretizer(16, 5, 2, 200);
        Random r = new Random(78);
        for (int i = 0; i < 1000; i++) f.update(r.nextGaussian(), r.nextInt(2));
        long before = f.getLayer1().totalCount();
        f.softReset(0.1);
        long after = f.getLayer1().totalCount();
        report("softReset shrinks counts but keeps ready (" + before + " -> " + after + ")",
                f.isReady() && after < before / 5);
    }

    private static void testFeatureDiscretizerRecomputeResetsCounter() {
        FeatureDiscretizer f = new FeatureDiscretizer(10, 4, 2, 100);
        Random r = new Random(16);
        for (int i = 0; i < 150; i++) f.update(r.nextDouble(), r.nextInt(2));
        long before = f.getUpdatesSinceRecompute();
        f.recomputeLayer2();
        report("recomputeLayer2 resets counter", before > 0 && f.getUpdatesSinceRecompute() == 0);
    }

    private static void testFeatureDiscretizerResetClearsState() {
        FeatureDiscretizer f = new FeatureDiscretizer(10, 4, 2, 100);
        Random r = new Random(17);
        for (int i = 0; i < 200; i++) f.update(r.nextDouble(), r.nextInt(2));
        f.reset();
        report("reset clears state",
                !f.isReady() && f.getTotalUpdates() == 0
                        && f.getLayer1() == null && f.getL1ToL2Mapping() == null);
    }

    private static void testFeatureDiscretizerRejectsBadParams() {
        boolean t1=false,t2=false,t3=false,t4=false,t5=false,t6=false;
        try { new FeatureDiscretizer(3, 2, 2, 100); } catch (IllegalArgumentException e) { t1=true; }
        try { new FeatureDiscretizer(10, 1, 2, 100); } catch (IllegalArgumentException e) { t2=true; }
        try { new FeatureDiscretizer(10, 4, 1, 100); } catch (IllegalArgumentException e) { t3=true; }
        try { new FeatureDiscretizer(10, 4, 2, 5); } catch (IllegalArgumentException e) { t4=true; }
        try { new FeatureDiscretizer(10,4,2,100, 0.0, 1.0); } catch (IllegalArgumentException e) { t5=true; }
        try { new FeatureDiscretizer(10,4,2,100, 0.2, 0.0); } catch (IllegalArgumentException e) { t6=true; }
        report("FeatureDiscretizer rejects invalid params", t1&&t2&&t3&&t4&&t5&&t6);
    }

    private static void testPiDDiscretizerUpdatesAllFeatures() {
        PiDDiscretizer p = new PiDDiscretizer(5, 2, 16, 4, 100, 500);
        Random r = new Random(18);
        for (int i = 0; i < 100; i++) {
            double[] v = new double[5]; for (int k = 0; k < 5; k++) v[k] = r.nextGaussian();
            p.update(v, r.nextInt(2));
        }
        report("PiDDiscretizer ready after warmup", p.isReady());
    }

    private static void testPiDDiscretizerDiscretizeAllShape() {
        PiDDiscretizer p = new PiDDiscretizer(7, 3, 16, 5, 100, 500);
        Random r = new Random(19);
        for (int i = 0; i < 100; i++) {
            double[] v = new double[7]; for (int k = 0; k < 7; k++) v[k] = r.nextGaussian();
            p.update(v, r.nextInt(3));
        }
        int[] bins = p.discretizeAll(new double[]{0,0.5,-0.5,1,-1,2,-2});
        boolean ok = bins.length == 7;
        for (int b : bins) if (b < 0 || b >= 5) ok = false;
        report("discretizeAll returns bins in range", ok);
    }

    private static void testPiDDiscretizerAutoRecomputeFiresAfterInterval() {
        int warmup = 100, every = 200;
        PiDDiscretizer p = new PiDDiscretizer(2, 2, 16, 4, warmup, every);
        Random r = new Random(20);
        for (int i = 0; i < warmup; i++) p.update(new double[]{r.nextGaussian(), r.nextGaussian()}, r.nextInt(2));
        for (int i = 0; i < every + 5; i++) p.update(new double[]{r.nextGaussian(), r.nextGaussian()}, r.nextInt(2));
        long after = p.getFeature(0).getUpdatesSinceRecompute();
        report("auto-recompute fires after interval (after=" + after + ")", after < every);
    }

    private static void testPiDDiscretizerSeparatesClassesOnInformativeFeature() {
        PiDDiscretizer p = new PiDDiscretizer(1, 2, 32, 6, 400, 200);
        Random r = new Random(21);
        for (int i = 0; i < 2000; i++) {
            int cls = r.nextInt(2);
            double v = (cls == 0) ? r.nextGaussian() - 2.0 : r.nextGaussian() + 2.0;
            p.update(new double[]{v}, cls);
        }
        int[][] cc = p.getL2ClassCounts(0);
        boolean ok = cc[0][0] > cc[0][1] && cc[cc.length - 1][1] > cc[cc.length - 1][0];
        report("L2 bins capture class separation", ok);
    }

    private static void testPiDDiscretizerResetFeatureIsolated() {
        PiDDiscretizer p = new PiDDiscretizer(3, 2, 16, 4, 100, 500);
        Random r = new Random(22);
        for (int i = 0; i < 200; i++)
            p.update(new double[]{r.nextGaussian(), r.nextGaussian(), r.nextGaussian()}, r.nextInt(2));
        p.resetFeature(1);
        report("resetFeature isolated", p.isReady(0) && !p.isReady(1) && p.isReady(2));
    }

    private static void testPiDDiscretizerRejectsBadInputs() {
        PiDDiscretizer p = new PiDDiscretizer(4, 2, 16, 4, 100, 500);
        boolean t1=false,t2=false,t3=false,t4=false;
        try { p.update(new double[]{0,0}, 0); } catch (IllegalArgumentException e) { t1=true; }
        try { p.update(new double[]{0,0,0,0}, -1); } catch (IllegalArgumentException e) { t2=true; }
        try { p.update(new double[]{0,0,0,0}, 5); } catch (IllegalArgumentException e) { t3=true; }
        try { new PiDDiscretizer(0, 2); } catch (IllegalArgumentException e) { t4=true; }
        report("PiDDiscretizer rejects bad inputs", t1&&t2&&t3&&t4);
    }

    private static double mutualInfo(int[][] cc, int K) {
        long total = 0;
        for (int[] row : cc) for (int v : row) total += v;
        if (total == 0) return 0.0;
        double[] pX = new double[cc.length];
        double[] pY = new double[K];
        for (int i = 0; i < cc.length; i++) {
            for (int c = 0; c < K; c++) { pX[i] += cc[i][c]; pY[c] += cc[i][c]; }
        }
        for (int i = 0; i < pX.length; i++) pX[i] /= total;
        for (int c = 0; c < K; c++) pY[c] /= total;
        double mi = 0.0;
        for (int i = 0; i < cc.length; i++) {
            for (int c = 0; c < K; c++) {
                double pxy = cc[i][c] / (double) total;
                if (pxy > 0 && pX[i] > 0 && pY[c] > 0) mi += pxy * (Math.log(pxy / (pX[i] * pY[c])) / Math.log(2));
            }
        }
        return mi;
    }

    private static int argmax(double[] a) {
        int best = 0; for (int i = 1; i < a.length; i++) if (a[i] > a[best]) best = i; return best;
    }

    private static void testPiDRankingChangesAfterDriftWhenSoftReset() {
        int F = 4;
        PiDDiscretizer p = new PiDDiscretizer(F, 2, 32, 8, 300, 200, 0.20, 1.0);
        Random r = new Random(123);
        for (int t = 0; t < 3000; t++) {
            int cls = r.nextInt(2);
            double[] v = new double[F];
            v[0] = (cls == 0 ? -2 : 2) + r.nextGaussian();
            for (int i = 1; i < F; i++) v[i] = r.nextGaussian();
            p.update(v, cls);
        }
        double[] miBefore = new double[F];
        for (int i = 0; i < F; i++) miBefore[i] = mutualInfo(p.getL2ClassCounts(i), 2);
        int bestBefore = argmax(miBefore);

        for (int t = 0; t < 3000; t++) {
            int cls = r.nextInt(2);
            double[] v = new double[F];
            v[0] = r.nextGaussian();
            v[2] = (cls == 0 ? -2 : 2) + r.nextGaussian();
            v[1] = r.nextGaussian(); v[3] = r.nextGaussian();
            p.update(v, cls);
        }
        double[] miNoReset = new double[F];
        for (int i = 0; i < F; i++) miNoReset[i] = mutualInfo(p.getL2ClassCounts(i), 2);
        int bestNoReset = argmax(miNoReset);

        Set<Integer> drifting = new LinkedHashSet<>();
        drifting.add(0); drifting.add(2);
        p.onDriftAlarm(drifting, 0.05);
        for (int t = 0; t < 1500; t++) {
            int cls = r.nextInt(2);
            double[] v = new double[F];
            v[0] = r.nextGaussian();
            v[2] = (cls == 0 ? -2 : 2) + r.nextGaussian();
            v[1] = r.nextGaussian(); v[3] = r.nextGaussian();
            p.update(v, cls);
        }
        p.recomputeLayer2();
        double[] miAfter = new double[F];
        for (int i = 0; i < F; i++) miAfter[i] = mutualInfo(p.getL2ClassCounts(i), 2);
        int bestAfter = argmax(miAfter);

        boolean ok = bestBefore == 0 && bestAfter == 2;
        report("MI ranking adapts after drift+softReset (before=" + bestBefore +
                ", noReset=" + bestNoReset + ", afterReset=" + bestAfter + ")", ok);
    }

    private static void testPiDDoesNotDegenerateUnderTrend() {
        PiDDiscretizer p = new PiDDiscretizer(1, 2, 32, 6, 300, 500, 0.20, 1.0);
        Random r = new Random(321);
        for (int t = 0; t < 5000; t++) {
            double trend = t * 0.01;
            double v = trend + r.nextGaussian();
            p.update(new double[]{v}, r.nextInt(2));
        }
        int[] counts = p.getL2Counts(0);
        long total = 0; for (int c : counts) total += c;
        int nonEmpty = 0; for (int c : counts) if (c > 0) nonEmpty++;
        long maxBin = 0; for (int c : counts) if (c > maxBin) maxBin = c;
        double maxFrac = (double) maxBin / Math.max(1, total);
        report("L2 not degenerate under trend (nonEmpty=" + nonEmpty + "/" + counts.length +
                        ", maxFrac=" + String.format("%.2f", maxFrac) + ", expansions=" +
                        p.getFeature(0).getExpansions() + ")",
                nonEmpty >= 4 && maxFrac < 0.6);
    }

    private static void testPiDOnDriftAlarmAppliesOnlyToDriftingFeatures() {
        int F = 3;
        PiDDiscretizer p = new PiDDiscretizer(F, 2, 16, 4, 100, 500, 0.20, 1.0);
        Random r = new Random(7);
        for (int t = 0; t < 1000; t++) {
            double[] v = new double[F]; for (int i = 0; i < F; i++) v[i] = r.nextGaussian();
            p.update(v, r.nextInt(2));
        }
        long t1Before = p.getFeature(1).getLayer1().totalCount();
        long t0Before = p.getFeature(0).getLayer1().totalCount();
        Set<Integer> drift = new HashSet<>(); drift.add(1);
        p.onDriftAlarm(drift, 0.1);
        long t1After = p.getFeature(1).getLayer1().totalCount();
        long t0After = p.getFeature(0).getLayer1().totalCount();
        report("onDriftAlarm only affects drifting features (f0: " + t0Before + "->" + t0After +
                        ", f1: " + t1Before + "->" + t1After + ")",
                t0After == t0Before && t1After < t1Before / 5);
    }

    private static void report(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  [PASSED] " + name); }
        else    { failed++; System.out.println("  [FAILED] " + name); }
    }
}