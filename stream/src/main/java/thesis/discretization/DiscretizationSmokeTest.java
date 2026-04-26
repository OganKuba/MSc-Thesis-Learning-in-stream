package thesis.discretization;

import java.util.Arrays;
import java.util.HashSet;
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

        testLayer2MergerReducesToB2();
        testLayer2MergerMonotonicMapping();
        testLayer2MergerKeepsTotalCount();
        testLayer2MergerRejectsBadB2();

        testFeatureDiscretizerNotReadyDuringWarmup();
        testFeatureDiscretizerReadyAfterWarmup();
        testFeatureDiscretizerProducesValidBins();
        testFeatureDiscretizerRecomputeResetsCounter();
        testFeatureDiscretizerResetClearsState();
        testFeatureDiscretizerRejectsBadParams();

        testPiDDiscretizerUpdatesAllFeatures();
        testPiDDiscretizerDiscretizeAllShape();
        testPiDDiscretizerAutoRecomputeFiresAfterInterval();
        testPiDDiscretizerSeparatesClassesOnInformativeFeature();
        testPiDDiscretizerResetFeatureIsolated();
        testPiDDiscretizerRejectsBadInputs();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    private static void testLayer1HistogramBinning() {
        Layer1Histogram h = new Layer1Histogram(10, 2, 0.0, 10.0);
        boolean ok = h.bin(0.0) == 0
                && h.bin(0.999) == 0
                && h.bin(1.0) == 1
                && h.bin(9.999) == 9
                && h.bin(5.0) == 5;
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
        for (int i = 0; i < n; i++) {
            vals[i] = r.nextDouble() * 100.0;
            cls[i] = r.nextInt(2);
        }
        Layer1Histogram h = Layer1Histogram.fromWarmup(vals, cls, 16, 2);
        int total = 0;
        for (int c : h.getBinCounts()) total += c;
        boolean rangeOk = h.getMin() < 0.0 + 1e-9 + Arrays.stream(vals).min().getAsDouble()
                && h.getMax() > Arrays.stream(vals).max().getAsDouble() - 1e-9;
        report("Layer1Histogram.fromWarmup ingests all samples (total=" + total + ")",
                total == n && rangeOk);
    }

    private static void testLayer1HistogramRejectsBadParams() {
        boolean t1 = false, t2 = false, t3 = false;
        try { new Layer1Histogram(2, 2, 0, 1); } catch (IllegalArgumentException e) { t1 = true; }
        try { new Layer1Histogram(8, 1, 0, 1); } catch (IllegalArgumentException e) { t2 = true; }
        try { new Layer1Histogram(8, 2, 1, 1); } catch (IllegalArgumentException e) { t3 = true; }
        report("Layer1Histogram rejects invalid params", t1 && t2 && t3);
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
        boolean ok = map.length == b1 && groups.size() == b2
                && groups.contains(0) && groups.contains(b2 - 1);
        report("Layer2Merger reduces b1=" + b1 + " to b2=" + b2 + " groups", ok);
    }

    private static void testLayer2MergerMonotonicMapping() {
        int b1 = 30, b2 = 6, K = 2;
        int[] counts = new int[b1];
        Arrays.fill(counts, 5);
        int[][] cc = new int[b1][K];
        for (int i = 0; i < b1; i++) { cc[i][0] = 3; cc[i][1] = 2; }
        int[] map = Layer2Merger.merge(counts, cc, b2, K);
        boolean monotonic = true;
        for (int i = 1; i < map.length; i++) {
            if (map[i] < map[i - 1]) { monotonic = false; break; }
        }
        report("Layer2Merger produces monotonic L1->L2 mapping", monotonic);
    }

    private static void testLayer2MergerKeepsTotalCount() {
        int b1 = 25, b2 = 4, K = 3;
        int[] counts = new int[b1];
        int[][] cc = new int[b1][K];
        Random r = new Random(13);
        int total = 0;
        for (int i = 0; i < b1; i++) {
            counts[i] = 1 + r.nextInt(50);
            total += counts[i];
            int remaining = counts[i];
            for (int c = 0; c < K - 1; c++) {
                cc[i][c] = r.nextInt(remaining + 1);
                remaining -= cc[i][c];
            }
            cc[i][K - 1] = remaining;
        }
        int[] map = Layer2Merger.merge(counts, cc, b2, K);
        int[] grouped = new int[b2];
        for (int i = 0; i < b1; i++) grouped[map[i]] += counts[i];
        int sum = 0;
        for (int g : grouped) sum += g;
        report("Layer2Merger preserves total count (" + total + " == " + sum + ")", total == sum);
    }

    private static void testLayer2MergerRejectsBadB2() {
        boolean t1 = false, t2 = false;
        try { Layer2Merger.merge(new int[10], new int[10][2], 1, 2); }
        catch (IllegalArgumentException e) { t1 = true; }
        try { Layer2Merger.merge(new int[10], new int[10][2], 11, 2); }
        catch (IllegalArgumentException e) { t2 = true; }
        report("Layer2Merger rejects invalid b2", t1 && t2);
    }

    private static void testFeatureDiscretizerNotReadyDuringWarmup() {
        FeatureDiscretizer f = new FeatureDiscretizer(10, 4, 2, 100);
        for (int i = 0; i < 50; i++) f.update(i, i % 2);
        boolean ok = !f.isReady() && f.discretize(42.0) == 0;
        report("FeatureDiscretizer not ready during warmup", ok);
    }

    private static void testFeatureDiscretizerReadyAfterWarmup() {
        FeatureDiscretizer f = new FeatureDiscretizer(10, 4, 2, 100);
        Random r = new Random(14);
        for (int i = 0; i < 100; i++) f.update(r.nextDouble(), r.nextInt(2));
        boolean ok = f.isReady() && f.getLayer1() != null && f.getL1ToL2Mapping() != null;
        report("FeatureDiscretizer ready after warmup", ok);
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

    private static void testFeatureDiscretizerRecomputeResetsCounter() {
        FeatureDiscretizer f = new FeatureDiscretizer(10, 4, 2, 100);
        Random r = new Random(16);
        for (int i = 0; i < 100; i++) f.update(r.nextDouble(), r.nextInt(2));
        for (int i = 0; i < 50; i++) f.update(r.nextDouble(), r.nextInt(2));
        long before = f.getUpdatesSinceRecompute();
        f.recomputeLayer2();
        long after = f.getUpdatesSinceRecompute();
        report("FeatureDiscretizer recomputeLayer2 resets counter (" + before + " -> " + after + ")",
                before > 0 && after == 0);
    }

    private static void testFeatureDiscretizerResetClearsState() {
        FeatureDiscretizer f = new FeatureDiscretizer(10, 4, 2, 100);
        Random r = new Random(17);
        for (int i = 0; i < 200; i++) f.update(r.nextDouble(), r.nextInt(2));
        f.reset();
        boolean ok = !f.isReady() && f.getTotalUpdates() == 0
                && f.getLayer1() == null && f.getL1ToL2Mapping() == null;
        report("FeatureDiscretizer reset clears state", ok);
    }

    private static void testFeatureDiscretizerRejectsBadParams() {
        boolean t1 = false, t2 = false, t3 = false, t4 = false;
        try { new FeatureDiscretizer(3, 2, 2, 100); } catch (IllegalArgumentException e) { t1 = true; }
        try { new FeatureDiscretizer(10, 1, 2, 100); } catch (IllegalArgumentException e) { t2 = true; }
        try { new FeatureDiscretizer(10, 4, 1, 100); } catch (IllegalArgumentException e) { t3 = true; }
        try { new FeatureDiscretizer(10, 4, 2, 5);  } catch (IllegalArgumentException e) { t4 = true; }
        report("FeatureDiscretizer rejects invalid params", t1 && t2 && t3 && t4);
    }

    private static void testPiDDiscretizerUpdatesAllFeatures() {
        PiDDiscretizer p = new PiDDiscretizer(5, 2, 16, 4, 100, 500);
        Random r = new Random(18);
        for (int i = 0; i < 100; i++) {
            double[] v = new double[5];
            for (int k = 0; k < 5; k++) v[k] = r.nextGaussian();
            p.update(v, r.nextInt(2));
        }
        report("PiDDiscretizer ready after warmup on all features", p.isReady());
    }

    private static void testPiDDiscretizerDiscretizeAllShape() {
        PiDDiscretizer p = new PiDDiscretizer(7, 3, 16, 5, 100, 500);
        Random r = new Random(19);
        for (int i = 0; i < 100; i++) {
            double[] v = new double[7];
            for (int k = 0; k < 7; k++) v[k] = r.nextGaussian();
            p.update(v, r.nextInt(3));
        }
        int[] bins = p.discretizeAll(new double[]{0, 0.5, -0.5, 1, -1, 2, -2});
        boolean ok = bins.length == 7;
        for (int b : bins) if (b < 0 || b >= 5) ok = false;
        report("PiDDiscretizer.discretizeAll returns " + bins.length + " bins in range", ok);
    }

    private static void testPiDDiscretizerAutoRecomputeFiresAfterInterval() {
        int warmup = 100, every = 200;
        PiDDiscretizer p = new PiDDiscretizer(2, 2, 16, 4, warmup, every);
        Random r = new Random(20);
        for (int i = 0; i < warmup; i++) {
            p.update(new double[]{r.nextGaussian(), r.nextGaussian()}, r.nextInt(2));
        }
        long beforeFirst = p.getFeature(0).getUpdatesSinceRecompute();
        for (int i = 0; i < every + 5; i++) {
            p.update(new double[]{r.nextGaussian(), r.nextGaussian()}, r.nextInt(2));
        }
        long after = p.getFeature(0).getUpdatesSinceRecompute();
        report("PiDDiscretizer auto-recomputes layer2 (before=" + beforeFirst + ", after=" + after + ")",
                after < every);
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
        int b0Class0 = cc[0][0], b0Class1 = cc[0][1];
        int bLastClass0 = cc[cc.length - 1][0], bLastClass1 = cc[cc.length - 1][1];
        boolean ok = b0Class0 > b0Class1 && bLastClass1 > bLastClass0;
        report("PiDDiscretizer L2 bins capture class separation"
                + " (low: " + b0Class0 + "/" + b0Class1
                + ", high: " + bLastClass0 + "/" + bLastClass1 + ")", ok);
    }

    private static void testPiDDiscretizerResetFeatureIsolated() {
        PiDDiscretizer p = new PiDDiscretizer(3, 2, 16, 4, 100, 500);
        Random r = new Random(22);
        for (int i = 0; i < 200; i++) {
            p.update(new double[]{r.nextGaussian(), r.nextGaussian(), r.nextGaussian()}, r.nextInt(2));
        }
        p.resetFeature(1);
        boolean ok = p.isReady(0) && !p.isReady(1) && p.isReady(2);
        report("PiDDiscretizer.resetFeature only resets one feature", ok);
    }

    private static void testPiDDiscretizerRejectsBadInputs() {
        PiDDiscretizer p = new PiDDiscretizer(4, 2, 16, 4, 100, 500);
        boolean t1 = false, t2 = false, t3 = false, t4 = false;
        try { p.update(new double[]{0, 0}, 0); } catch (IllegalArgumentException e) { t1 = true; }
        try { p.update(new double[]{0, 0, 0, 0}, -1); } catch (IllegalArgumentException e) { t2 = true; }
        try { p.update(new double[]{0, 0, 0, 0}, 5); } catch (IllegalArgumentException e) { t3 = true; }
        try { new PiDDiscretizer(0, 2); } catch (IllegalArgumentException e) { t4 = true; }
        report("PiDDiscretizer rejects bad inputs", t1 && t2 && t3 && t4);
    }

    private static void report(String name, boolean ok) {
        if (ok) {
            passed++;
            System.out.println("  [PASSED] " + name);
        } else {
            failed++;
            System.out.println("  [FAILED] " + name);
        }
    }
}