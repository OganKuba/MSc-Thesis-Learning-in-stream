package thesis.selection;

import java.util.Arrays;
import java.util.Random;

public class SelectionSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("SELECTION (FILTER RANKER) SMOKE TESTS");
        System.out.println("=".repeat(70));

        testUpdateAccumulatesCounts();
        testUpdateRejectsBadShape();
        testUpdateRejectsBadClassLabel();
        testUpdateSkipsUnknownBinPerFeature();
        testUpdateSkipsOutOfRangeBinPerFeature();
        testIsReadyTransitions();

        testIgZeroForUninformativeFeature();
        testIgHigherForInformativeFeature();
        testIgRanksInformativeFirst();

        testSelectTopKDeterministicOnTies();
        testSelectTopKHonorsPreferredOrder();
        testSelectTopKReactsToNewData();
        testSelectTopKRejectsBadK();

        testDecayShrinksAllCounts();
        testDecayFeatureOnlyAffectsOne();
        testResetClearsState();
        testResetFeatureIsolated();

        testRankingChangesAfterDriftWithDecay();
        testS4LikeSelectivePreservesStable();
        testRareClassDoesNotBreakRanker();
        testEmptyBinsDoNotInflateScore();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    private static FilterRanker[] allRankers(int F, int B, int K) {
        return new FilterRanker[]{
                new InformationGainRanker(F, B, K, 1)
        };
    }

    private static int binFromGaussian(double v, int B) {
        int b = (int) Math.floor((v + 3.0) * B / 6.0);
        if (b < 0) b = 0; if (b >= B) b = B - 1; return b;
    }

    private static void testUpdateAccumulatesCounts() {
        InformationGainRanker r = new InformationGainRanker(3, 4, 2, 1);
        r.update(new int[]{0, 1, 2}, 0);
        r.update(new int[]{0, 1, 3}, 1);
        boolean ok = r.getTotalSamples() == 2
                && r.featureTotals[0] == 2.0
                && r.featureBinTotals[0][0] == 2.0
                && r.featureClassMarginals[0][0] == 1.0
                && r.featureClassMarginals[0][1] == 1.0;
        report("update accumulates counts", ok);
    }

    private static void testUpdateRejectsBadShape() {
        InformationGainRanker r = new InformationGainRanker(3, 4, 2, 1);
        boolean threw = false;
        try { r.update(new int[]{0, 1}, 0); } catch (IllegalArgumentException e) { threw = true; }
        report("update rejects bad shape", threw);
    }

    private static void testUpdateRejectsBadClassLabel() {
        InformationGainRanker r = new InformationGainRanker(3, 4, 2, 1);
        long before = r.getRejectedSamples();
        r.update(new int[]{0, 1, 2}, 5);
        report("update soft-rejects bad classLabel", r.getRejectedSamples() == before + 1
                && r.getTotalSamples() == 0);
    }

    private static void testUpdateSkipsUnknownBinPerFeature() {
        InformationGainRanker r = new InformationGainRanker(3, 4, 2, 1);
        r.update(new int[]{0, AbstractFrequencyRanker.UNKNOWN_BIN, 2}, 0);
        boolean ok = r.featureTotals[0] == 1.0
                && r.featureTotals[1] == 0.0
                && r.featureTotals[2] == 1.0
                && r.getTotalSamples() == 1;
        report("UNKNOWN_BIN is skipped per-feature, others still updated", ok);
    }

    private static void testUpdateSkipsOutOfRangeBinPerFeature() {
        InformationGainRanker r = new InformationGainRanker(3, 4, 2, 1);
        r.update(new int[]{0, 99, 2}, 0);
        boolean ok = r.featureTotals[0] == 1.0
                && r.featureTotals[1] == 0.0
                && r.featureTotals[2] == 1.0;
        report("out-of-range bin skipped per-feature", ok);
    }

    private static void testIsReadyTransitions() {
        InformationGainRanker r = new InformationGainRanker(2, 4, 2, 10);
        for (int i = 0; i < 5; i++) r.update(new int[]{i % 4, i % 4}, i % 2);
        boolean notReady = !r.isReady();
        for (int i = 0; i < 20; i++) r.update(new int[]{i % 4, i % 4}, i % 2);
        report("isReady transitions after minSamplesReady", notReady && r.isReady());
    }

    private static int[] makeInstance(int B, int F, int cls, int informativeIdx, Random r) {
        int[] x = new int[F];
        for (int i = 0; i < F; i++) x[i] = r.nextInt(B);
        x[informativeIdx] = (cls == 0) ? r.nextInt(B / 2) : (B / 2 + r.nextInt(B - B / 2));
        return x;
    }

    private static void testIgZeroForUninformativeFeature() {
        InformationGainRanker r = new InformationGainRanker(2, 4, 2, 1);
        Random rnd = new Random(1);
        for (int i = 0; i < 5000; i++) {
            int cls = rnd.nextInt(2);
            r.update(new int[]{rnd.nextInt(4), rnd.nextInt(4)}, cls);
        }
        double[] s = r.getFeatureScores();
        report("IG ~ 0 for both noise features (s=" + Arrays.toString(s) + ")",
                s[0] < 0.02 && s[1] < 0.02);
    }

    private static void testIgHigherForInformativeFeature() {
        InformationGainRanker r = new InformationGainRanker(2, 4, 2, 1);
        Random rnd = new Random(2);
        for (int i = 0; i < 5000; i++) {
            int cls = rnd.nextInt(2);
            int informative = (cls == 0) ? rnd.nextInt(2) : 2 + rnd.nextInt(2);
            r.update(new int[]{informative, rnd.nextInt(4)}, cls);
        }
        double[] s = r.getFeatureScores();
        report("IG higher for informative feature (s=" + Arrays.toString(s) + ")",
                s[0] > s[1] + 0.3);
    }

    private static void testIgRanksInformativeFirst() {
        int F = 5;
        InformationGainRanker r = new InformationGainRanker(F, 6, 2, 1);
        Random rnd = new Random(3);
        for (int i = 0; i < 5000; i++) {
            int cls = rnd.nextInt(2);
            r.update(makeInstance(6, F, cls, 3, rnd), cls);
        }
        int[] top = r.selectTopK(1);
        report("IG selects feature 3 as top-1 (got=" + top[0] + ")", top[0] == 3);
    }

    private static void testSelectTopKDeterministicOnTies() {
        InformationGainRanker r = new InformationGainRanker(5, 4, 2, 1);
        int[] a = r.selectTopK(3);
        int[] b = r.selectTopK(3);
        report("selectTopK deterministic on all-zero scores (a=" + Arrays.toString(a) +
                ", b=" + Arrays.toString(b) + ")", Arrays.equals(a, b));
    }

    private static void testSelectTopKHonorsPreferredOrder() {
        InformationGainRanker r = new InformationGainRanker(5, 4, 2, 1);
        int[] preferred = new int[]{4, 3, 2, 1, 0};
        int[] sel = r.selectTopK(3, preferred);
        report("selectTopK honors preferredOrder on ties (got=" + Arrays.toString(sel) + ")",
                sel[0] == 4 && sel[1] == 3 && sel[2] == 2);
    }

    private static void testSelectTopKReactsToNewData() {
        InformationGainRanker r = new InformationGainRanker(4, 4, 2, 1);
        Random rnd = new Random(10);
        for (int i = 0; i < 2000; i++) {
            int cls = rnd.nextInt(2);
            int[] x = new int[]{rnd.nextInt(4), rnd.nextInt(4), rnd.nextInt(4), rnd.nextInt(4)};
            x[1] = (cls == 0) ? 0 : 3;
            r.update(x, cls);
        }
        int top1 = r.selectTopK(1)[0];
        r.decay(0.01);
        for (int i = 0; i < 2000; i++) {
            int cls = rnd.nextInt(2);
            int[] x = new int[]{rnd.nextInt(4), rnd.nextInt(4), rnd.nextInt(4), rnd.nextInt(4)};
            x[2] = (cls == 0) ? 0 : 3;
            r.update(x, cls);
        }
        int top2 = r.selectTopK(1)[0];
        report("topK reacts to new data after decay (top1=" + top1 + ", top2=" + top2 + ")",
                top1 == 1 && top2 == 2);
    }

    private static void testSelectTopKRejectsBadK() {
        InformationGainRanker r = new InformationGainRanker(3, 4, 2, 1);
        boolean t1 = false, t2 = false;
        try { r.selectTopK(0); } catch (IllegalArgumentException e) { t1 = true; }
        try { r.selectTopK(4); } catch (IllegalArgumentException e) { t2 = true; }
        report("selectTopK rejects bad k", t1 && t2);
    }

    private static void testDecayShrinksAllCounts() {
        InformationGainRanker r = new InformationGainRanker(2, 4, 2, 1);
        for (int i = 0; i < 1000; i++) r.update(new int[]{i % 4, (i + 1) % 4}, i % 2);
        double before = r.featureTotals[0];
        r.decay(0.5);
        double after = r.featureTotals[0];
        report("decay halves counts (" + before + " -> " + after + ")",
                Math.abs(after - before * 0.5) < 1e-6);
    }

    private static void testDecayFeatureOnlyAffectsOne() {
        InformationGainRanker r = new InformationGainRanker(3, 4, 2, 1);
        for (int i = 0; i < 1000; i++) r.update(new int[]{0, 1, 2}, i % 2);
        double f0Before = r.featureTotals[0], f1Before = r.featureTotals[1], f2Before = r.featureTotals[2];
        r.decayFeature(1, 0.1);
        report("decayFeature only affects target feature",
                r.featureTotals[0] == f0Before
                        && Math.abs(r.featureTotals[1] - f1Before * 0.1) < 1e-6
                        && r.featureTotals[2] == f2Before);
    }

    private static void testResetClearsState() {
        InformationGainRanker r = new InformationGainRanker(2, 4, 2, 1);
        for (int i = 0; i < 100; i++) r.update(new int[]{0, 1}, i % 2);
        r.reset();
        report("reset clears state",
                r.getTotalSamples() == 0 && r.featureTotals[0] == 0.0 && r.featureTotals[1] == 0.0);
    }

    private static void testResetFeatureIsolated() {
        InformationGainRanker r = new InformationGainRanker(3, 4, 2, 1);
        for (int i = 0; i < 100; i++) r.update(new int[]{0, 1, 2}, i % 2);
        r.resetFeature(1);
        report("resetFeature isolated",
                r.featureTotals[0] == 100.0 && r.featureTotals[1] == 0.0 && r.featureTotals[2] == 100.0);
    }

    private static void testRankingChangesAfterDriftWithDecay() {
        int F = 4;
        InformationGainRanker r = new InformationGainRanker(F, 4, 2, 1);
        Random rnd = new Random(11);
        for (int i = 0; i < 5000; i++) {
            int cls = rnd.nextInt(2);
            int[] x = new int[F];
            for (int f = 0; f < F; f++) x[f] = rnd.nextInt(4);
            x[0] = (cls == 0) ? 0 : 3;
            r.update(x, cls);
        }
        int topBefore = r.selectTopK(1)[0];

        for (int i = 0; i < 5000; i++) {
            int cls = rnd.nextInt(2);
            int[] x = new int[F];
            for (int f = 0; f < F; f++) x[f] = rnd.nextInt(4);
            x[2] = (cls == 0) ? 0 : 3;
            r.update(x, cls);
        }
        int topNoDecay = r.selectTopK(1)[0];

        r.decay(0.01);
        for (int i = 0; i < 2000; i++) {
            int cls = rnd.nextInt(2);
            int[] x = new int[F];
            for (int f = 0; f < F; f++) x[f] = rnd.nextInt(4);
            x[2] = (cls == 0) ? 0 : 3;
            r.update(x, cls);
        }
        int topAfterDecay = r.selectTopK(1)[0];
        report("ranking switches after drift with decay (before=" + topBefore +
                        ", noDecay=" + topNoDecay + ", afterDecay=" + topAfterDecay + ")",
                topBefore == 0 && topAfterDecay == 2);
    }

    private static void testS4LikeSelectivePreservesStable() {
        int F = 5;
        InformationGainRanker r = new InformationGainRanker(F, 4, 2, 1);
        Random rnd = new Random(12);
        for (int i = 0; i < 5000; i++) {
            int cls = rnd.nextInt(2);
            int[] x = new int[F];
            for (int f = 0; f < F; f++) x[f] = rnd.nextInt(4);
            x[0] = (cls == 0) ? 0 : 3;
            x[1] = (cls == 0) ? rnd.nextInt(2) : 2 + rnd.nextInt(2);
            r.update(x, cls);
        }
        int[] beforeTop2 = r.selectTopK(2);
        r.decayFeature(1, 0.01);
        for (int i = 0; i < 3000; i++) {
            int cls = rnd.nextInt(2);
            int[] x = new int[F];
            for (int f = 0; f < F; f++) x[f] = rnd.nextInt(4);
            x[0] = (cls == 0) ? 0 : 3;
            x[3] = (cls == 0) ? 0 : 3;
            r.update(x, cls);
        }
        int[] afterTop2 = r.selectTopK(2);
        boolean stableKept = contains(afterTop2, 0);
        boolean driftingReplaced = !contains(afterTop2, 1) && contains(afterTop2, 3);
        report("S4-like swap: stable kept (0), drifting (1) replaced by new (3); before=" +
                        Arrays.toString(beforeTop2) + ", after=" + Arrays.toString(afterTop2),
                stableKept && driftingReplaced);
    }

    private static boolean contains(int[] a, int x) {
        for (int v : a) if (v == x) return true;
        return false;
    }

    private static void testRareClassDoesNotBreakRanker() {
        for (FilterRanker r : allRankers(3, 4, 3)) {
            Random rnd = new Random(13);
            for (int i = 0; i < 5000; i++) {
                int cls = (rnd.nextInt(100) < 1) ? 2 : rnd.nextInt(2);
                int[] x = new int[]{rnd.nextInt(4), rnd.nextInt(4), rnd.nextInt(4)};
                x[0] = (cls == 0) ? 0 : (cls == 1 ? 3 : 1);
                r.update(x, cls);
            }
            double[] s = r.getFeatureScores();
            boolean finite = true;
            for (double v : s) if (Double.isNaN(v) || Double.isInfinite(v) || v < 0.0) finite = false;
            report(r.name() + " handles rare class without NaN/Inf (s=" + Arrays.toString(s) + ")", finite);
        }
    }

    private static void testEmptyBinsDoNotInflateScore() {
        for (FilterRanker r : allRankers(2, 8, 2)) {
            Random rnd = new Random(14);
            for (int i = 0; i < 2000; i++) {
                int cls = rnd.nextInt(2);
                int[] x = new int[]{cls == 0 ? 0 : 7, rnd.nextInt(2)};
                r.update(x, cls);
            }
            double[] s = r.getFeatureScores();
            report(r.name() + " informative > noise even with many empty bins (s=" + Arrays.toString(s) + ")",
                    s[0] > s[1] + 0.05);
        }
    }

    private static void report(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  [PASSED] " + name); }
        else    { failed++; System.out.println("  [FAILED] " + name); }
    }
}