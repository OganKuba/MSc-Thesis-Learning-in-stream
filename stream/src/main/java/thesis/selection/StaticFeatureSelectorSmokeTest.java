package thesis.selection;

import thesis.discretization.PiDDiscretizer;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class StaticFeatureSelectorSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("STATIC FEATURE SELECTOR SMOKE TESTS");
        System.out.println("=".repeat(70));

        testRejectsBadConstructorArgs();
        testRejectsDiscretizerMismatch();
        testInitializeIdentityProducesIdentitySelection();
        testInitializeWithExplicitSelection();
        testInitializeWithRejectsBadSelection();
        testInitializeFromWindowPicksInformativeFeature();
        testInitializeFromWindowPicksTopK();
        testInitializeIsDeterministicAcrossRuns();
        testSelectionFrozenAfterInitialize();
        testUpdateDoesNotChangeSelection();
        testUpdateValidatesShapeAndLabel();
        testFilterInstancePreservesOrderOfSelection();
        testFilterInstanceRejectsBadShape();
        testInitializeRejectsEmptyAndShapeMismatches();
        testInitializeIgnoresNonFiniteRows();
        testInitializeRefusesIfDiscretizerNotReady();
        testInitializeRefusesDoubleInit();
        testInitialScoresAreReturned();
        testGetSelectedFeaturesReturnsCopy();
        testRankerFactoryGetsCorrectNumFeatures();
        testWorksWithAllThreeRankers();
        testS1IsDeterministicallySortedSelection();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    private static double[][] genWindow(int n, int F, int informativeIdx, long seed) {
        Random r = new Random(seed);
        double[][] w = new double[n][F];
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) {
            int cls = r.nextInt(2);
            labels[i] = cls;
            for (int f = 0; f < F; f++) w[i][f] = r.nextGaussian();
            w[i][informativeIdx] = (cls == 0 ? -3.0 : 3.0) + r.nextGaussian() * 0.3;
        }
        return w;
    }

    private static int[] genLabels(double[][] window, int informativeIdx, long seed) {
        int[] labels = new int[window.length];
        for (int i = 0; i < window.length; i++) labels[i] = window[i][informativeIdx] > 0 ? 1 : 0;
        return labels;
    }

    private static StaticFeatureSelector buildSelector(int F, int K, int B2) {
        PiDDiscretizer pid = new PiDDiscretizer(F, 2, 32, B2, 200, 500);
        return new StaticFeatureSelector(F, 2, K, pid,
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc, 1));
    }

    private static void testRejectsBadConstructorArgs() {
        boolean t1 = false, t2 = false, t3 = false;
        try { new StaticFeatureSelector(0, 2); } catch (IllegalArgumentException e) { t1 = true; }
        try { new StaticFeatureSelector(5, 1); } catch (IllegalArgumentException e) { t2 = true; }
        try {
            new StaticFeatureSelector(5, 2, 6, new PiDDiscretizer(5, 2),
                    (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc, 1));
        } catch (IllegalArgumentException e) { t3 = true; }
        report("rejects bad constructor args", t1 && t2 && t3);
    }

    private static void testRejectsDiscretizerMismatch() {
        boolean t1 = false, t2 = false;
        try {
            new StaticFeatureSelector(5, 2, 2, new PiDDiscretizer(4, 2),
                    (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc, 1));
        } catch (IllegalArgumentException e) { t1 = true; }
        try {
            new StaticFeatureSelector(5, 2, 2, new PiDDiscretizer(5, 3),
                    (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc, 1));
        } catch (IllegalArgumentException e) { t2 = true; }
        report("rejects discretizer mismatch", t1 && t2);
    }

    private static void testInitializeIdentityProducesIdentitySelection() {
        StaticFeatureSelector s = buildSelector(8, 3, 4);
        s.initializeIdentity();
        report("initializeIdentity selects [0,1,2]",
                Arrays.equals(s.getSelectedFeatures(), new int[]{0, 1, 2}));
    }

    private static void testInitializeWithExplicitSelection() {
        StaticFeatureSelector s = buildSelector(8, 3, 4);
        s.initializeWith(new int[]{5, 1, 7});
        report("initializeWith stores sorted explicit selection",
                Arrays.equals(s.getSelectedFeatures(), new int[]{1, 5, 7}));
    }

    private static void testInitializeWithRejectsBadSelection() {
        StaticFeatureSelector s = buildSelector(8, 3, 4);
        boolean t1 = false, t2 = false, t3 = false;
        try { s.initializeWith(new int[]{0, 1}); } catch (IllegalArgumentException e) { t1 = true; }
        try { s.initializeWith(new int[]{0, 1, 8}); } catch (IllegalArgumentException e) { t2 = true; }
        try { s.initializeWith(new int[]{0, 1, 1}); } catch (IllegalArgumentException e) { t3 = true; }
        report("initializeWith rejects bad selection", t1 && t2 && t3);
    }

    private static void testInitializeFromWindowPicksInformativeFeature() {
        int F = 6;
        StaticFeatureSelector s = buildSelector(F, 1, 4);
        double[][] win = genWindow(400, F, 3, 42);
        int[] labels = new int[win.length];
        for (int i = 0; i < win.length; i++) labels[i] = win[i][3] > 0 ? 1 : 0;
        s.initialize(win, labels);
        int[] sel = s.getSelectedFeatures();
        report("S1 picks informative feature 3 as top-1 (sel=" + Arrays.toString(sel) + ")",
                sel.length == 1 && sel[0] == 3);
    }

    private static void testInitializeFromWindowPicksTopK() {
        int F = 8, K = 3;
        StaticFeatureSelector s = buildSelector(F, K, 4);
        Random r = new Random(123);
        int n = 800;
        double[][] win = new double[n][F];
        int[] labels = new int[n];
        int[] informative = {1, 4, 6};
        for (int i = 0; i < n; i++) {
            int cls = r.nextInt(2);
            labels[i] = cls;
            for (int f = 0; f < F; f++) win[i][f] = r.nextGaussian();
            for (int idx : informative) win[i][idx] = (cls == 0 ? -3.0 : 3.0) + r.nextGaussian() * 0.3;
        }
        s.initialize(win, labels);
        int[] sel = s.getSelectedFeatures();
        Set<Integer> selSet = new HashSet<>();
        for (int x : sel) selSet.add(x);
        boolean ok = sel.length == K
                && selSet.contains(1) && selSet.contains(4) && selSet.contains(6);
        report("S1 selects top-K informative features (sel=" + Arrays.toString(sel) + ")", ok);
    }

    private static void testInitializeIsDeterministicAcrossRuns() {
        int F = 8;
        StaticFeatureSelector a = buildSelector(F, 3, 4);
        StaticFeatureSelector b = buildSelector(F, 3, 4);
        double[][] win = genWindow(500, F, 2, 7);
        int[] labels = genLabels(win, 2, 7);
        a.initialize(win, labels);
        b.initialize(win, labels);
        report("S1 deterministic on identical input",
                Arrays.equals(a.getSelectedFeatures(), b.getSelectedFeatures()));
    }

    private static void testSelectionFrozenAfterInitialize() {
        int F = 6;
        StaticFeatureSelector s = buildSelector(F, 2, 4);
        double[][] win = genWindow(400, F, 1, 11);
        int[] labels = genLabels(win, 1, 11);
        s.initialize(win, labels);
        int[] before = s.getSelectedFeatures();
        Random r = new Random(99);
        for (int i = 0; i < 10000; i++) {
            double[] inst = new double[F];
            for (int f = 0; f < F; f++) inst[f] = r.nextGaussian();
            int cls = r.nextInt(2);
            boolean alarm = r.nextInt(50) == 0;
            Set<Integer> drift = alarm ? new HashSet<>(Arrays.asList(0, 2, 4)) : Collections.emptySet();
            s.update(inst, cls, alarm, drift);
        }
        int[] after = s.getSelectedFeatures();
        report("S1 selection frozen across many updates with drift alarms",
                Arrays.equals(before, after));
    }

    private static void testUpdateDoesNotChangeSelection() {
        int F = 4;
        StaticFeatureSelector s = buildSelector(F, 2, 4);
        s.initializeIdentity();
        int[] before = s.getSelectedFeatures();
        s.update(new double[]{1, 2, 3, 4}, 0, true, new HashSet<>(Arrays.asList(0, 1, 2, 3)));
        int[] after = s.getSelectedFeatures();
        report("update is a no-op on selection", Arrays.equals(before, after));
    }

    private static void testUpdateValidatesShapeAndLabel() {
        StaticFeatureSelector s = buildSelector(4, 2, 4);
        s.initializeIdentity();
        boolean t1 = false, t2 = false, t3 = false;
        try { s.update(new double[]{1, 2}, 0, false, Collections.emptySet()); }
        catch (IllegalArgumentException e) { t1 = true; }
        try { s.update(null, 0, false, Collections.emptySet()); }
        catch (IllegalArgumentException e) { t2 = true; }
        try { s.update(new double[]{1, 2, 3, 4}, 7, false, Collections.emptySet()); }
        catch (IllegalArgumentException e) { t3 = true; }
        report("update validates shape and label", t1 && t2 && t3);
    }

    private static void testFilterInstancePreservesOrderOfSelection() {
        StaticFeatureSelector s = buildSelector(6, 3, 4);
        s.initializeWith(new int[]{4, 0, 2});
        double[] x = new double[]{10, 11, 12, 13, 14, 15};
        double[] f = s.filterInstance(x);
        report("filterInstance applies sorted selection (got=" + Arrays.toString(f) + ")",
                Arrays.equals(f, new double[]{10, 12, 14}));
    }

    private static void testFilterInstanceRejectsBadShape() {
        StaticFeatureSelector s = buildSelector(4, 2, 4);
        s.initializeIdentity();
        boolean t1 = false, t2 = false;
        try { s.filterInstance(new double[]{1, 2}); } catch (IllegalArgumentException e) { t1 = true; }
        try { s.filterInstance(null); } catch (IllegalArgumentException e) { t2 = true; }
        report("filterInstance rejects bad shape", t1 && t2);
    }

    private static void testInitializeRejectsEmptyAndShapeMismatches() {
        StaticFeatureSelector s = buildSelector(4, 2, 4);
        boolean t1 = false, t2 = false, t3 = false, t4 = false;
        try { s.initialize(new double[0][4], new int[0]); } catch (IllegalArgumentException e) { t1 = true; }
        try { s.initialize(new double[][]{{1, 2, 3, 4}}, new int[]{0, 1}); } catch (IllegalArgumentException e) { t2 = true; }
        try { s.initialize(new double[][]{{1, 2}}, new int[]{0}); } catch (IllegalArgumentException e) { t3 = true; }
        try { s.initialize(new double[][]{{1, 2, 3, 4}}, new int[]{5}); } catch (IllegalArgumentException e) { t4 = true; }
        report("initialize rejects empty and shape mismatches", t1 && t2 && t3 && t4);
    }

    private static void testInitializeIgnoresNonFiniteRows() {
        int F = 4;
        StaticFeatureSelector s = buildSelector(F, 2, 4);
        double[][] win = genWindow(400, F, 0, 5);
        int[] labels = genLabels(win, 0, 5);
        win[10][1] = Double.NaN;
        win[20][2] = Double.POSITIVE_INFINITY;
        s.initialize(win, labels);
        report("initialize ignores non-finite rows (ignored=" + s.getIgnoredNonFiniteRows() + ")",
                s.isInitialized() && s.getIgnoredNonFiniteRows() == 2);
    }

    private static void testInitializeRefusesIfDiscretizerNotReady() {
        int F = 4;
        PiDDiscretizer pid = new PiDDiscretizer(F, 2, 32, 4, 1000, 500);
        StaticFeatureSelector s = new StaticFeatureSelector(F, 2, 2, pid,
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc, 1));
        double[][] win = genWindow(50, F, 0, 6);
        int[] labels = genLabels(win, 0, 6);
        boolean threw = false;
        try { s.initialize(win, labels); } catch (IllegalStateException e) { threw = true; }
        report("initialize refuses when discretizer not ready", threw && !s.isInitialized());
    }

    private static void testInitializeRefusesDoubleInit() {
        StaticFeatureSelector s = buildSelector(4, 2, 4);
        s.initializeIdentity();
        boolean threw = false;
        try { s.initializeIdentity(); } catch (IllegalStateException e) { threw = true; }
        boolean threw2 = false;
        try { s.initialize(new double[][]{{1, 2, 3, 4}}, new int[]{0}); } catch (IllegalStateException e) { threw2 = true; }
        report("initialize refuses double init", threw && threw2);
    }

    private static void testInitialScoresAreReturned() {
        int F = 5;
        StaticFeatureSelector s = buildSelector(F, 2, 4);
        double[][] win = genWindow(400, F, 2, 8);
        int[] labels = genLabels(win, 2, 8);
        s.initialize(win, labels);
        double[] scores = s.getInitialScores();
        boolean ok = scores != null && scores.length == F;
        if (ok) {
            int argmax = 0;
            for (int i = 1; i < F; i++) if (scores[i] > scores[argmax]) argmax = i;
            ok = argmax == 2;
        }
        report("initial scores returned and informative feature has max score", ok);
    }

    private static void testGetSelectedFeaturesReturnsCopy() {
        StaticFeatureSelector s = buildSelector(5, 2, 4);
        s.initializeIdentity();
        int[] a = s.getSelectedFeatures();
        a[0] = -1;
        int[] b = s.getSelectedFeatures();
        report("getSelectedFeatures returns defensive copy", b[0] == 0);
    }

    private static void testRankerFactoryGetsCorrectNumFeatures() {
        int F = 7;
        int[] capturedNumFeatures = new int[1];
        PiDDiscretizer pid = new PiDDiscretizer(F, 2, 32, 4, 200, 500);
        StaticFeatureSelector s = new StaticFeatureSelector(F, 2, 2, pid,
                (nf, nb, nc) -> {
                    capturedNumFeatures[0] = nf;
                    return new InformationGainRanker(nf, nb, nc, 1);
                });
        double[][] win = genWindow(400, F, 0, 9);
        int[] labels = genLabels(win, 0, 9);
        s.initialize(win, labels);
        report("rankerFactory receives correct numFeatures (got=" + capturedNumFeatures[0] + ")",
                capturedNumFeatures[0] == F);
    }

    private static void testWorksWithAllThreeRankers() {
        int F = 6;
        int[] informativeForEach = new int[3];
        FeatureSelector.RankerFactory[] factories = new FeatureSelector.RankerFactory[]{
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc, 1),
                (nf, nb, nc) -> new MutualInformationRanker(nf, nb, nc, 1),
                (nf, nb, nc) -> new ChiSquaredRanker(nf, nb, nc, 1, true)
        };
        for (int t = 0; t < factories.length; t++) {
            PiDDiscretizer pid = new PiDDiscretizer(F, 2, 32, 4, 200, 500);
            StaticFeatureSelector s = new StaticFeatureSelector(F, 2, 1, pid, factories[t]);
            double[][] win = genWindow(400, F, 4, 100 + t);
            int[] labels = genLabels(win, 4, 100 + t);
            s.initialize(win, labels);
            informativeForEach[t] = s.getSelectedFeatures()[0];
        }
        boolean ok = informativeForEach[0] == 4 && informativeForEach[1] == 4 && informativeForEach[2] == 4;
        report("all three rankers pick informative feature 4 (got=" +
                Arrays.toString(informativeForEach) + ")", ok);
    }

    private static void testS1IsDeterministicallySortedSelection() {
        StaticFeatureSelector s = buildSelector(8, 4, 4);
        Random r = new Random(31);
        int n = 600;
        double[][] win = new double[n][8];
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) {
            int cls = r.nextInt(2);
            labels[i] = cls;
            for (int f = 0; f < 8; f++) win[i][f] = r.nextGaussian();
            win[i][7] = (cls == 0 ? -3 : 3); win[i][2] = (cls == 0 ? -3 : 3);
            win[i][5] = (cls == 0 ? -3 : 3); win[i][0] = (cls == 0 ? -3 : 3);
        }
        s.initialize(win, labels);
        int[] sel = s.getSelectedFeatures();
        boolean sorted = true;
        for (int i = 1; i < sel.length; i++) if (sel[i] < sel[i - 1]) sorted = false;
        report("S1 selection is sorted ascending (sel=" + Arrays.toString(sel) + ")", sorted);
    }

    private static void report(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  [PASSED] " + name); }
        else    { failed++; System.out.println("  [FAILED] " + name); }
    }
}