package thesis.selection;

import thesis.discretization.PiDDiscretizer;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class SelectionSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("SELECTION SMOKE TESTS");
        System.out.println("=".repeat(70));

        testRankerUpdateValidation(new InformationGainRanker(4, 5, 2), "IG");
        testRankerUpdateValidation(new MutualInformationRanker(4, 5, 2), "MI");
        testRankerUpdateValidation(new ChiSquaredRanker(4, 5, 2), "Chi2");

        testRankerSelectTopKShape(new InformationGainRanker(6, 5, 2), "IG");
        testRankerSelectTopKShape(new MutualInformationRanker(6, 5, 2), "MI");
        testRankerSelectTopKShape(new ChiSquaredRanker(6, 5, 2), "Chi2");

        testRankerRanksInformativeAboveNoise(
                new InformationGainRanker(5, 4, 2), "IG");
        testRankerRanksInformativeAboveNoise(
                new MutualInformationRanker(5, 4, 2), "MI");
        testRankerRanksInformativeAboveNoise(
                new ChiSquaredRanker(5, 4, 2), "Chi2");

        testRankerZeroScoreOnEmpty(new InformationGainRanker(3, 4, 2), "IG");
        testRankerZeroScoreOnEmpty(new MutualInformationRanker(3, 4, 2), "MI");
        testRankerZeroScoreOnEmpty(new ChiSquaredRanker(3, 4, 2), "Chi2");

        testRankerResetClearsCounts(new InformationGainRanker(3, 4, 2), "IG");
        testRankerResetClearsCounts(new MutualInformationRanker(3, 4, 2), "MI");
        testRankerResetClearsCounts(new ChiSquaredRanker(3, 4, 2), "Chi2");

        testRankerNonNegativeScores(new InformationGainRanker(4, 4, 3), "IG");
        testRankerNonNegativeScores(new MutualInformationRanker(4, 4, 3), "MI");
        testRankerNonNegativeScores(new ChiSquaredRanker(4, 4, 3), "Chi2");

        testRankerConstructorRejectsBadParams();

        testStaticSelectorInitializeAndFilter();
        testStaticSelectorPicksInformativeFeatures();
        testStaticSelectorDefaultK();
        testStaticSelectorUpdateIsNoop();
        testStaticSelectorThrowsBeforeInit();
        testStaticSelectorRejectsBadConstructor();
        testStaticSelectorRejectsBadInitialWindow();
        testStaticSelectorFilterValidatesLength();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    private static void testRankerUpdateValidation(FilterRanker r, String tag) {
        boolean t1 = false, t2 = false, t3 = false;
        try { r.update(new int[]{0, 0}, 0); } catch (IllegalArgumentException e) { t1 = true; }
        try { r.update(new int[]{0, 0, 0, 0}, -1); } catch (IllegalArgumentException e) { t2 = true; }
        try { r.update(new int[]{0, 0, 0, 99}, 0); } catch (IllegalArgumentException e) { t3 = true; }
        report(tag + " update() validates inputs", t1 && t2 && t3);
    }

    private static void testRankerSelectTopKShape(FilterRanker r, String tag) {
        Random rng = new Random(31);
        int F = r.getNumFeatures();
        for (int i = 0; i < 200; i++) {
            int[] bins = new int[F];
            for (int f = 0; f < F; f++) bins[f] = rng.nextInt(5);
            r.update(bins, rng.nextInt(2));
        }
        int[] top = r.selectTopK(3);
        Set<Integer> seen = new HashSet<>();
        for (int idx : top) seen.add(idx);
        boolean ok = top.length == 3 && seen.size() == 3
                && Arrays.stream(top).allMatch(i -> i >= 0 && i < F);
        report(tag + " selectTopK returns k unique valid indices", ok);

        boolean threw1 = false, threw2 = false;
        try { r.selectTopK(0); } catch (IllegalArgumentException e) { threw1 = true; }
        try { r.selectTopK(F + 1); } catch (IllegalArgumentException e) { threw2 = true; }
        report(tag + " selectTopK rejects bad k", threw1 && threw2);
    }

    private static void testRankerRanksInformativeAboveNoise(FilterRanker r, String tag) {
        Random rng = new Random(32);
        int F = r.getNumFeatures();
        int informative = 0;
        for (int i = 0; i < 4000; i++) {
            int cls = rng.nextInt(2);
            int[] bins = new int[F];
            for (int f = 0; f < F; f++) {
                if (f == informative) {
                    bins[f] = (cls == 0) ? rng.nextInt(2) : 2 + rng.nextInt(2);
                } else {
                    bins[f] = rng.nextInt(4);
                }
            }
            r.update(bins, cls);
        }
        double[] scores = r.getFeatureScores();
        boolean ok = true;
        for (int f = 0; f < F; f++) {
            if (f == informative) continue;
            if (scores[informative] <= scores[f]) { ok = false; break; }
        }
        int top1 = r.selectTopK(1)[0];
        report(tag + " informative feature wins (scores=" + Arrays.toString(scores)
                + ", top=" + top1 + ")", ok && top1 == informative);
    }

    private static void testRankerZeroScoreOnEmpty(FilterRanker r, String tag) {
        double[] s = r.getFeatureScores();
        boolean ok = true;
        for (double v : s) if (v != 0.0) { ok = false; break; }
        report(tag + " zero scores before any update", ok);
    }

    private static void testRankerResetClearsCounts(FilterRanker r, String tag) {
        Random rng = new Random(33);
        int F = r.getNumFeatures();
        for (int i = 0; i < 100; i++) {
            int[] bins = new int[F];
            for (int f = 0; f < F; f++) bins[f] = rng.nextInt(4);
            r.update(bins, rng.nextInt(2));
        }
        r.reset();
        double[] s = r.getFeatureScores();
        boolean ok = true;
        for (double v : s) if (v != 0.0) { ok = false; break; }
        report(tag + " reset() zeros all scores", ok);
    }

    private static void testRankerNonNegativeScores(FilterRanker r, String tag) {
        Random rng = new Random(34);
        int F = r.getNumFeatures();
        for (int i = 0; i < 500; i++) {
            int[] bins = new int[F];
            for (int f = 0; f < F; f++) bins[f] = rng.nextInt(4);
            r.update(bins, rng.nextInt(3));
        }
        double[] s = r.getFeatureScores();
        boolean ok = true;
        for (double v : s) if (v < 0.0 || Double.isNaN(v)) { ok = false; break; }
        report(tag + " all scores >= 0 (" + Arrays.toString(s) + ")", ok);
    }

    private static void testRankerConstructorRejectsBadParams() {
        boolean t1 = false, t2 = false, t3 = false;
        try { new InformationGainRanker(0, 4, 2); } catch (IllegalArgumentException e) { t1 = true; }
        try { new MutualInformationRanker(3, 1, 2); } catch (IllegalArgumentException e) { t2 = true; }
        try { new ChiSquaredRanker(3, 4, 1); } catch (IllegalArgumentException e) { t3 = true; }
        report("Ranker constructors reject bad params", t1 && t2 && t3);
    }

    private static void testStaticSelectorInitializeAndFilter() {
        int F = 6, N = 600;
        Random rng = new Random(35);
        double[][] win = new double[N][F];
        int[] y = new int[N];
        for (int i = 0; i < N; i++) {
            int cls = rng.nextInt(2);
            y[i] = cls;
            for (int f = 0; f < F; f++) {
                win[i][f] = (f == 0)
                        ? (cls == 0 ? rng.nextGaussian() - 2 : rng.nextGaussian() + 2)
                        : rng.nextGaussian();
            }
        }
        StaticFeatureSelector sel = new StaticFeatureSelector(F, 2);
        sel.initialize(win, y);
        int[] selected = sel.getSelectedFeatures();
        double[] filtered = sel.filterInstance(win[0]);
        boolean ok = sel.isInitialized()
                && selected.length == sel.getK()
                && filtered.length == selected.length
                && Arrays.equals(selected, sel.getCurrentSelection());
        report("StaticFeatureSelector initialize + filter works (selected="
                + Arrays.toString(selected) + ")", ok);
    }

    private static void testStaticSelectorPicksInformativeFeatures() {
        int F = 8, N = 1500;
        Random rng = new Random(36);
        double[][] win = new double[N][F];
        int[] y = new int[N];
        Set<Integer> info = new HashSet<>(Arrays.asList(2, 5));
        for (int i = 0; i < N; i++) {
            int cls = rng.nextInt(2);
            y[i] = cls;
            for (int f = 0; f < F; f++) {
                win[i][f] = info.contains(f)
                        ? (cls == 0 ? rng.nextGaussian() - 3 : rng.nextGaussian() + 3)
                        : rng.nextGaussian();
            }
        }
        StaticFeatureSelector sel = new StaticFeatureSelector(F, 2, 2,
                new PiDDiscretizer(F, 2),
                (bins, classes) -> new InformationGainRanker(F, bins, classes));
        sel.initialize(win, y);
        Set<Integer> selected = new HashSet<>();
        for (int idx : sel.getSelectedFeatures()) selected.add(idx);
        report("StaticFeatureSelector selects informative features (got=" + selected + ")",
                selected.equals(info));
    }

    private static void testStaticSelectorDefaultK() {
        boolean ok = StaticFeatureSelector.defaultK(1) == 1
                && StaticFeatureSelector.defaultK(4) == 2
                && StaticFeatureSelector.defaultK(10) == 4
                && StaticFeatureSelector.defaultK(100) == 10;
        report("StaticFeatureSelector.defaultK = ceil(sqrt(F))", ok);
    }

    private static void testStaticSelectorUpdateIsNoop() {
        int F = 4, N = 600;
        Random rng = new Random(37);
        double[][] win = new double[N][F];
        int[] y = new int[N];
        for (int i = 0; i < N; i++) {
            y[i] = rng.nextInt(2);
            for (int f = 0; f < F; f++) win[i][f] = rng.nextGaussian();
        }
        StaticFeatureSelector sel = new StaticFeatureSelector(F, 2);
        sel.initialize(win, y);
        int[] before = sel.getSelectedFeatures();
        for (int i = 0; i < 1000; i++) {
            sel.update(new double[]{rng.nextGaussian(), rng.nextGaussian(),
                            rng.nextGaussian(), rng.nextGaussian()},
                    rng.nextInt(2), false, Collections.emptySet());
        }
        int[] after = sel.getSelectedFeatures();
        report("StaticFeatureSelector.update is no-op", Arrays.equals(before, after));
    }

    private static void testStaticSelectorThrowsBeforeInit() {
        StaticFeatureSelector sel = new StaticFeatureSelector(4, 2);
        boolean t1 = false, t2 = false, t3 = false;
        try { sel.getSelectedFeatures(); } catch (IllegalStateException e) { t1 = true; }
        try { sel.getCurrentSelection(); } catch (IllegalStateException e) { t2 = true; }
        try { sel.filterInstance(new double[]{0, 0, 0, 0}); } catch (IllegalStateException e) { t3 = true; }
        report("StaticFeatureSelector throws before initialize()", t1 && t2 && t3
                && !sel.isInitialized());
    }

    private static void testStaticSelectorRejectsBadConstructor() {
        boolean t1 = false, t2 = false, t3 = false, t4 = false;
        try { new StaticFeatureSelector(0, 2); } catch (IllegalArgumentException e) { t1 = true; }
        try { new StaticFeatureSelector(4, 1); } catch (IllegalArgumentException e) { t2 = true; }
        try { new StaticFeatureSelector(4, 2, 0,
                new PiDDiscretizer(4, 2),
                (b, c) -> new InformationGainRanker(4, b, c)); }
        catch (IllegalArgumentException e) { t3 = true; }
        try { new StaticFeatureSelector(4, 2, 5,
                new PiDDiscretizer(4, 2),
                (b, c) -> new InformationGainRanker(4, b, c)); }
        catch (IllegalArgumentException e) { t4 = true; }
        report("StaticFeatureSelector rejects bad constructor params",
                t1 && t2 && t3 && t4);
    }

    private static void testStaticSelectorRejectsBadInitialWindow() {
        StaticFeatureSelector sel = new StaticFeatureSelector(4, 2);
        boolean t1 = false, t2 = false, t3 = false;
        try { sel.initialize(null, new int[]{0}); } catch (IllegalArgumentException e) { t1 = true; }
        try { sel.initialize(new double[][]{{0,0,0,0}}, new int[]{0,1}); }
        catch (IllegalArgumentException e) { t2 = true; }
        try { sel.initialize(new double[][]{{0,0,0}}, new int[]{0}); }
        catch (IllegalArgumentException e) { t3 = true; }
        report("StaticFeatureSelector rejects bad initial window", t1 && t2 && t3);
    }

    private static void testStaticSelectorFilterValidatesLength() {
        int F = 5, N = 600;
        Random rng = new Random(38);
        double[][] win = new double[N][F];
        int[] y = new int[N];
        for (int i = 0; i < N; i++) {
            y[i] = rng.nextInt(2);
            for (int f = 0; f < F; f++) win[i][f] = rng.nextGaussian();
        }
        StaticFeatureSelector sel = new StaticFeatureSelector(F, 2);
        sel.initialize(win, y);
        boolean threw = false;
        try { sel.filterInstance(new double[]{0, 0}); }
        catch (IllegalArgumentException e) { threw = true; }
        report("StaticFeatureSelector.filterInstance validates length", threw);
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