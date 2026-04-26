package thesis.selection;

import thesis.discretization.PiDDiscretizer;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class PeriodicSelectorSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("PERIODIC SELECTOR SMOKE TESTS");
        System.out.println("=".repeat(70));

        testInitializeSetsSelection();
        testThrowsBeforeInit();
        testRejectsBadConstructor();
        testRejectsBadInitialWindow();
        testFilterInstanceShapeAndValidation();
        testNoSwapWhenStreamIsStationary();
        testSwapsTowardNewInformativeFeatures();
        testReSelectionFiresOnPeriodBoundary();
        testMaxSwapsPerCycleRespected();
        testMinTenureBlocksImmediateEviction();
        testUpdateBeforeInitIsNoop();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    private static double[][] makeWindow(int N, int F, long seed,
                                         Set<Integer> informative, int[] outLabels) {
        Random rng = new Random(seed);
        double[][] win = new double[N][F];
        for (int i = 0; i < N; i++) {
            int cls = rng.nextInt(2);
            outLabels[i] = cls;
            for (int f = 0; f < F; f++) {
                win[i][f] = informative.contains(f)
                        ? (cls == 0 ? rng.nextGaussian() - 3 : rng.nextGaussian() + 3)
                        : rng.nextGaussian();
            }
        }
        return win;
    }

    private static PeriodicSelector buildSelector(int F, int K, int periodN, int minTenure) {
        return new PeriodicSelector(F, 2, K, periodN, minTenure,
                new PiDDiscretizer(F, 2),
                (bins, classes) -> new InformationGainRanker(F, bins, classes));
    }

    private static void testInitializeSetsSelection() {
        int F = 6, N = 600, K = 2;
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 201, new HashSet<>(Arrays.asList(1, 4)), y);
        PeriodicSelector sel = buildSelector(F, K, 200, 100);
        sel.initialize(win, y);
        int[] s = sel.getSelectedFeatures();
        boolean ok = sel.isInitialized()
                && s.length == K
                && Arrays.equals(s, sel.getCurrentSelection())
                && sel.getSwapEvents() == 0
                && sel.getSwappedFeatures() == 0;
        report("initialize sets selection (got=" + Arrays.toString(s) + ")", ok);
    }

    private static void testThrowsBeforeInit() {
        PeriodicSelector sel = buildSelector(4, 2, 200, 50);
        boolean t1 = false, t2 = false, t3 = false;
        try { sel.getSelectedFeatures(); } catch (IllegalStateException e) { t1 = true; }
        try { sel.getCurrentSelection(); } catch (IllegalStateException e) { t2 = true; }
        try { sel.filterInstance(new double[]{0, 0, 0, 0}); } catch (IllegalStateException e) { t3 = true; }
        report("throws before initialize()", t1 && t2 && t3 && !sel.isInitialized());
    }

    private static void testRejectsBadConstructor() {
        boolean t1 = false, t2 = false, t3 = false, t4 = false, t5 = false, t6 = false;
        try { new PeriodicSelector(0, 2); } catch (IllegalArgumentException e) { t1 = true; }
        try { new PeriodicSelector(4, 1); } catch (IllegalArgumentException e) { t2 = true; }
        try { new PeriodicSelector(4, 2, 0, 200, 50,
                new PiDDiscretizer(4, 2),
                (b, c) -> new InformationGainRanker(4, b, c)); }
        catch (IllegalArgumentException e) { t3 = true; }
        try { new PeriodicSelector(4, 2, 5, 200, 50,
                new PiDDiscretizer(4, 2),
                (b, c) -> new InformationGainRanker(4, b, c)); }
        catch (IllegalArgumentException e) { t4 = true; }
        try { new PeriodicSelector(4, 2, 2, 50, 50,
                new PiDDiscretizer(4, 2),
                (b, c) -> new InformationGainRanker(4, b, c)); }
        catch (IllegalArgumentException e) { t5 = true; }
        try { new PeriodicSelector(4, 2, 2, 200, -1,
                new PiDDiscretizer(4, 2),
                (b, c) -> new InformationGainRanker(4, b, c)); }
        catch (IllegalArgumentException e) { t6 = true; }
        report("constructor rejects bad params", t1 && t2 && t3 && t4 && t5 && t6);
    }

    private static void testRejectsBadInitialWindow() {
        PeriodicSelector sel = buildSelector(4, 2, 200, 50);
        boolean t1 = false, t2 = false, t3 = false;
        try { sel.initialize(null, new int[]{0}); } catch (IllegalArgumentException e) { t1 = true; }
        try { sel.initialize(new double[][]{{0,0,0,0}}, new int[]{0,1}); }
        catch (IllegalArgumentException e) { t2 = true; }
        try { sel.initialize(new double[][]{{0,0,0}}, new int[]{0}); }
        catch (IllegalArgumentException e) { t3 = true; }
        report("rejects bad initial window", t1 && t2 && t3);
    }

    private static void testFilterInstanceShapeAndValidation() {
        int F = 5, N = 600, K = 3;
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 202, new HashSet<>(Arrays.asList(2)), y);
        PeriodicSelector sel = buildSelector(F, K, 200, 50);
        sel.initialize(win, y);
        double[] out = sel.filterInstance(win[0]);
        boolean shapeOk = out.length == K;
        boolean threw = false;
        try { sel.filterInstance(new double[]{0, 0}); }
        catch (IllegalArgumentException e) { threw = true; }
        report("filterInstance shape + length validation", shapeOk && threw);
    }

    private static void testNoSwapWhenStreamIsStationary() {
        int F = 6, K = 2, N = 600, P = 200;
        int[] y = new int[N];
        Set<Integer> info = new HashSet<>(Arrays.asList(1, 4));
        double[][] win = makeWindow(N, F, 203, info, y);
        PeriodicSelector sel = buildSelector(F, K, P, 0);
        sel.initialize(win, y);
        int[] before = sel.getSelectedFeatures();

        Random rng = new Random(303);
        for (int i = 0; i < 5 * P; i++) {
            int cls = rng.nextInt(2);
            double[] x = new double[F];
            for (int f = 0; f < F; f++) {
                x[f] = info.contains(f)
                        ? (cls == 0 ? rng.nextGaussian() - 3 : rng.nextGaussian() + 3)
                        : rng.nextGaussian();
            }
            sel.update(x, cls, false, Collections.emptySet());
        }
        int[] after = sel.getSelectedFeatures();
        report("no swaps on stationary stream (events=" + sel.getSwapEvents() + ")",
                Arrays.equals(before, after) && sel.getSwapEvents() == 0);
    }

    private static void testSwapsTowardNewInformativeFeatures() {
        int F = 6, K = 2, N = 600, P = 300;
        int[] y = new int[N];
        Set<Integer> oldInfo = new HashSet<>(Arrays.asList(0, 1));
        double[][] win = makeWindow(N, F, 204, oldInfo, y);
        PeriodicSelector sel = buildSelector(F, K, P, 0);
        sel.initialize(win, y);
        Set<Integer> initialSel = new HashSet<>();
        for (int idx : sel.getSelectedFeatures()) initialSel.add(idx);

        Set<Integer> newInfo = new HashSet<>(Arrays.asList(4, 5));
        Random rng = new Random(404);
        for (int i = 0; i < 4 * P; i++) {
            int cls = rng.nextInt(2);
            double[] x = new double[F];
            for (int f = 0; f < F; f++) {
                x[f] = newInfo.contains(f)
                        ? (cls == 0 ? rng.nextGaussian() - 3 : rng.nextGaussian() + 3)
                        : rng.nextGaussian();
            }
            sel.update(x, cls, false, Collections.emptySet());
        }
        Set<Integer> finalSel = new HashSet<>();
        for (int idx : sel.getSelectedFeatures()) finalSel.add(idx);
        boolean migrated = finalSel.contains(4) || finalSel.contains(5);
        report("selection migrates toward new informative features ("
                        + initialSel + " -> " + finalSel + ", events=" + sel.getSwapEvents() + ")",
                migrated && sel.getSwapEvents() >= 1);
    }

    private static void testReSelectionFiresOnPeriodBoundary() {
        int F = 6, K = 2, N = 600, P = 200;
        int[] y = new int[N];
        Set<Integer> oldInfo = new HashSet<>(Arrays.asList(0, 1));
        double[][] win = makeWindow(N, F, 205, oldInfo, y);
        PeriodicSelector sel = buildSelector(F, K, P, 0);
        sel.initialize(win, y);

        Set<Integer> newInfo = new HashSet<>(Arrays.asList(4, 5));
        Random rng = new Random(505);
        long beforeEvents = sel.getSwapEvents();
        for (int i = 0; i < P - 1; i++) {
            int cls = rng.nextInt(2);
            double[] x = new double[F];
            for (int f = 0; f < F; f++) {
                x[f] = newInfo.contains(f)
                        ? (cls == 0 ? rng.nextGaussian() - 3 : rng.nextGaussian() + 3)
                        : rng.nextGaussian();
            }
            sel.update(x, cls, false, Collections.emptySet());
        }
        boolean midOk = sel.getSwapEvents() == beforeEvents;

        int cls = rng.nextInt(2);
        double[] x = new double[F];
        for (int f = 0; f < F; f++) {
            x[f] = newInfo.contains(f)
                    ? (cls == 0 ? rng.nextGaussian() - 3 : rng.nextGaussian() + 3)
                    : rng.nextGaussian();
        }
        sel.update(x, cls, false, Collections.emptySet());
        boolean fired = sel.getSwapEvents() == beforeEvents + 1;
        report("re-selection fires exactly on period boundary", midOk && fired);
    }

    private static void testMaxSwapsPerCycleRespected() {
        int F = 10, K = 6, P = 300;
        int N = 600;
        int[] y = new int[N];
        Set<Integer> oldInfo = new HashSet<>(Arrays.asList(0, 1, 2, 3, 4, 5));
        double[][] win = makeWindow(N, F, 206, oldInfo, y);
        PeriodicSelector sel = buildSelector(F, K, P, 0);
        sel.initialize(win, y);
        int expectedMax = (int) Math.ceil(K * 0.3);

        Set<Integer> newInfo = new HashSet<>(Arrays.asList(6, 7, 8, 9));
        Random rng = new Random(606);
        for (int i = 0; i < P; i++) {
            int cls = rng.nextInt(2);
            double[] x = new double[F];
            for (int f = 0; f < F; f++) {
                x[f] = newInfo.contains(f)
                        ? (cls == 0 ? rng.nextGaussian() - 3 : rng.nextGaussian() + 3)
                        : rng.nextGaussian();
            }
            sel.update(x, cls, false, Collections.emptySet());
        }
        long swapped = sel.getSwappedFeatures();
        report("first cycle swaps <= maxSwapsPerCycle=" + expectedMax + " (got=" + swapped + ")",
                sel.getSwapEvents() == 1 && swapped >= 1 && swapped <= expectedMax);
    }

    private static void testMinTenureBlocksImmediateEviction() {
        int F = 6, K = 2, P = 300, N = 600;

        int[] y = new int[N];
        Set<Integer> phase1Info = new HashSet<>(Arrays.asList(0, 1));
        double[][] win = makeWindow(N, F, 207, phase1Info, y);

        int minTenure = 2 * P;
        PeriodicSelector sel = buildSelector(F, K, P, minTenure);
        sel.initialize(win, y);

        Random rng = new Random(707);

        Set<Integer> phase2Info = new HashSet<>(Arrays.asList(4, 5));
        for (int i = 0; i < P; i++) {
            int cls = rng.nextInt(2);
            double[] x = new double[F];
            for (int f = 0; f < F; f++) {
                x[f] = phase2Info.contains(f)
                        ? (cls == 0 ? rng.nextGaussian() - 3 : rng.nextGaussian() + 3)
                        : rng.nextGaussian();
            }
            sel.update(x, cls, false, Collections.emptySet());
        }
        Set<Integer> afterPhase2 = new HashSet<>();
        for (int idx : sel.getSelectedFeatures()) afterPhase2.add(idx);
        boolean migrated = afterPhase2.contains(4) || afterPhase2.contains(5);
        long eventsAfterPhase2 = sel.getSwapEvents();

        Set<Integer> phase3Info = new HashSet<>(Arrays.asList(2, 3));
        for (int i = 0; i < P; i++) {
            int cls = rng.nextInt(2);
            double[] x = new double[F];
            for (int f = 0; f < F; f++) {
                x[f] = phase3Info.contains(f)
                        ? (cls == 0 ? rng.nextGaussian() - 3 : rng.nextGaussian() + 3)
                        : rng.nextGaussian();
            }
            sel.update(x, cls, false, Collections.emptySet());
        }
        Set<Integer> afterPhase3 = new HashSet<>();
        for (int idx : sel.getSelectedFeatures()) afterPhase3.add(idx);

        boolean newcomerProtected = afterPhase3.containsAll(
                afterPhase2.stream().filter(i -> i == 4 || i == 5)
                        .collect(java.util.stream.Collectors.toSet()));

        report("minTenure protects newly swapped-in features from immediate eviction "
                        + "(phase2=" + afterPhase2 + ", phase3=" + afterPhase3 + ")",
                migrated && newcomerProtected);
    }

    private static void testUpdateBeforeInitIsNoop() {
        PeriodicSelector sel = buildSelector(4, 2, 200, 50);
        sel.update(new double[]{0, 0, 0, 0}, 0, false, Collections.emptySet());
        report("update() before initialize is silent no-op", !sel.isInitialized());
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