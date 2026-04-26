package thesis.selection;

import thesis.discretization.PiDDiscretizer;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class DriftAwareSelectorSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("DRIFT-AWARE SELECTOR SMOKE TESTS");
        System.out.println("=".repeat(70));

        testInitializeSetsSelection();
        testThrowsBeforeInit();
        testRejectsBadConstructor();
        testRejectsBadInitialWindow();
        testFilterInstanceShapeAndValidation();
        testNoSwapsOnStationaryNoAlarm();
        testPeriodicReSelectionMigratesAfterDrift();
        testAlarmDrivenSwapAfterWPostDrift();
        testAlarmIgnoredWhenNoDriftingFeaturesGiven();
        testAlarmRespectsMinTenure();
        testRepeatedAlarmDuringCollectionIsIgnored();
        testInCandidatesExcludeDriftingFeatures();
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

    private static DriftAwareSelector buildSelector(int F, int K, int periodN,
                                                    int minTenure, int wPostDrift) {
        return new DriftAwareSelector(F, 2, K, periodN, minTenure, wPostDrift,
                new PiDDiscretizer(F, 2),
                (bins, classes) -> new InformationGainRanker(F, bins, classes));
    }

    private static double[] sample(int F, Set<Integer> info, int cls, Random rng) {
        double[] x = new double[F];
        for (int f = 0; f < F; f++) {
            x[f] = info.contains(f)
                    ? (cls == 0 ? rng.nextGaussian() - 3 : rng.nextGaussian() + 3)
                    : rng.nextGaussian();
        }
        return x;
    }

    private static void testInitializeSetsSelection() {
        int F = 6, N = 600, K = 2;
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 901, new HashSet<>(Arrays.asList(1, 4)), y);
        DriftAwareSelector sel = buildSelector(F, K, 200, 100, 100);
        sel.initialize(win, y);
        int[] s = sel.getSelectedFeatures();
        boolean ok = sel.isInitialized()
                && s.length == K
                && Arrays.equals(s, sel.getCurrentSelection())
                && sel.getPeriodicSwapEvents() == 0
                && sel.getAlarmSwapEvents() == 0
                && !sel.isCollecting();
        report("initialize sets selection (got=" + Arrays.toString(s) + ")", ok);
    }

    private static void testThrowsBeforeInit() {
        DriftAwareSelector sel = buildSelector(4, 2, 200, 50, 100);
        boolean t1 = false, t2 = false, t3 = false;
        try { sel.getSelectedFeatures(); } catch (IllegalStateException e) { t1 = true; }
        try { sel.getCurrentSelection(); } catch (IllegalStateException e) { t2 = true; }
        try { sel.filterInstance(new double[]{0, 0, 0, 0}); } catch (IllegalStateException e) { t3 = true; }
        report("throws before initialize()", t1 && t2 && t3 && !sel.isInitialized());
    }

    private static void testRejectsBadConstructor() {
        boolean t1 = false, t2 = false, t3 = false, t4 = false, t5 = false, t6 = false, t7 = false;
        try { new DriftAwareSelector(0, 2); } catch (IllegalArgumentException e) { t1 = true; }
        try { new DriftAwareSelector(4, 1); } catch (IllegalArgumentException e) { t2 = true; }
        try { new DriftAwareSelector(4, 2, 0, 200, 50, 100,
                new PiDDiscretizer(4, 2),
                (b, c) -> new InformationGainRanker(4, b, c)); }
        catch (IllegalArgumentException e) { t3 = true; }
        try { new DriftAwareSelector(4, 2, 5, 200, 50, 100,
                new PiDDiscretizer(4, 2),
                (b, c) -> new InformationGainRanker(4, b, c)); }
        catch (IllegalArgumentException e) { t4 = true; }
        try { new DriftAwareSelector(4, 2, 2, 50, 50, 100,
                new PiDDiscretizer(4, 2),
                (b, c) -> new InformationGainRanker(4, b, c)); }
        catch (IllegalArgumentException e) { t5 = true; }
        try { new DriftAwareSelector(4, 2, 2, 200, -1, 100,
                new PiDDiscretizer(4, 2),
                (b, c) -> new InformationGainRanker(4, b, c)); }
        catch (IllegalArgumentException e) { t6 = true; }
        try { new DriftAwareSelector(4, 2, 2, 200, 50, 10,
                new PiDDiscretizer(4, 2),
                (b, c) -> new InformationGainRanker(4, b, c)); }
        catch (IllegalArgumentException e) { t7 = true; }
        report("constructor rejects bad params",
                t1 && t2 && t3 && t4 && t5 && t6 && t7);
    }

    private static void testRejectsBadInitialWindow() {
        DriftAwareSelector sel = buildSelector(4, 2, 200, 50, 100);
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
        double[][] win = makeWindow(N, F, 902, new HashSet<>(Arrays.asList(2)), y);
        DriftAwareSelector sel = buildSelector(F, K, 200, 50, 100);
        sel.initialize(win, y);
        double[] out = sel.filterInstance(win[0]);
        boolean shapeOk = out.length == K;
        boolean threw = false;
        try { sel.filterInstance(new double[]{0, 0}); }
        catch (IllegalArgumentException e) { threw = true; }
        report("filterInstance shape + length validation", shapeOk && threw);
    }

    private static void testNoSwapsOnStationaryNoAlarm() {
        int F = 6, K = 2, N = 600, P = 200;
        Set<Integer> info = new HashSet<>(Arrays.asList(1, 4));
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 903, info, y);
        DriftAwareSelector sel = buildSelector(F, K, P, 0, 100);
        sel.initialize(win, y);
        int[] before = sel.getSelectedFeatures();

        Random rng = new Random(303);
        for (int i = 0; i < 5 * P; i++) {
            int cls = rng.nextInt(2);
            sel.update(sample(F, info, cls, rng), cls, false, Collections.emptySet());
        }
        int[] after = sel.getSelectedFeatures();
        boolean ok = Arrays.equals(before, after)
                && sel.getPeriodicSwapEvents() == 0
                && sel.getAlarmSwapEvents() == 0;
        report("no swaps on stationary stream (no alarm)", ok);
    }

    private static void testPeriodicReSelectionMigratesAfterDrift() {
        int F = 6, K = 2, N = 600, P = 300;
        Set<Integer> oldInfo = new HashSet<>(Arrays.asList(0, 1));
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 904, oldInfo, y);
        DriftAwareSelector sel = buildSelector(F, K, P, 0, 100);
        sel.initialize(win, y);

        Set<Integer> newInfo = new HashSet<>(Arrays.asList(4, 5));
        Random rng = new Random(404);
        for (int i = 0; i < 4 * P; i++) {
            int cls = rng.nextInt(2);
            sel.update(sample(F, newInfo, cls, rng), cls, false, Collections.emptySet());
        }
        Set<Integer> finalSel = new HashSet<>();
        for (int idx : sel.getSelectedFeatures()) finalSel.add(idx);
        boolean migrated = finalSel.contains(4) || finalSel.contains(5);
        boolean periodicTriggered = sel.getPeriodicSwapEvents() >= 1;
        boolean noAlarmPath = sel.getAlarmSwapEvents() == 0;
        report("periodic re-selection migrates after drift (final=" + finalSel
                        + ", periodicEvents=" + sel.getPeriodicSwapEvents() + ")",
                migrated && periodicTriggered && noAlarmPath);
    }

    private static void testAlarmDrivenSwapAfterWPostDrift() {
        int F = 6, K = 2, N = 600, P = 100_000, W = 250;
        Set<Integer> oldInfo = new HashSet<>(Arrays.asList(0, 1));
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 905, oldInfo, y);
        DriftAwareSelector sel = buildSelector(F, K, P, 0, W);
        sel.initialize(win, y);
        Set<Integer> initial = new HashSet<>();
        for (int idx : sel.getSelectedFeatures()) initial.add(idx);

        Set<Integer> newInfo = new HashSet<>(Arrays.asList(4, 5));
        Set<Integer> drifting = new HashSet<>(Arrays.asList(0, 1));
        Random rng = new Random(505);

        int cls = rng.nextInt(2);
        sel.update(sample(F, newInfo, cls, rng), cls, true, drifting);
        boolean startedCollecting = sel.isCollecting();

        for (int i = 0; i < W - 2; i++) {
            cls = rng.nextInt(2);
            sel.update(sample(F, newInfo, cls, rng), cls, false, Collections.emptySet());
        }
        boolean midCollecting = sel.isCollecting() && sel.getAlarmSwapEvents() == 0;

        cls = rng.nextInt(2);
        sel.update(sample(F, newInfo, cls, rng), cls, false, Collections.emptySet());

        Set<Integer> finalSel = new HashSet<>();
        for (int idx : sel.getSelectedFeatures()) finalSel.add(idx);
        boolean swapped = sel.getAlarmSwapEvents() == 1
                && !sel.isCollecting()
                && (finalSel.contains(4) || finalSel.contains(5));
        report("alarm-driven swap after W=" + W + " post-drift updates (initial=" + initial
                        + ", final=" + finalSel + ")",
                startedCollecting && midCollecting && swapped);
    }

    private static void testAlarmIgnoredWhenNoDriftingFeaturesGiven() {
        int F = 6, K = 2, N = 600, P = 100_000, W = 200;
        Set<Integer> oldInfo = new HashSet<>(Arrays.asList(0, 1));
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 906, oldInfo, y);
        DriftAwareSelector sel = buildSelector(F, K, P, 0, W);
        sel.initialize(win, y);
        int[] before = sel.getSelectedFeatures();

        Set<Integer> newInfo = new HashSet<>(Arrays.asList(4, 5));
        Random rng = new Random(606);
        int cls = rng.nextInt(2);
        sel.update(sample(F, newInfo, cls, rng), cls, true, Collections.emptySet());
        for (int i = 0; i < W; i++) {
            cls = rng.nextInt(2);
            sel.update(sample(F, newInfo, cls, rng), cls, false, Collections.emptySet());
        }
        int[] after = sel.getSelectedFeatures();
        boolean ok = Arrays.equals(before, after)
                && sel.getAlarmSwapEvents() == 0
                && !sel.isCollecting();
        report("alarm with empty drifting set produces no alarm-swap", ok);
    }

    private static void testAlarmRespectsMinTenure() {
        int F = 6, K = 2, N = 600, P = 100_000, W = 200;
        int minTenure = 10 * W;
        Set<Integer> phase1Info = new HashSet<>(Arrays.asList(0, 1));
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 907, phase1Info, y);
        DriftAwareSelector sel = buildSelector(F, K, P, minTenure, W);
        sel.initialize(win, y);

        Random rng = new Random(707);

        Set<Integer> phase2Info = new HashSet<>(Arrays.asList(4, 5));
        Set<Integer> drifting1 = new HashSet<>(Arrays.asList(0, 1));
        int cls = rng.nextInt(2);
        sel.update(sample(F, phase2Info, cls, rng), cls, true, drifting1);
        for (int i = 0; i < W; i++) {
            cls = rng.nextInt(2);
            sel.update(sample(F, phase2Info, cls, rng), cls, false, Collections.emptySet());
        }
        Set<Integer> afterFirstAlarm = new HashSet<>();
        for (int idx : sel.getSelectedFeatures()) afterFirstAlarm.add(idx);
        boolean firstAlarmSwapped = sel.getAlarmSwapEvents() == 1
                && (afterFirstAlarm.contains(4) || afterFirstAlarm.contains(5));

        Set<Integer> phase3Info = new HashSet<>(Arrays.asList(2, 3));
        Set<Integer> drifting2 = new HashSet<>(afterFirstAlarm);
        cls = rng.nextInt(2);
        sel.update(sample(F, phase3Info, cls, rng), cls, true, drifting2);
        for (int i = 0; i < W; i++) {
            cls = rng.nextInt(2);
            sel.update(sample(F, phase3Info, cls, rng), cls, false, Collections.emptySet());
        }
        Set<Integer> afterSecondAlarm = new HashSet<>();
        for (int idx : sel.getSelectedFeatures()) afterSecondAlarm.add(idx);
        Set<Integer> newcomersFromPhase2 = new HashSet<>();
        for (int idx : afterFirstAlarm) {
            if (idx == 4 || idx == 5) newcomersFromPhase2.add(idx);
        }
        boolean newcomersProtected = afterSecondAlarm.containsAll(newcomersFromPhase2);
        boolean noSecondAlarmSwap = sel.getAlarmSwapEvents() == 1;

        report("alarm respects minTenure (newcomers protected; phase2=" + afterFirstAlarm
                        + ", phase3=" + afterSecondAlarm + ")",
                firstAlarmSwapped && newcomersProtected && noSecondAlarmSwap);
    }

    private static void testRepeatedAlarmDuringCollectionIsIgnored() {
        int F = 6, K = 2, N = 600, P = 100_000, W = 300;
        Set<Integer> oldInfo = new HashSet<>(Arrays.asList(0, 1));
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 908, oldInfo, y);
        DriftAwareSelector sel = buildSelector(F, K, P, 0, W);
        sel.initialize(win, y);

        Set<Integer> newInfo = new HashSet<>(Arrays.asList(4, 5));
        Set<Integer> drifting = new HashSet<>(Arrays.asList(0, 1));
        Random rng = new Random(808);

        int cls = rng.nextInt(2);
        sel.update(sample(F, newInfo, cls, rng), cls, true, drifting);
        for (int i = 0; i < 100; i++) {
            cls = rng.nextInt(2);
            sel.update(sample(F, newInfo, cls, rng), cls, true,
                    new HashSet<>(Arrays.asList(2, 3)));
        }
        boolean stillFirst = sel.isCollecting() && sel.getAlarmSwapEvents() == 0;

        for (int i = 0; i < W; i++) {
            cls = rng.nextInt(2);
            sel.update(sample(F, newInfo, cls, rng), cls, false, Collections.emptySet());
        }
        boolean exactlyOne = sel.getAlarmSwapEvents() == 1 && !sel.isCollecting();
        report("repeated alarms during collection are ignored", stillFirst && exactlyOne);
    }

    private static void testInCandidatesExcludeDriftingFeatures() {
        int F = 6, K = 2, N = 600, P = 100_000, W = 250;
        Set<Integer> oldInfo = new HashSet<>(Arrays.asList(0, 1));
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 909, oldInfo, y);
        DriftAwareSelector sel = buildSelector(F, K, P, 0, W);
        sel.initialize(win, y);

        Set<Integer> newInfo = new HashSet<>(Arrays.asList(4, 5));
        Set<Integer> drifting = new HashSet<>(Arrays.asList(0, 1, 4));
        Random rng = new Random(909);

        int cls = rng.nextInt(2);
        sel.update(sample(F, newInfo, cls, rng), cls, true, drifting);
        for (int i = 0; i < W; i++) {
            cls = rng.nextInt(2);
            sel.update(sample(F, newInfo, cls, rng), cls, false, Collections.emptySet());
        }
        Set<Integer> finalSel = new HashSet<>();
        for (int idx : sel.getSelectedFeatures()) finalSel.add(idx);
        boolean noDriftingIn = !finalSel.contains(4);
        boolean swapped = sel.getAlarmSwapEvents() == 1
                && (finalSel.contains(5) || finalSel.contains(2) || finalSel.contains(3));
        report("alarm in-candidates exclude drifting features (final=" + finalSel + ")",
                noDriftingIn && swapped);
    }

    private static void testUpdateBeforeInitIsNoop() {
        DriftAwareSelector sel = buildSelector(4, 2, 200, 50, 100);
        sel.update(new double[]{0, 0, 0, 0}, 0, true,
                new HashSet<>(Arrays.asList(0)));
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