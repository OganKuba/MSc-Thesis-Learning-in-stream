package thesis.selection;

import thesis.discretization.PiDDiscretizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
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
        testPeriodicReSelectionMigratesAfterDriftWithoutAlarm();
        testAlarmDrivenSwapAfterWPostDrift();
        testAlarmIgnoredWhenNoDriftingFeaturesGiven();
        testAlarmIgnoredWhenDriftingDisjointFromSelection();
        testAlarmRespectsMinTenure();
        testRepeatedAlarmDuringCollectionIsIgnored();
        testInCandidatesExcludeDriftingFeatures();
        testOutOfRangeDriftingIndicesAreSanitized();
        testListenerEmitsAlarmAndSwapEvents();
        testStabilityRatioInRange();
        testTieEpsilonPreventsTrivialAlarmSwap();
        testS4PreservesStableMoreThanS2();
        testUpdateBeforeInitIncrementsCounter();
        testDoubleInitializeRejected();
        testRejectsBadInstanceInUpdate();

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
                new PiDDiscretizer(F, 2, 32, 4, 200, 500),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc, 1));
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
                && !sel.isCollecting()
                && contains(s, 1) && contains(s, 4);
        report("initialize sets selection containing informative {1,4} (got=" +
                Arrays.toString(s) + ")", ok);
    }

    private static void testThrowsBeforeInit() {
        DriftAwareSelector sel = buildSelector(4, 2, 200, 50, 100);
        boolean t1=false,t2=false,t3=false;
        try { sel.getSelectedFeatures(); } catch (IllegalStateException e) { t1 = true; }
        try { sel.getCurrentSelection(); } catch (IllegalStateException e) { t2 = true; }
        try { sel.filterInstance(new double[]{0,0,0,0}); } catch (IllegalStateException e) { t3 = true; }
        report("throws before initialize()", t1 && t2 && t3 && !sel.isInitialized());
    }

    private static void testRejectsBadConstructor() {
        boolean t1=false,t2=false,t3=false,t4=false,t5=false,t6=false,t7=false,t8=false,t9=false;
        try { new DriftAwareSelector(0, 2); } catch (IllegalArgumentException e) { t1 = true; }
        try { new DriftAwareSelector(4, 1); } catch (IllegalArgumentException e) { t2 = true; }
        try { new DriftAwareSelector(4, 2, 0, 200, 50, 100,
                new PiDDiscretizer(4, 2),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc)); }
        catch (IllegalArgumentException e) { t3 = true; }
        try { new DriftAwareSelector(4, 2, 5, 200, 50, 100,
                new PiDDiscretizer(4, 2),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc)); }
        catch (IllegalArgumentException e) { t4 = true; }
        try { new DriftAwareSelector(4, 2, 2, 50, 50, 100,
                new PiDDiscretizer(4, 2),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc)); }
        catch (IllegalArgumentException e) { t5 = true; }
        try { new DriftAwareSelector(4, 2, 2, 200, -1, 100,
                new PiDDiscretizer(4, 2),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc)); }
        catch (IllegalArgumentException e) { t6 = true; }
        try { new DriftAwareSelector(4, 2, 2, 200, 50, 10,
                new PiDDiscretizer(4, 2),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc)); }
        catch (IllegalArgumentException e) { t7 = true; }
        try { new DriftAwareSelector(4, 2, 2, 200, 50, 100,
                new PiDDiscretizer(4, 2),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc),
                -0.01, 0.05, 0.3); }
        catch (IllegalArgumentException e) { t8 = true; }
        try { new DriftAwareSelector(4, 2, 2, 200, 50, 100,
                new PiDDiscretizer(4, 2),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc),
                0.01, 0.0, 0.3); }
        catch (IllegalArgumentException e) { t9 = true; }
        report("constructor rejects bad params",
                t1 && t2 && t3 && t4 && t5 && t6 && t7 && t8 && t9);
    }

    private static void testRejectsBadInitialWindow() {
        DriftAwareSelector sel = buildSelector(4, 2, 200, 50, 100);
        boolean t1=false,t2=false,t3=false;
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
        try { sel.filterInstance(new double[]{0,0}); } catch (IllegalArgumentException e) { threw = true; }
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
        report("no swaps on stationary stream",
                Arrays.equals(before, after) && sel.getPeriodicSwapEvents() == 0
                        && sel.getAlarmSwapEvents() == 0);
    }

    private static void testPeriodicReSelectionMigratesAfterDriftWithoutAlarm() {
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
        report("periodic migrates after drift, no alarm path used (final=" + finalSel +
                        ", periodic=" + sel.getPeriodicSwapEvents() +
                        ", alarm=" + sel.getAlarmSwapEvents() + ")",
                migrated && sel.getPeriodicSwapEvents() >= 1 && sel.getAlarmSwapEvents() == 0);
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

        for (int i = 0; i < W - 1; i++) {
            cls = rng.nextInt(2);
            sel.update(sample(F, newInfo, cls, rng), cls, false, Collections.emptySet());
        }
        Set<Integer> finalSel = new HashSet<>();
        for (int idx : sel.getSelectedFeatures()) finalSel.add(idx);
        boolean swapped = sel.getAlarmSwapEvents() == 1
                && !sel.isCollecting()
                && (finalSel.contains(4) || finalSel.contains(5));
        report("alarm-driven swap after W=" + W + " post-drift updates (initial=" + initial +
                        ", final=" + finalSel + ")",
                startedCollecting && swapped);
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
        report("alarm with empty drifting set is ignored (no swap, no collect)",
                Arrays.equals(before, after)
                        && sel.getAlarmSwapEvents() == 0
                        && !sel.isCollecting()
                        && sel.getAlarmsIgnoredNoTargets() == 1);
    }

    private static void testAlarmIgnoredWhenDriftingDisjointFromSelection() {
        int F = 6, K = 2, N = 600, P = 100_000, W = 200;
        Set<Integer> oldInfo = new HashSet<>(Arrays.asList(0, 1));
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 907, oldInfo, y);
        DriftAwareSelector sel = buildSelector(F, K, P, 0, W);
        sel.initialize(win, y);
        int[] before = sel.getSelectedFeatures();

        Set<Integer> drifting = new HashSet<>(Arrays.asList(2, 3));
        Random rng = new Random(707);
        int cls = rng.nextInt(2);
        sel.update(sample(F, oldInfo, cls, rng), cls, true, drifting);
        for (int i = 0; i < W; i++) {
            cls = rng.nextInt(2);
            sel.update(sample(F, oldInfo, cls, rng), cls, false, Collections.emptySet());
        }
        int[] after = sel.getSelectedFeatures();
        report("alarm where drifting ∩ selection = ∅ → ignored (allStable)",
                Arrays.equals(before, after)
                        && sel.getAlarmSwapEvents() == 0
                        && !sel.isCollecting()
                        && sel.getAlarmsIgnoredAllStable() == 1);
    }

    private static void testAlarmRespectsMinTenure() {
        int F = 6, K = 2, N = 600, P = 100_000, W = 200;
        int minTenure = 10 * W;
        Set<Integer> phase1Info = new HashSet<>(Arrays.asList(0, 1));
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 908, phase1Info, y);
        DriftAwareSelector sel = buildSelector(F, K, P, minTenure, W);
        sel.initialize(win, y);
        Random rng = new Random(808);

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
        Set<Integer> newcomers = new HashSet<>();
        for (int idx : afterFirstAlarm) if (idx == 4 || idx == 5) newcomers.add(idx);
        boolean firstAlarmSwapped = sel.getAlarmSwapEvents() == 1 && !newcomers.isEmpty();

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
        boolean newcomersProtected = afterSecondAlarm.containsAll(newcomers);
        report("minTenure protects newly swapped-in features in alarm path (phase2=" +
                        afterFirstAlarm + ", phase3=" + afterSecondAlarm + ")",
                firstAlarmSwapped && newcomersProtected);
    }

    private static void testRepeatedAlarmDuringCollectionIsIgnored() {
        int F = 6, K = 2, N = 600, P = 100_000, W = 300;
        Set<Integer> oldInfo = new HashSet<>(Arrays.asList(0, 1));
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 909, oldInfo, y);
        DriftAwareSelector sel = buildSelector(F, K, P, 0, W);
        sel.initialize(win, y);

        Set<Integer> newInfo = new HashSet<>(Arrays.asList(4, 5));
        Set<Integer> drifting = new HashSet<>(Arrays.asList(0, 1));
        Random rng = new Random(909);

        int cls = rng.nextInt(2);
        sel.update(sample(F, newInfo, cls, rng), cls, true, drifting);
        for (int i = 0; i < 100; i++) {
            cls = rng.nextInt(2);
            sel.update(sample(F, newInfo, cls, rng), cls, true,
                    new HashSet<>(Arrays.asList(2, 3)));
        }
        boolean stillCollecting = sel.isCollecting() && sel.getAlarmSwapEvents() == 0
                && sel.getAlarmsIgnoredWhileBusy() == 100;

        for (int i = 0; i < W; i++) {
            cls = rng.nextInt(2);
            sel.update(sample(F, newInfo, cls, rng), cls, false, Collections.emptySet());
        }
        boolean exactlyOne = sel.getAlarmSwapEvents() == 1 && !sel.isCollecting();
        report("repeated alarms during collection are ignored", stillCollecting && exactlyOne);
    }

    private static void testInCandidatesExcludeDriftingFeatures() {
        int F = 6, K = 2, N = 600, P = 100_000, W = 250;
        Set<Integer> oldInfo = new HashSet<>(Arrays.asList(0, 1));
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 910, oldInfo, y);
        DriftAwareSelector sel = buildSelector(F, K, P, 0, W);
        sel.initialize(win, y);

        Set<Integer> newInfo = new HashSet<>(Arrays.asList(4, 5));
        Set<Integer> drifting = new HashSet<>(Arrays.asList(0, 1, 4));
        Random rng = new Random(1010);

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

    private static void testOutOfRangeDriftingIndicesAreSanitized() {
        int F = 4, K = 2, N = 600, P = 100_000, W = 200;
        Set<Integer> info = new HashSet<>(Arrays.asList(0, 1));
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 911, info, y);
        DriftAwareSelector sel = buildSelector(F, K, P, 0, W);
        sel.initialize(win, y);
        int[] before = sel.getSelectedFeatures();

        Random rng = new Random(1111);
        Set<Integer> garbage = new HashSet<>(Arrays.asList(-1, 99, 100));
        int cls = rng.nextInt(2);
        sel.update(sample(F, info, cls, rng), cls, true, garbage);
        for (int i = 0; i < W; i++) {
            cls = rng.nextInt(2);
            sel.update(sample(F, info, cls, rng), cls, false, Collections.emptySet());
        }
        int[] after = sel.getSelectedFeatures();
        report("out-of-range drifting indices sanitized → alarm ignored as no-targets",
                Arrays.equals(before, after)
                        && sel.getAlarmSwapEvents() == 0
                        && sel.getAlarmsIgnoredNoTargets() == 1);
    }

    private static void testListenerEmitsAlarmAndSwapEvents() {
        int F = 5, K = 2, N = 600, P = 100_000, W = 200;
        Set<Integer> oldInfo = new HashSet<>(Arrays.asList(0, 1));
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 912, oldInfo, y);
        DriftAwareSelector sel = buildSelector(F, K, P, 0, W);
        sel.initialize(win, y);

        List<String> events = new ArrayList<>();
        sel.setEventListener(new DriftAwareSelector.EventListener() {
            @Override public void onAlarm(long i, boolean acc, Set<Integer> drift,
                                          Set<Integer> ds, Set<Integer> ss) {
                events.add("alarm#" + i + ":" + (acc ? "accept" : "ignore") +
                        ":drift=" + drift + ":ds=" + ds + ":ss=" + ss);
            }
            @Override public void onPeriodicTick(long i, boolean t) {}
            @Override public void onSwap(DriftAwareSelector.TriggerType type, long i,
                                         int[] o, int[] ns, Set<Integer> ro, Set<Integer> ri,
                                         double[] sc, long[] ten, double stab) {
                events.add("swap:" + type + ":" + Arrays.toString(o) + "->" + Arrays.toString(ns) +
                        ":out=" + ro + ":in=" + ri + ":stab=" + String.format("%.2f", stab));
            }
        });

        Set<Integer> newInfo = new HashSet<>(Arrays.asList(3, 4));
        Set<Integer> drifting = new HashSet<>(Arrays.asList(0, 1));
        Random rng = new Random(1212);
        int cls = rng.nextInt(2);
        sel.update(sample(F, newInfo, cls, rng), cls, true, drifting);
        for (int i = 0; i < W; i++) {
            cls = rng.nextInt(2);
            sel.update(sample(F, newInfo, cls, rng), cls, false, Collections.emptySet());
        }
        boolean hasAlarm = events.stream().anyMatch(s -> s.startsWith("alarm#1:accept"));
        boolean hasSwap = events.stream().anyMatch(s -> s.startsWith("swap:ALARM_WHERE"));
        report("listener emits alarm + ALARM_WHERE swap events (events=" + events + ")",
                hasAlarm && hasSwap);
    }

    private static void testStabilityRatioInRange() {
        int F = 6, K = 2, N = 600, P = 100_000, W = 200;
        Set<Integer> oldInfo = new HashSet<>(Arrays.asList(0, 1));
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 913, oldInfo, y);
        DriftAwareSelector sel = buildSelector(F, K, P, 0, W);
        sel.initialize(win, y);
        double[] captured = new double[]{-1.0};
        sel.setEventListener(new DriftAwareSelector.EventListener() {
            @Override public void onAlarm(long i, boolean a, Set<Integer> d, Set<Integer> ds, Set<Integer> ss) {}
            @Override public void onPeriodicTick(long i, boolean t) {}
            @Override public void onSwap(DriftAwareSelector.TriggerType type, long i,
                                         int[] o, int[] ns, Set<Integer> ro, Set<Integer> ri,
                                         double[] sc, long[] ten, double stab) {
                captured[0] = stab;
            }
        });
        Set<Integer> newInfo = new HashSet<>(Arrays.asList(4, 5));
        Set<Integer> drifting = new HashSet<>(Arrays.asList(0, 1));
        Random rng = new Random(1313);
        int cls = rng.nextInt(2);
        sel.update(sample(F, newInfo, cls, rng), cls, true, drifting);
        for (int i = 0; i < W; i++) {
            cls = rng.nextInt(2);
            sel.update(sample(F, newInfo, cls, rng), cls, false, Collections.emptySet());
        }
        boolean ok = captured[0] >= 0.0 && captured[0] <= 1.0;
        report("stabilityRatio in [0,1] (got=" + captured[0] + ")", ok);
    }

    private static void testTieEpsilonPreventsTrivialAlarmSwap() {
        int F = 5, K = 2, N = 600, P = 100_000, W = 200;
        Set<Integer> info = new HashSet<>(Arrays.asList(0, 1));
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 914, info, y);
        DriftAwareSelector sel = new DriftAwareSelector(F, 2, K, P, 0, W,
                new PiDDiscretizer(F, 2, 32, 4, 200, 500),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc, 1),
                0.5, 0.05, 0.3);
        sel.initialize(win, y);
        int[] before = sel.getSelectedFeatures();

        Set<Integer> drifting = new HashSet<>(Arrays.asList(0, 1));
        Random rng = new Random(1414);
        int cls = rng.nextInt(2);
        sel.update(noiseSample(F, rng), cls, true, drifting);
        for (int i = 0; i < W; i++) {
            cls = rng.nextInt(2);
            sel.update(noiseSample(F, rng), cls, false, Collections.emptySet());
        }
        int[] after = sel.getSelectedFeatures();
        report("large tieEpsilon prevents alarm-swap on noise (before=" + Arrays.toString(before) +
                        ", after=" + Arrays.toString(after) + ", swaps=" + sel.getAlarmSwapEvents() + ")",
                Arrays.equals(before, after) && sel.getAlarmSwapEvents() == 0);
    }

    private static double[] noiseSample(int F, Random rng) {
        double[] x = new double[F];
        for (int f = 0; f < F; f++) x[f] = rng.nextGaussian();
        return x;
    }

    private static void testS4PreservesStableMoreThanS2() {
        int F = 6, K = 3, N = 600, P = 100_000, W = 250;
        Set<Integer> oldInfo = new HashSet<>(Arrays.asList(0, 1, 2));
        int[] y1 = new int[N], y2 = new int[N];
        double[][] win1 = makeWindow(N, F, 915, oldInfo, y1);
        double[][] win2 = makeWindow(N, F, 915, oldInfo, y2);

        DriftAwareSelector s4 = buildSelector(F, K, P, 0, W);
        s4.initialize(win1, y1);

        AlarmTriggeredSelector s2 = new AlarmTriggeredSelector(F, 2, K, W,
                new PiDDiscretizer(F, 2, 32, 4, 200, 500),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc, 1));
        s2.initialize(win2, y2);

        int[] s4Initial = s4.getSelectedFeatures();
        int[] s2Initial = s2.getSelectedFeatures();

        Set<Integer> newInfoForOne = new HashSet<>(Arrays.asList(3));
        Set<Integer> drifting = new HashSet<>(Arrays.asList(0));
        Random rng = new Random(1515);

        for (int round = 0; round < 1; round++) {
            int cls = rng.nextInt(2);
            double[] x1 = mixedSample(F, oldInfo, newInfoForOne, cls, rng);
            double[] x2 = x1.clone();
            s4.update(x1, cls, true, drifting);
            s2.update(x2, cls, true, drifting);
            for (int i = 0; i < W; i++) {
                cls = rng.nextInt(2);
                x1 = mixedSample(F, oldInfo, newInfoForOne, cls, rng);
                x2 = x1.clone();
                s4.update(x1, cls, false, Collections.emptySet());
                s2.update(x2, cls, false, Collections.emptySet());
            }
        }
        double s4Stab = stab(s4Initial, s4.getSelectedFeatures());
        double s2Stab = stab(s2Initial, s2.getSelectedFeatures());
        report("S4 preserves stable features at least as much as S2 (s4Stab=" + s4Stab +
                        ", s2Stab=" + s2Stab + ")",
                s4Stab >= s2Stab);
    }

    private static double[] mixedSample(int F, Set<Integer> oldInfo, Set<Integer> newInfo,
                                        int cls, Random rng) {
        double[] x = new double[F];
        for (int f = 0; f < F; f++) {
            if (oldInfo.contains(f) && !newInfo.contains(f)) {
                x[f] = rng.nextGaussian();
            } else if (newInfo.contains(f)) {
                x[f] = (cls == 0 ? rng.nextGaussian() - 3 : rng.nextGaussian() + 3);
            } else {
                x[f] = rng.nextGaussian();
            }
        }
        for (int f : oldInfo) {
            if (!newInfo.contains(f)) x[f] = rng.nextGaussian();
        }
        if (oldInfo.contains(1) || oldInfo.contains(2)) {
            for (int f : oldInfo) {
                if (f == 1 || f == 2) {
                    x[f] = (cls == 0 ? rng.nextGaussian() - 3 : rng.nextGaussian() + 3);
                }
            }
        }
        return x;
    }

    private static double stab(int[] oldS, int[] newS) {
        Set<Integer> os = new HashSet<>();
        for (int i : oldS) os.add(i);
        int kept = 0;
        for (int i : newS) if (os.contains(i)) kept++;
        return (double) kept / oldS.length;
    }

    private static void testUpdateBeforeInitIncrementsCounter() {
        DriftAwareSelector sel = buildSelector(4, 2, 200, 50, 100);
        sel.update(new double[]{0,0,0,0}, 0, true, new HashSet<>(Arrays.asList(0)));
        sel.update(new double[]{0,0,0,0}, 0, false, Collections.emptySet());
        report("update() before init counted but no-op (count=" + sel.getUpdatesBeforeInit() + ")",
                !sel.isInitialized() && sel.getUpdatesBeforeInit() == 2);
    }

    private static void testDoubleInitializeRejected() {
        int F = 4, N = 600;
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 916, new HashSet<>(Arrays.asList(0)), y);
        DriftAwareSelector sel = buildSelector(F, 2, 200, 0, 100);
        sel.initialize(win, y);
        boolean threw = false;
        try { sel.initialize(win, y); } catch (IllegalStateException e) { threw = true; }
        report("double initialize rejected", threw);
    }

    private static void testRejectsBadInstanceInUpdate() {
        int F = 4, N = 600;
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 917, new HashSet<>(Arrays.asList(0)), y);
        DriftAwareSelector sel = buildSelector(F, 2, 200, 0, 100);
        sel.initialize(win, y);
        boolean t1=false,t2=false,t3=false;
        try { sel.update(new double[]{0,0}, 0, false, Collections.emptySet()); }
        catch (IllegalArgumentException e) { t1 = true; }
        try { sel.update(null, 0, false, Collections.emptySet()); }
        catch (IllegalArgumentException e) { t2 = true; }
        try { sel.update(new double[]{0,0,0,0}, 9, false, Collections.emptySet()); }
        catch (IllegalArgumentException e) { t3 = true; }
        report("update validates instance shape and label", t1 && t2 && t3);
    }

    private static boolean contains(int[] a, int x) {
        for (int v : a) if (v == x) return true;
        return false;
    }

    private static void report(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  [PASSED] " + name); }
        else    { failed++; System.out.println("  [FAILED] " + name); }
    }
}