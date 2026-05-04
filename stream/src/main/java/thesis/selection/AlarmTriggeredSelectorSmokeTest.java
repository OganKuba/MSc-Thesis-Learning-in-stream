package thesis.selection;

import thesis.discretization.PiDDiscretizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class AlarmTriggeredSelectorSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("ALARM-TRIGGERED SELECTOR SMOKE TESTS");
        System.out.println("=".repeat(70));

        testInitializeSetsSelection();
        testThrowsBeforeInit();
        testRejectsBadConstructor();
        testRejectsBadInitialWindow();
        testUpdateNoOpBeforeAlarm();
        testAlarmTriggersReSelectionAfterExactlyW();
        testReSelectionAdaptsToNewInformativeFeature();
        testSecondAlarmDuringCollectionIsIgnored();
        testFilterInstanceShapeAndValidation();
        testUpdateBeforeInitIncrementsCounter();
        testListenerEmitsAlarmAndReSelectionEvents();
        testNoOpIfWindowFedFullyButNoChangeInData();
        testTieBreakPreservesPreviousSelectionOnEqualScores();
        testDoubleInitializeRejected();
        testSoftResetKeepsDiscretizerReady();
        testRepeatedAlarmsBetweenWindowsAllAccepted();
        testRejectsBadInstanceInUpdate();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    private static double[][] makeWindow(int N, int F, long seed,
                                         int informative, int[] outLabels) {
        Random rng = new Random(seed);
        double[][] win = new double[N][F];
        for (int i = 0; i < N; i++) {
            int cls = rng.nextInt(2);
            outLabels[i] = cls;
            for (int f = 0; f < F; f++) {
                win[i][f] = (f == informative)
                        ? (cls == 0 ? rng.nextGaussian() - 3 : rng.nextGaussian() + 3)
                        : rng.nextGaussian();
            }
        }
        return win;
    }

    private static AlarmTriggeredSelector buildSelector(int F, int K, int W) {
        return new AlarmTriggeredSelector(F, 2, K, W,
                new PiDDiscretizer(F, 2, 32, 4, 200, 500),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc, 1));
    }

    private static void testInitializeSetsSelection() {
        int F = 6, N = 600, K = 2;
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 101, 1, y);
        AlarmTriggeredSelector sel = buildSelector(F, K, 200);
        sel.initialize(win, y);
        int[] s = sel.getSelectedFeatures();
        boolean ok = sel.isInitialized()
                && s.length == K
                && Arrays.equals(s, sel.getCurrentSelection())
                && sel.getReSelections() == 0
                && contains(s, 1);
        report("initialize sets selection containing informative feature 1 (got=" +
                Arrays.toString(s) + ")", ok);
    }

    private static void testThrowsBeforeInit() {
        AlarmTriggeredSelector sel = buildSelector(4, 2, 100);
        boolean t1 = false, t2 = false, t3 = false;
        try { sel.getSelectedFeatures(); } catch (IllegalStateException e) { t1 = true; }
        try { sel.getCurrentSelection(); } catch (IllegalStateException e) { t2 = true; }
        try { sel.filterInstance(new double[]{0, 0, 0, 0}); } catch (IllegalStateException e) { t3 = true; }
        report("throws before initialize()", t1 && t2 && t3 && !sel.isInitialized());
    }

    private static void testRejectsBadConstructor() {
        boolean t1 = false, t2 = false, t3 = false, t4 = false, t5 = false, t6 = false;
        try { new AlarmTriggeredSelector(0, 2); } catch (IllegalArgumentException e) { t1 = true; }
        try { new AlarmTriggeredSelector(4, 1); } catch (IllegalArgumentException e) { t2 = true; }
        try { new AlarmTriggeredSelector(4, 2, 0, 100,
                new PiDDiscretizer(4, 2),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc)); }
        catch (IllegalArgumentException e) { t3 = true; }
        try { new AlarmTriggeredSelector(4, 2, 5, 100,
                new PiDDiscretizer(4, 2),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc)); }
        catch (IllegalArgumentException e) { t4 = true; }
        try { new AlarmTriggeredSelector(4, 2, 2, 10,
                new PiDDiscretizer(4, 2),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc)); }
        catch (IllegalArgumentException e) { t5 = true; }
        try { new AlarmTriggeredSelector(4, 2, 2, 100,
                new PiDDiscretizer(4, 2),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc), 0.0); }
        catch (IllegalArgumentException e) { t6 = true; }
        report("constructor rejects bad params", t1 && t2 && t3 && t4 && t5 && t6);
    }

    private static void testRejectsBadInitialWindow() {
        AlarmTriggeredSelector sel = buildSelector(4, 2, 100);
        boolean t1 = false, t2 = false, t3 = false;
        try { sel.initialize(null, new int[]{0}); } catch (IllegalArgumentException e) { t1 = true; }
        try { sel.initialize(new double[][]{{0,0,0,0}}, new int[]{0,1}); }
        catch (IllegalArgumentException e) { t2 = true; }
        try { sel.initialize(new double[][]{{0,0,0}}, new int[]{0}); }
        catch (IllegalArgumentException e) { t3 = true; }
        report("rejects bad initial window", t1 && t2 && t3);
    }

    private static void testUpdateNoOpBeforeAlarm() {
        int F = 5, N = 600, K = 2, W = 200;
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 102, 0, y);
        AlarmTriggeredSelector sel = buildSelector(F, K, W);
        sel.initialize(win, y);
        int[] before = sel.getSelectedFeatures();
        Random rng = new Random(202);
        for (int i = 0; i < 1000; i++) {
            double[] x = new double[F];
            for (int f = 0; f < F; f++) x[f] = rng.nextGaussian();
            sel.update(x, rng.nextInt(2), false, Collections.emptySet());
        }
        int[] after = sel.getSelectedFeatures();
        report("update without alarm leaves selection unchanged",
                Arrays.equals(before, after) && sel.getReSelections() == 0);
    }

    private static void testAlarmTriggersReSelectionAfterExactlyW() {
        int F = 5, N = 600, K = 1, W = 200;
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 103, 0, y);
        AlarmTriggeredSelector sel = buildSelector(F, K, W);
        sel.initialize(win, y);
        long beforeCount = sel.getReSelections();

        Random rng = new Random(303);
        double[] x = new double[F];

        for (int f = 0; f < F; f++) x[f] = rng.nextGaussian();
        sel.update(x, rng.nextInt(2), true, Collections.emptySet());
        boolean afterAlarm = sel.isCollectingPostDrift() && sel.getPostDriftCount() == 1;

        for (int i = 0; i < W - 2; i++) {
            for (int f = 0; f < F; f++) x[f] = rng.nextGaussian();
            sel.update(x, rng.nextInt(2), false, Collections.emptySet());
        }
        boolean midOk = sel.getReSelections() == beforeCount
                && sel.isCollectingPostDrift()
                && sel.getPostDriftCount() == W - 1;

        for (int f = 0; f < F; f++) x[f] = rng.nextGaussian();
        sel.update(x, rng.nextInt(2), false, Collections.emptySet());

        boolean fired = sel.getReSelections() == beforeCount + 1
                && !sel.isCollectingPostDrift()
                && sel.getPostDriftCount() == 0;
        report("alarm triggers re-selection exactly after W ticks",
                afterAlarm && midOk && fired);
    }

    private static void testReSelectionAdaptsToNewInformativeFeature() {
        int F = 6, K = 1, W = 400, N = 600;
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 104, 0, y);
        AlarmTriggeredSelector sel = buildSelector(F, K, W);
        sel.initialize(win, y);
        int initialPick = sel.getSelectedFeatures()[0];

        Random rng = new Random(404);
        boolean firstTick = true;
        for (int i = 0; i < W + 10; i++) {
            int cls = rng.nextInt(2);
            double[] x = new double[F];
            for (int f = 0; f < F; f++) {
                x[f] = (f == 3)
                        ? (cls == 0 ? rng.nextGaussian() - 3 : rng.nextGaussian() + 3)
                        : rng.nextGaussian();
            }
            sel.update(x, cls, firstTick, Collections.emptySet());
            firstTick = false;
        }
        int newPick = sel.getSelectedFeatures()[0];
        report("after alarm + " + W + " drifted samples, top-1 switches "
                        + initialPick + " -> " + newPick,
                initialPick == 0 && newPick == 3 && sel.getReSelections() == 1);
    }

    private static void testSecondAlarmDuringCollectionIsIgnored() {
        int F = 5, K = 1, W = 300, N = 600;
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 105, 0, y);
        AlarmTriggeredSelector sel = buildSelector(F, K, W);
        sel.initialize(win, y);

        Random rng = new Random(505);
        double[] x = new double[F];
        for (int f = 0; f < F; f++) x[f] = rng.nextGaussian();
        sel.update(x, rng.nextInt(2), true, Collections.emptySet());

        for (int i = 0; i < 100; i++) {
            for (int f = 0; f < F; f++) x[f] = rng.nextGaussian();
            sel.update(x, rng.nextInt(2), true, Collections.emptySet());
        }
        boolean stillCollecting = sel.getReSelections() == 0
                && sel.isCollectingPostDrift()
                && sel.getAlarmsIgnoredWhileBusy() == 100
                && sel.getAlarmsAccepted() == 1;

        for (int i = 0; i < W; i++) {
            for (int f = 0; f < F; f++) x[f] = rng.nextGaussian();
            sel.update(x, rng.nextInt(2), false, Collections.emptySet());
        }
        report("repeated alarms during collection are ignored, single re-selection performed",
                stillCollecting && sel.getReSelections() == 1);
    }

    private static void testFilterInstanceShapeAndValidation() {
        int F = 5, N = 600, K = 3;
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 106, 2, y);
        AlarmTriggeredSelector sel = buildSelector(F, K, 200);
        sel.initialize(win, y);
        double[] out = sel.filterInstance(win[0]);
        boolean shapeOk = out.length == K;
        boolean threw = false;
        try { sel.filterInstance(new double[]{0, 0}); }
        catch (IllegalArgumentException e) { threw = true; }
        report("filterInstance shape + length validation", shapeOk && threw);
    }

    private static void testUpdateBeforeInitIncrementsCounter() {
        AlarmTriggeredSelector sel = buildSelector(4, 2, 100);
        sel.update(new double[]{0, 0, 0, 0}, 0, true, Collections.emptySet());
        sel.update(new double[]{0, 0, 0, 0}, 0, false, Collections.emptySet());
        report("update() before init increments updatesBeforeInit and is silent no-op",
                !sel.isInitialized() && sel.getUpdatesBeforeInit() == 2);
    }

    private static void testListenerEmitsAlarmAndReSelectionEvents() {
        int F = 5, N = 600, K = 1, W = 250;
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 107, 0, y);
        AlarmTriggeredSelector sel = buildSelector(F, K, W);
        sel.initialize(win, y);

        List<String> events = new ArrayList<>();
        sel.setEventListener(new AlarmTriggeredSelector.EventListener() {
            @Override public void onAlarm(long alarmIdx, boolean accepted, Set<Integer> drifting) {
                events.add("alarm#" + alarmIdx + ":" + (accepted ? "accept" : "ignore"));
            }
            @Override public void onCollectingStart(int w, int[] currentSelection) {
                events.add("collectStart:" + Arrays.toString(currentSelection));
            }
            @Override public void onReSelection(int[] oldSel, int[] newSel, double[] scores, boolean changed) {
                events.add("reselect:" + Arrays.toString(oldSel) + "->" +
                        Arrays.toString(newSel) + ":changed=" + changed);
            }
        });

        Random rng = new Random(707);
        boolean firstTick = true;
        for (int i = 0; i < W + 10; i++) {
            int cls = rng.nextInt(2);
            double[] x = new double[F];
            for (int f = 0; f < F; f++) {
                x[f] = (f == 4)
                        ? (cls == 0 ? rng.nextGaussian() - 3 : rng.nextGaussian() + 3)
                        : rng.nextGaussian();
            }
            sel.update(x, cls, firstTick, Collections.emptySet());
            firstTick = false;
        }
        boolean hasAlarm = events.stream().anyMatch(s -> s.startsWith("alarm#1:accept"));
        boolean hasCollect = events.stream().anyMatch(s -> s.startsWith("collectStart:"));
        boolean hasReselect = events.stream().anyMatch(s -> s.startsWith("reselect:"));
        report("listener receives alarm+collectStart+reselect events (events=" + events + ")",
                hasAlarm && hasCollect && hasReselect);
    }

    private static void testNoOpIfWindowFedFullyButNoChangeInData() {
        int F = 5, N = 600, K = 1, W = 250;
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 108, 2, y);
        AlarmTriggeredSelector sel = buildSelector(F, K, W);
        sel.initialize(win, y);
        int[] before = sel.getSelectedFeatures();

        Random rng = new Random(808);
        boolean firstTick = true;
        for (int i = 0; i < W + 10; i++) {
            int cls = rng.nextInt(2);
            double[] x = new double[F];
            for (int f = 0; f < F; f++) {
                x[f] = (f == 2)
                        ? (cls == 0 ? rng.nextGaussian() - 3 : rng.nextGaussian() + 3)
                        : rng.nextGaussian();
            }
            sel.update(x, cls, firstTick, Collections.emptySet());
            firstTick = false;
        }
        int[] after = sel.getSelectedFeatures();
        report("re-selection on stable distribution keeps top feature 2 (before=" +
                        Arrays.toString(before) + ", after=" + Arrays.toString(after) + ")",
                contains(before, 2) && contains(after, 2) && sel.getReSelections() == 1);
    }

    private static void testTieBreakPreservesPreviousSelectionOnEqualScores() {
        int F = 4, K = 2, W = 200, N = 600;
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 109, 0, y);
        AlarmTriggeredSelector sel = buildSelector(F, K, W);
        sel.initialize(win, y);
        int[] before = sel.getSelectedFeatures();

        Random rng = new Random(909);
        boolean firstTick = true;
        for (int i = 0; i < W + 10; i++) {
            double[] x = new double[F];
            for (int f = 0; f < F; f++) x[f] = rng.nextGaussian();
            sel.update(x, rng.nextInt(2), firstTick, Collections.emptySet());
            firstTick = false;
        }
        int[] after = sel.getSelectedFeatures();
        boolean preserved = Arrays.equals(before, after);
        report("on equal/noise scores, previous selection is preserved (before=" +
                        Arrays.toString(before) + ", after=" + Arrays.toString(after) + ")",
                preserved && sel.getReSelections() == 1);
    }

    private static void testDoubleInitializeRejected() {
        int F = 4, N = 600;
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 110, 0, y);
        AlarmTriggeredSelector sel = buildSelector(F, 2, 100);
        sel.initialize(win, y);
        boolean threw = false;
        try { sel.initialize(win, y); } catch (IllegalStateException e) { threw = true; }
        report("double initialize rejected", threw);
    }

    private static void testSoftResetKeepsDiscretizerReady() {
        int F = 4, N = 600, K = 1, W = 100;
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 111, 0, y);
        PiDDiscretizer pid = new PiDDiscretizer(F, 2, 32, 4, 200, 500);
        AlarmTriggeredSelector sel = new AlarmTriggeredSelector(F, 2, K, W, pid,
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc, 1), 0.05);
        sel.initialize(win, y);
        boolean readyBefore = pid.isReady();
        Random rng = new Random(1010);
        double[] x = new double[F];
        for (int f = 0; f < F; f++) x[f] = rng.nextGaussian();
        sel.update(x, 0, true, Collections.emptySet());
        boolean readyAfter = pid.isReady();
        report("soft-reset on alarm keeps discretizer ready (before=" + readyBefore +
                ", after=" + readyAfter + ")", readyBefore && readyAfter);
    }

    private static void testRepeatedAlarmsBetweenWindowsAllAccepted() {
        int F = 5, K = 1, W = 100, N = 600;
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 112, 0, y);
        AlarmTriggeredSelector sel = buildSelector(F, K, W);
        sel.initialize(win, y);

        Random rng = new Random(1111);
        double[] x = new double[F];
        for (int round = 0; round < 3; round++) {
            for (int f = 0; f < F; f++) x[f] = rng.nextGaussian();
            sel.update(x, rng.nextInt(2), true, Collections.emptySet());
            for (int i = 0; i < W; i++) {
                for (int f = 0; f < F; f++) x[f] = rng.nextGaussian();
                sel.update(x, rng.nextInt(2), false, Collections.emptySet());
            }
        }
        report("3 alarms separated by W ticks each → 3 accepted, 3 reselections (acc=" +
                        sel.getAlarmsAccepted() + ", reSel=" + sel.getReSelections() + ")",
                sel.getAlarmsAccepted() == 3 && sel.getReSelections() == 3
                        && sel.getAlarmsIgnoredWhileBusy() == 0);
    }

    private static void testRejectsBadInstanceInUpdate() {
        int F = 4, N = 600;
        int[] y = new int[N];
        double[][] win = makeWindow(N, F, 113, 0, y);
        AlarmTriggeredSelector sel = buildSelector(F, 2, 100);
        sel.initialize(win, y);
        boolean t1 = false, t2 = false, t3 = false;
        try { sel.update(new double[]{0,0}, 0, false, Collections.emptySet()); }
        catch (IllegalArgumentException e) { t1 = true; }
        try { sel.update(null, 0, false, Collections.emptySet()); }
        catch (IllegalArgumentException e) { t2 = true; }
        try { sel.update(new double[]{0,0,0,0}, 9, false, Collections.emptySet()); }
        catch (IllegalArgumentException e) { t3 = true; }
        report("update() validates instance shape and label", t1 && t2 && t3);
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