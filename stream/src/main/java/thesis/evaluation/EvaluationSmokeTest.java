package thesis.evaluation;

import java.util.Arrays;

public class EvaluationSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) throws InterruptedException {
        System.out.println("=".repeat(70));
        System.out.println("EVALUATION SMOKE TESTS");
        System.out.println("=".repeat(70));

        testPrequentialAccuracyBasic();
        testPrequentialAccuracySlidesWindow();
        testPrequentialAccuracyResetAndBadCtor();

        testCohenKappaPerfectIsOne();
        testCohenKappaRandomIsZeroish();
        testCohenKappaSlidesWindow();
        testCohenKappaIgnoresOutOfRange();
        testCohenKappaResetAndBadCtor();

        testTemporalKappaPerfectVsNoChange();
        testTemporalKappaWorseThanNoChangeIsNegative();
        testTemporalKappaNoChangeAccuracy();
        testTemporalKappaResetAndBadCtor();

        testFeatureStabilityIdenticalSelection();
        testFeatureStabilityPartialOverlap();
        testFeatureStabilityFirstUpdateIsBaseline();
        testFeatureStabilityIgnoresNullAndEmpty();
        testFeatureStabilityReset();

        testRamHoursAccumulatesAndPeak();
        testRamHoursSampleFromRuntimeAutoStarts();
        testRamHoursReset();

        testRecoveryTimeRecoversWithinTolerance();
        testRecoveryTimeUnrecoveredAfterMaxWindow();
        testRecoveryTimeSecondAlarmCountsPreviousAsUnrecovered();
        testRecoveryTimeBadCtorAndReset();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    private static void testPrequentialAccuracyBasic() {
        PrequentialAccuracy p = new PrequentialAccuracy(100);
        for (int i = 0; i < 10; i++) p.update(0, 0);
        for (int i = 0; i < 10; i++) p.update(0, 1);
        boolean ok = p.getCount() == 20 && Math.abs(p.getAccuracy() - 0.5) < 1e-9;
        report("PrequentialAccuracy basic 50%", ok);
    }

    private static void testPrequentialAccuracySlidesWindow() {
        PrequentialAccuracy p = new PrequentialAccuracy(5);
        for (int i = 0; i < 5; i++) p.update(0, 1);
        for (int i = 0; i < 5; i++) p.update(0, 0);
        boolean ok = p.getCount() == 5 && p.getAccuracy() == 1.0;
        report("PrequentialAccuracy slides window (acc=" + p.getAccuracy() + ")", ok);
    }

    private static void testPrequentialAccuracyResetAndBadCtor() {
        PrequentialAccuracy p = new PrequentialAccuracy(10);
        p.update(0, 0);
        p.reset();
        boolean ok = p.getCount() == 0 && p.getAccuracy() == 0.0;
        boolean threw = false;
        try { new PrequentialAccuracy(0); } catch (IllegalArgumentException e) { threw = true; }
        report("PrequentialAccuracy reset + bad ctor", ok && threw);
    }

    private static void testCohenKappaPerfectIsOne() {
        CohenKappa k = new CohenKappa(2, 1000);
        for (int i = 0; i < 200; i++) k.update(i % 2, i % 2);
        boolean ok = Math.abs(k.getKappa() - 1.0) < 1e-9 && k.getAccuracy() == 1.0;
        report("CohenKappa perfect = 1.0", ok);
    }

    private static void testCohenKappaRandomIsZeroish() {
        CohenKappa k = new CohenKappa(2, 10_000);
        long seed = 42;
        java.util.Random rng = new java.util.Random(seed);
        for (int i = 0; i < 5000; i++) k.update(rng.nextInt(2), rng.nextInt(2));
        boolean ok = Math.abs(k.getKappa()) < 0.1;
        report("CohenKappa random ≈ 0 (got=" + k.getKappa() + ")", ok);
    }

    private static void testCohenKappaSlidesWindow() {
        CohenKappa k = new CohenKappa(2, 50);
        for (int i = 0; i < 50; i++) k.update(0, 1);
        for (int i = 0; i < 50; i++) k.update(0, 0);
        boolean ok = k.getWindowCount() == 50 && k.getAccuracy() == 1.0;
        report("CohenKappa slides window", ok);
    }

    private static void testCohenKappaIgnoresOutOfRange() {
        CohenKappa k = new CohenKappa(2, 100);
        k.update(0, 0);
        k.update(-1, 0);
        k.update(0, 99);
        boolean ok = k.getWindowCount() == 1 && k.getAccuracy() == 1.0;
        report("CohenKappa ignores out-of-range labels", ok);
    }

    private static void testCohenKappaResetAndBadCtor() {
        CohenKappa k = new CohenKappa(2, 10);
        k.update(0, 0);
        k.reset();
        boolean ok = k.getWindowCount() == 0 && k.getKappa() == 0.0;
        boolean t1 = false, t2 = false;
        try { new CohenKappa(1, 10); } catch (IllegalArgumentException e) { t1 = true; }
        try { new CohenKappa(2, 0); } catch (IllegalArgumentException e) { t2 = true; }
        report("CohenKappa reset + bad ctor", ok && t1 && t2);
    }

    private static void testTemporalKappaPerfectVsNoChange() {
        TemporalKappa tk = new TemporalKappa(1000);
        int[] labels = {0,1,0,1,0,1,0,1,0,1};
        for (int i = 0; i < 200; i++) {
            int y = labels[i % labels.length];
            tk.update(y, y);
        }
        boolean ok = tk.getKappaTemporal() > 0.9
                && tk.getNoChangeAccuracy() < 0.1;
        report("TemporalKappa perfect vs alternating no-change baseline (kT="
                + tk.getKappaTemporal() + ", noChangeAcc=" + tk.getNoChangeAccuracy() + ")", ok);
    }

    private static void testTemporalKappaWorseThanNoChangeIsNegative() {
        TemporalKappa tk = new TemporalKappa(1000);
        java.util.Random rng = new java.util.Random(123);
        int prev = 0;
        for (int i = 0; i < 2000; i++) {
            int y = (rng.nextDouble() < 0.3) ? 1 - prev : prev;
            tk.update(y, 1 - y);
            prev = y;
        }
        boolean ok = tk.getNoChangeAccuracy() > 0.5
                && tk.getNoChangeAccuracy() < 1.0
                && tk.getKappaTemporal() < 0;
        report("TemporalKappa < 0 when worse than no-change (kT="
                + tk.getKappaTemporal() + ", noChangeAcc="
                + tk.getNoChangeAccuracy() + ")", ok);
    }

    private static void testFeatureStabilityIgnoresNullAndEmpty() {
        FeatureStabilityRatio f = new FeatureStabilityRatio();

        f.update(null);
        boolean ok1 = f.getUpdateCount() == 0 && Double.isNaN(f.getLastRatio());

        f.update(new int[]{1, 2});
        boolean ok2 = f.getUpdateCount() == 0;

        f.update(new int[]{1, 2});
        boolean ok3 = f.getUpdateCount() == 1 && f.getLastRatio() == 1.0;

        f.update(new int[0]);
        boolean ok4 = f.getUpdateCount() == 2 && f.getLastRatio() == 0.0;

        f.update(new int[]{3, 4});
        boolean ok5 = f.getUpdateCount() == 2 && f.getLastRatio() == 0.0;

        report("FeatureStabilityRatio null is ignored, empty resets baseline "
                        + "(updates=" + f.getUpdateCount() + ", last=" + f.getLastRatio() + ")",
                ok1 && ok2 && ok3 && ok4 && ok5);
    }

    private static void testTemporalKappaNoChangeAccuracy() {
        TemporalKappa tk = new TemporalKappa(100);
        for (int i = 0; i < 100; i++) tk.update(7, 0);
        boolean ok = Math.abs(tk.getNoChangeAccuracy() - 0.99) < 1e-9
                || Math.abs(tk.getNoChangeAccuracy() - 1.0) < 1e-9;
        report("TemporalKappa no-change acc on constant labels (got="
                + tk.getNoChangeAccuracy() + ")", ok);
    }

    private static void testTemporalKappaResetAndBadCtor() {
        TemporalKappa tk = new TemporalKappa(10);
        tk.update(0, 1);
        tk.reset();
        boolean ok = tk.getKappaTemporal() == 0.0 && tk.getNoChangeAccuracy() == 0.0;
        boolean threw = false;
        try { new TemporalKappa(0); } catch (IllegalArgumentException e) { threw = true; }
        report("TemporalKappa reset + bad ctor", ok && threw);
    }

    private static void testFeatureStabilityIdenticalSelection() {
        FeatureStabilityRatio f = new FeatureStabilityRatio();
        f.update(new int[]{1, 2, 3});
        f.update(new int[]{1, 2, 3});
        f.update(new int[]{1, 2, 3});
        boolean ok = f.getUpdateCount() == 2
                && f.getLastRatio() == 1.0
                && f.getAverageRatio() == 1.0;
        report("FeatureStabilityRatio identical selections → 1.0", ok);
    }

    private static void testFeatureStabilityPartialOverlap() {
        FeatureStabilityRatio f = new FeatureStabilityRatio();
        f.update(new int[]{1, 2, 3, 4});
        f.update(new int[]{1, 2, 5, 6});
        boolean ok = Math.abs(f.getLastRatio() - 0.5) < 1e-9;
        f.update(new int[]{7, 8, 9, 10});
        boolean ok2 = f.getLastRatio() == 0.0
                && Math.abs(f.getAverageRatio() - 0.25) < 1e-9;
        report("FeatureStabilityRatio partial / disjoint overlap", ok && ok2);
    }

    private static void testFeatureStabilityFirstUpdateIsBaseline() {
        FeatureStabilityRatio f = new FeatureStabilityRatio();
        f.update(new int[]{1, 2});
        boolean ok = f.getUpdateCount() == 0 && Double.isNaN(f.getLastRatio());
        report("FeatureStabilityRatio first call only seeds baseline", ok);
    }

    private static void testFeatureStabilityReset() {
        FeatureStabilityRatio f = new FeatureStabilityRatio();
        f.update(new int[]{1, 2});
        f.update(new int[]{1, 3});
        f.reset();
        boolean ok = f.getUpdateCount() == 0 && Double.isNaN(f.getLastRatio())
                && Double.isNaN(f.getAverageRatio());
        report("FeatureStabilityRatio.reset clears state", ok);
    }

    private static void testRamHoursAccumulatesAndPeak() throws InterruptedException {
        RAMHours r = new RAMHours();
        r.start();
        r.sample(100L * 1024 * 1024);
        Thread.sleep(20);
        r.sample(200L * 1024 * 1024);
        Thread.sleep(20);
        r.sample(150L * 1024 * 1024);
        boolean ok = r.getRamHours() > 0.0
                && r.getPeakBytes() == 200L * 1024 * 1024
                && Math.abs(r.getPeakMB() - 200.0) < 1e-6
                && r.getElapsedHours() > 0.0;
        report("RAMHours accumulates + tracks peak (rh=" + r.getRamHours()
                + ", peakMB=" + r.getPeakMB() + ")", ok);
    }

    private static void testRamHoursSampleFromRuntimeAutoStarts() throws InterruptedException {
        RAMHours r = new RAMHours();
        r.sampleFromRuntime();
        Thread.sleep(10);
        r.sampleFromRuntime();
        boolean ok = r.getRamHours() >= 0.0 && r.getPeakBytes() > 0;
        report("RAMHours.sampleFromRuntime auto-starts", ok);
    }

    private static void testRamHoursReset() {
        RAMHours r = new RAMHours();
        r.start();
        r.sample(1024 * 1024);
        r.reset();
        boolean ok = r.getRamHours() == 0.0
                && r.getPeakBytes() == 0
                && r.getElapsedHours() == 0.0;
        report("RAMHours.reset clears state", ok);
    }

    private static void testRecoveryTimeRecoversWithinTolerance() {
        RecoveryTime rt = new RecoveryTime(0.05, 1000);
        for (int i = 0; i < 10; i++) { rt.update(0.9); rt.tick(); }
        rt.onDriftAlarm(0.9);
        for (int i = 0; i < 50; i++) { rt.update(0.4); rt.tick(); }
        rt.update(0.88);
        boolean ok = !rt.isTracking()
                && rt.getRecoveredCount() == 1
                && rt.getUnrecoveredCount() == 0
                && rt.getLastRecoveryTime() == 50
                && Math.abs(rt.getAverageRecoveryTime() - 50.0) < 1e-9;
        report("RecoveryTime recovers within tolerance (lastRT="
                + rt.getLastRecoveryTime() + ")", ok);
    }

    private static void testRecoveryTimeUnrecoveredAfterMaxWindow() {
        RecoveryTime rt = new RecoveryTime(0.05, 100);
        rt.onDriftAlarm(0.9);
        for (int i = 0; i < 200; i++) { rt.update(0.1); rt.tick(); }
        boolean ok = !rt.isTracking()
                && rt.getRecoveredCount() == 0
                && rt.getUnrecoveredCount() == 1
                && rt.getLastRecoveryTime() == -1;
        report("RecoveryTime gives up after maxRecoveryWindow", ok);
    }

    private static void testRecoveryTimeSecondAlarmCountsPreviousAsUnrecovered() {
        RecoveryTime rt = new RecoveryTime(0.05, 10_000);
        rt.onDriftAlarm(0.9);
        for (int i = 0; i < 30; i++) { rt.update(0.2); rt.tick(); }
        rt.onDriftAlarm(0.5);
        for (int i = 0; i < 10; i++) { rt.update(0.49); rt.tick(); }
        boolean ok = rt.getUnrecoveredCount() == 1
                && rt.getRecoveredCount() == 1
                && !rt.isTracking();
        report("RecoveryTime counts overlapping alarms (rec=" + rt.getRecoveredCount()
                + ", unrec=" + rt.getUnrecoveredCount() + ")", ok);
    }

    private static void testRecoveryTimeBadCtorAndReset() {
        boolean t1 = false, t2 = false, t3 = false;
        try { new RecoveryTime(-0.1, 100); } catch (IllegalArgumentException e) { t1 = true; }
        try { new RecoveryTime(1.5, 100); } catch (IllegalArgumentException e) { t2 = true; }
        try { new RecoveryTime(0.05, 0); } catch (IllegalArgumentException e) { t3 = true; }
        RecoveryTime rt = new RecoveryTime();
        rt.onDriftAlarm(0.5);
        rt.tick(); rt.update(0.1);
        rt.reset();
        boolean ok = !rt.isTracking() && rt.getRecoveredCount() == 0
                && rt.getUnrecoveredCount() == 0 && rt.getLastRecoveryTime() == -1;
        report("RecoveryTime bad ctor + reset", t1 && t2 && t3 && ok);
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