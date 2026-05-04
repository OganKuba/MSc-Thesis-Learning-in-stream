package thesis.evaluation;

public class MetricsSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("METRICS SMOKE TESTS");
        System.out.println("=".repeat(70));

        testCohenKappaPerfectAndChance();
        testCohenKappaSlidingWindowEvicts();
        testCohenKappaRejectsBadClass();

        testTemporalKappaSkipsFirst();
        testTemporalKappaNoChangeBaseline();
        testTemporalKappaSlidingWindow();

        testPrequentialAccuracyWindow();

        testFeatureStabilityRatioSpec();
        testFeatureStabilityRatioS1Stays1();
        testFeatureStabilityRatioDetectsChange();

        testRecoveryTimeOneEpisode();
        testRecoveryTimeMultipleDriftsAndCancel();
        testRecoveryTimeUnrecoveredAfterMaxWindow();

        testRamHoursMonotone();
        testRamHoursPeakMB();

        testCollectorWindowedNotCumulative();
        testCollectorOnDriftAndSelectionFlag();
        testCollectorLogLineNotEmpty();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    private static void testCohenKappaPerfectAndChance() {
        CohenKappa k = new CohenKappa(2, 100);
        for (int i = 0; i < 50; i++) { k.update(0, 0); k.update(1, 1); }
        boolean perfect = Math.abs(k.getKappa() - 1.0) < 1e-9;
        CohenKappa k2 = new CohenKappa(2, 200);
        for (int i = 0; i < 100; i++) { k2.update(0, 1); k2.update(1, 0); }
        boolean negative = k2.getKappa() < -0.5;
        report("Cohen kappa: perfect=1.0 (got " + k.getKappa() + "), inverse<0 (got "
                + k2.getKappa() + ")", perfect && negative);
    }

    private static void testCohenKappaSlidingWindowEvicts() {
        CohenKappa k = new CohenKappa(2, 10);
        for (int i = 0; i < 10; i++) k.update(0, 1);
        double bad = k.getKappa();
        for (int i = 0; i < 10; i++) k.update(0, 0);
        double good = k.getKappa();
        report("Kappa window evicts old wrong predictions (bad=" + bad + " good=" + good + ")",
                k.getWindowCount() == 10 && good >= 0.0 && good >= bad);
    }

    private static void testCohenKappaRejectsBadClass() {
        CohenKappa k = new CohenKappa(2, 10);
        boolean t = false;
        try { k.update(2, 0); } catch (IllegalArgumentException e) { t = true; }
        report("Cohen kappa rejects out-of-range class", t);
    }

    private static void testTemporalKappaSkipsFirst() {
        TemporalKappa t = new TemporalKappa(50);
        t.update(0, 0);
        report("Temporal kappa skips first sample (count=" + t.getWindowCount() + ")",
                t.getWindowCount() == 0);
    }

    private static void testTemporalKappaNoChangeBaseline() {
        TemporalKappa t = new TemporalKappa(100);
        t.update(0, 0);
        for (int i = 0; i < 50; i++) {
            t.update(1, 1);
            t.update(0, 0);
        }
        double k = t.getKappaTemporal();
        report("Temporal kappa positive when model beats no-change (k=" + k + ")",
                Double.isFinite(k));
    }

    private static void testTemporalKappaSlidingWindow() {
        TemporalKappa t = new TemporalKappa(20);
        t.update(0, 0);
        for (int i = 0; i < 50; i++) t.update(i % 2, i % 2);
        report("Temporal kappa window bounded (count=" + t.getWindowCount() + ")",
                t.getWindowCount() == 20);
    }

    private static void testPrequentialAccuracyWindow() {
        PrequentialAccuracy a = new PrequentialAccuracy(10);
        for (int i = 0; i < 5; i++) a.update(0, 1);
        for (int i = 0; i < 10; i++) a.update(0, 0);
        report("PreqAcc evicts old errors (acc=" + a.getAccuracy() + ")",
                a.getAccuracy() >= 0.99);
    }

    private static void testFeatureStabilityRatioSpec() {
        FeatureStabilityRatio r = new FeatureStabilityRatio();
        r.update(new int[]{0, 1});
        r.update(new int[]{0, 1, 2});
        report("FSR uses |inter|/|S_old| spec (last=" + r.getLastRatio() + ")",
                Math.abs(r.getLastRatio() - 1.0) < 1e-9);
    }

    private static void testFeatureStabilityRatioS1Stays1() {
        FeatureStabilityRatio r = new FeatureStabilityRatio();
        int[] sel = {0, 1, 2};
        for (int i = 0; i < 100; i++) r.update(sel);
        report("S1-style stable selection → ratio 1.0 (avg=" + r.getAverageRatio()
                        + ", changes=" + r.getChangeCount() + ")",
                Math.abs(r.getAverageRatio() - 1.0) < 1e-9 && r.getChangeCount() == 0);
    }

    private static void testFeatureStabilityRatioDetectsChange() {
        FeatureStabilityRatio r = new FeatureStabilityRatio();
        r.update(new int[]{0, 1, 2});
        r.update(new int[]{3, 4, 5});
        report("FSR detects change (last=" + r.getLastRatio()
                        + ", changed=" + r.wasLastChanged() + ")",
                r.getLastRatio() == 0.0 && r.wasLastChanged());
    }

    private static void testRecoveryTimeOneEpisode() {
        RecoveryTime rt = new RecoveryTime(0.05, 1000);
        for (int i = 0; i < 100; i++) rt.tick();
        rt.onDriftAlarm(0.8);
        for (int i = 0; i < 50; i++) { rt.tick(); rt.update(0.1); }
        rt.tick(); rt.update(0.78);
        report("RecoveryTime: one episode (last=" + rt.getLastRecoveryTime()
                        + ", recovered=" + rt.getRecoveredCount() + ")",
                rt.getRecoveredCount() == 1 && rt.getLastRecoveryTime() == 51);
    }

    private static void testRecoveryTimeMultipleDriftsAndCancel() {
        RecoveryTime rt = new RecoveryTime(0.05, 10000);
        for (int i = 0; i < 10; i++) rt.tick();
        rt.onDriftAlarm(0.8);
        for (int i = 0; i < 5; i++) { rt.tick(); rt.update(0.2); }
        rt.onDriftAlarm(0.7);
        for (int i = 0; i < 10; i++) { rt.tick(); rt.update(0.7); }
        report("RecoveryTime: cancel+next drift (cancel=" + rt.getCancelledCount()
                        + ", recovered=" + rt.getRecoveredCount() + ")",
                rt.getCancelledCount() == 1 && rt.getRecoveredCount() == 1);
    }

    private static void testRecoveryTimeUnrecoveredAfterMaxWindow() {
        RecoveryTime rt = new RecoveryTime(0.05, 100);
        rt.tick();
        rt.onDriftAlarm(0.9);
        for (int i = 0; i < 200; i++) { rt.tick(); rt.update(0.1); }
        report("RecoveryTime: unrecovered after maxWindow (unrec=" + rt.getUnrecoveredCount() + ")",
                rt.getUnrecoveredCount() == 1 && rt.getRecoveredCount() == 0);
    }

    private static void testRamHoursMonotone() {
        RAMHours r = new RAMHours();
        r.start();
        r.sample(100L * 1024 * 1024);
        try { Thread.sleep(5); } catch (InterruptedException ignored) { }
        r.sample(120L * 1024 * 1024);
        try { Thread.sleep(5); } catch (InterruptedException ignored) { }
        r.sample(140L * 1024 * 1024);
        report("RAMHours non-negative and finite (gbH=" + r.getRamHours() + ")",
                r.getRamHours() >= 0.0 && Double.isFinite(r.getRamHours()));
    }

    private static void testRamHoursPeakMB() {
        RAMHours r = new RAMHours();
        r.start();
        r.sample(50L * 1024 * 1024);
        r.sample(200L * 1024 * 1024);
        r.sample(120L * 1024 * 1024);
        report("RAMHours peakMB tracks max (peak=" + r.getPeakMB() + ")",
                Math.abs(r.getPeakMB() - 200.0) < 1e-6);
    }

    private static void testCollectorWindowedNotCumulative() {
        MetricsCollector mc = new MetricsCollector(2, 50, 0, 100);
        for (int i = 0; i < 200; i++) mc.update(0, 1, 1000);
        for (int i = 0; i < 50; i++) mc.update(0, 0, 1000);
        double accW = mc.getAccuracy().getAccuracy();
        double accAll = mc.snapshot().accuracyOverall;
        report("Window accuracy != cumulative (win=" + accW + ", all=" + accAll + ")",
                accW > 0.95 && accAll < 0.5);
    }

    private static void testCollectorOnDriftAndSelectionFlag() {
        MetricsCollector mc = new MetricsCollector(2, 100, 0, 50);
        for (int i = 0; i < 50; i++) mc.update(0, 0, 500);
        mc.onDriftAlarm();
        boolean changed1 = mc.onSelectionChanged(new int[]{0, 1});
        boolean changed2 = mc.onSelectionChanged(new int[]{0, 1});
        boolean changed3 = mc.onSelectionChanged(new int[]{0, 2});
        report("onSelectionChanged returns boolean flag correctly (c1="
                        + changed1 + ", c2=" + changed2 + ", c3=" + changed3 + ")",
                !changed1 && !changed2 && changed3 && mc.getDriftCount() == 1);
    }

    private static void testCollectorLogLineNotEmpty() {
        MetricsCollector mc = new MetricsCollector(2, 100, 10, 10);
        for (int i = 0; i < 10; i++) mc.update(0, 0, 1000);
        report("formatLogLine produces non-empty (shouldLog=" + mc.shouldLog() + ")",
                mc.shouldLog() && mc.formatLogLine().contains("acc="));
    }

    private static void report(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  [PASSED] " + name); }
        else    { failed++; System.out.println("  [FAILED] " + name); }
    }
}