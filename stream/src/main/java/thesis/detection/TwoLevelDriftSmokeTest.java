package thesis.detection;

import java.util.Random;
import java.util.Set;

public class TwoLevelDriftSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("TWO-LEVEL DRIFT DETECTOR SMOKE TESTS");
        System.out.println("=".repeat(70));

        testNoFalseAlarmsOnStationary();
        testDetectsAbruptDriftAndLocalizesCorrectFeatures();
        testIgnoresNonDriftingFeatures();
        testCooldownLimitsConsecutiveAlarms();
        testResetClearsState();
        testRejectsBadInputs();
        testHddmAVariantWorks();
        testRepeatedDriftsAreLocalizedConsistently();
        testGlobalAlarmCountMatchesLocalizedCount();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    private static TwoLevelDriftDetector build(int F, TwoLevelDriftDetector.Level1Type type) {
        TwoLevelDriftDetector.Config cfg = new TwoLevelDriftDetector.Config(F);
        cfg.level1Type = type;
        cfg.level1Delta = 0.002;
        cfg.level1AlphaW = 0.01;
        cfg.kswinAlpha = 0.005;
        cfg.kswinWindowSize = 150;
        cfg.bhQ = 0.10;
        cfg.postDriftCooldown = 50;
        return new TwoLevelDriftDetector(cfg);
    }

    private static double[] gauss(Random r, int F, double[] means) {
        double[] v = new double[F];
        for (int i = 0; i < F; i++) v[i] = r.nextGaussian() + means[i];
        return v;
    }

    private static void testNoFalseAlarmsOnStationary() {
        int F = 8;
        TwoLevelDriftDetector det = build(F, TwoLevelDriftDetector.Level1Type.ADWIN);
        Random r = new Random(42);
        double[] means = new double[F];
        int alarms = 0, localized = 0;
        for (int t = 0; t < 5000; t++) {
            double err = r.nextDouble() < 0.10 ? 1.0 : 0.0;
            det.update(err, gauss(r, F, means));
            if (det.isGlobalDriftDetected()) alarms++;
            if (!det.getDriftingFeatureIndices().isEmpty()) localized++;
        }
        report("no false global alarms on stationary (alarms=" + alarms + ")", alarms <= 2);
        report("no false localizations on stationary (localized=" + localized + ")", localized <= 1);
    }

    private static void testDetectsAbruptDriftAndLocalizesCorrectFeatures() {
        int F = 10;
        TwoLevelDriftDetector det = build(F, TwoLevelDriftDetector.Level1Type.ADWIN);
        Random r = new Random(7);
        double[] means = new double[F];
        for (int t = 0; t < 1500; t++) det.update(r.nextDouble() < 0.10 ? 1.0 : 0.0, gauss(r, F, means));
        means[2] = 4.0; means[7] = -3.5;
        boolean detected = false;
        Set<Integer> drifting = null;
        for (int t = 0; t < 2000; t++) {
            double err = r.nextDouble() < 0.55 ? 1.0 : 0.0;
            det.update(err, gauss(r, F, means));
            if (!det.getDriftingFeatureIndices().isEmpty()) {
                detected = true;
                drifting = det.getDriftingFeatureIndices();
                break;
            }
        }
        report("global drift detected after abrupt error shift", detected);
        report("localized features include 2 and 7 (got " + drifting + ")",
                drifting != null && drifting.contains(2) && drifting.contains(7));
    }

    private static void testIgnoresNonDriftingFeatures() {
        int F = 12;
        TwoLevelDriftDetector det = build(F, TwoLevelDriftDetector.Level1Type.ADWIN);
        Random r = new Random(11);
        double[] means = new double[F];
        for (int t = 0; t < 1500; t++) det.update(r.nextDouble() < 0.10 ? 1.0 : 0.0, gauss(r, F, means));
        means[3] = 5.0;
        Set<Integer> drifting = null;
        for (int t = 0; t < 3000 && drifting == null; t++) {
            det.update(r.nextDouble() < 0.55 ? 1.0 : 0.0, gauss(r, F, means));
            if (!det.getDriftingFeatureIndices().isEmpty()) drifting = det.getDriftingFeatureIndices();
        }
        long fp = drifting == null ? 0 : drifting.stream().filter(i -> i != 3).count();
        report("only feature 3 is flagged (got " + drifting + ", fp=" + fp + ")",
                drifting != null && drifting.contains(3) && fp <= 1);
    }

    private static void testCooldownLimitsConsecutiveAlarms() {
        int F = 5;
        TwoLevelDriftDetector det = build(F, TwoLevelDriftDetector.Level1Type.ADWIN);
        Random r = new Random(13);
        double[] means = new double[F];
        for (int t = 0; t < 1500; t++) det.update(r.nextDouble() < 0.10 ? 1.0 : 0.0, gauss(r, F, means));
        for (int i = 0; i < F; i++) means[i] = 3.0;
        long before = det.getGlobalAlarms();
        for (int t = 0; t < 500; t++) det.update(r.nextDouble() < 0.7 ? 1.0 : 0.0, gauss(r, F, means));
        long after = det.getGlobalAlarms();
        report("alarms in burst window are bounded (delta=" + (after - before) + ")",
                (after - before) <= 6);
    }

    private static void testResetClearsState() {
        int F = 4;
        TwoLevelDriftDetector det = build(F, TwoLevelDriftDetector.Level1Type.ADWIN);
        Random r = new Random(14);
        double[] means = new double[F];
        for (int t = 0; t < 500; t++) det.update(0.0, gauss(r, F, means));
        det.reset();
        boolean ok = det.getUpdateCount() == 0
                && det.getGlobalAlarms() == 0
                && det.getDriftingFeatureIndices().isEmpty()
                && !det.isGlobalDriftDetected();
        report("reset clears state", ok);
    }

    private static void testRejectsBadInputs() {
        boolean t1 = false, t2 = false, t3 = false;
        try { new TwoLevelDriftDetector(null); } catch (IllegalArgumentException e) { t1 = true; }
        try { new TwoLevelDriftDetector(new TwoLevelDriftDetector.Config(0)); } catch (IllegalArgumentException e) { t2 = true; }
        TwoLevelDriftDetector det = build(3, TwoLevelDriftDetector.Level1Type.ADWIN);
        try { det.update(0.0, new double[2]); } catch (IllegalArgumentException e) { t3 = true; }
        report("rejects bad inputs", t1 && t2 && t3);
    }

    private static void testHddmAVariantWorks() {
        int F = 6;
        TwoLevelDriftDetector det = build(F, TwoLevelDriftDetector.Level1Type.HDDM_A);
        Random r = new Random(15);
        double[] means = new double[F];
        for (int t = 0; t < 1500; t++) det.update(r.nextDouble() < 0.10 ? 1.0 : 0.0, gauss(r, F, means));
        means[1] = 4.0;
        boolean detected = false;
        for (int t = 0; t < 2000 && !detected; t++) {
            det.update(r.nextDouble() < 0.6 ? 1.0 : 0.0, gauss(r, F, means));
            if (det.isGlobalDriftDetected()) detected = true;
        }
        report("HDDM_A detects abrupt drift", detected);
    }

    private static void testRepeatedDriftsAreLocalizedConsistently() {
        int F = 8;
        TwoLevelDriftDetector det = build(F, TwoLevelDriftDetector.Level1Type.ADWIN);
        Random r = new Random(16);
        double[] means = new double[F];
        int hits = 0;
        for (int round = 0; round < 3; round++) {
            for (int t = 0; t < 1500; t++) det.update(r.nextDouble() < 0.10 ? 1.0 : 0.0, gauss(r, F, means));
            means[5] += 3.0;
            for (int t = 0; t < 1500; t++) {
                det.update(r.nextDouble() < 0.55 ? 1.0 : 0.0, gauss(r, F, means));
                if (det.getDriftingFeatureIndices().contains(5)) { hits++; break; }
            }
        }
        report("feature 5 localized across repeated drifts (hits=" + hits + "/3)", hits >= 2);
    }

    private static void testGlobalAlarmCountMatchesLocalizedCount() {
        int F = 5;
        TwoLevelDriftDetector det = build(F, TwoLevelDriftDetector.Level1Type.ADWIN);
        Random r = new Random(17);
        double[] means = new double[F];
        for (int t = 0; t < 1500; t++) det.update(r.nextDouble() < 0.10 ? 1.0 : 0.0, gauss(r, F, means));
        means[0] = 5.0;
        for (int t = 0; t < 3000; t++) det.update(r.nextDouble() < 0.55 ? 1.0 : 0.0, gauss(r, F, means));
        long g = det.getGlobalAlarms();
        long l = det.getLocalizedAlarms();
        report("localized <= global alarms (g=" + g + ", l=" + l + ")", l <= g && g >= 1);
    }

    private static void report(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  [PASSED] " + name); }
        else    { failed++; System.out.println("  [FAILED] " + name); }
    }
}