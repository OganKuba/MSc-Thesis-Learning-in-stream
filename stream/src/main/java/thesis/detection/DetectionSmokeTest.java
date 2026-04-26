package thesis.detection;

import java.util.Random;
import java.util.Set;

public class DetectionSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("DETECTION SMOKE TESTS");
        System.out.println("=".repeat(70));

        testAdwinDetectsAbruptDrift();
        testAdwinNoDriftOnStationary();
        testAdwinResetClearsState();
        testAdwinRejectsBadDelta();

        testKswinSingleDetectsDistributionShift();
        testKswinSingleNoDriftOnStationary();
        testKswinSingleNotReadyBeforeFill();
        testKswinSingleRejectsBadParams();
        testKswinSinglePromoteCurrentToReference();

        testPerFeatureKswinFlagsOnlyDriftingFeatures();
        testPerFeatureKswinNoFlagsOnStationary();
        testPerFeatureKswinFdrControlsFalsePositives();
        testPerFeatureKswinResetFeature();
        testPerFeatureKswinRejectsWrongVectorLength();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    private static void testAdwinDetectsAbruptDrift() {
        ADWINChangeDetector d = new ADWINChangeDetector(0.002);
        Random r = new Random(1);
        for (int i = 0; i < 2000; i++) d.update(r.nextGaussian() * 0.1);
        boolean detected = false;
        for (int i = 0; i < 2000; i++) {
            d.update(5.0 + r.nextGaussian() * 0.1);
            if (d.isChangeDetected()) { detected = true; break; }
        }
        report("ADWIN detects abrupt drift", detected);
    }

    private static void testAdwinNoDriftOnStationary() {
        ADWINChangeDetector d = new ADWINChangeDetector(0.002);
        Random r = new Random(2);
        int detections = 0;
        for (int i = 0; i < 5000; i++) {
            d.update(r.nextGaussian());
            if (d.isChangeDetected()) detections++;
        }
        report("ADWIN no false drift on stationary stream (got " + detections + ")", detections <= 2);
    }

    private static void testAdwinResetClearsState() {
        ADWINChangeDetector d = new ADWINChangeDetector(0.002);
        for (int i = 0; i < 500; i++) d.update(1.0);
        d.reset();
        boolean ok = !d.isChangeDetected() && d.getWindowLength() == 0;
        report("ADWIN reset clears state", ok);
    }

    private static void testAdwinRejectsBadDelta() {
        boolean threw = false;
        try { new ADWINChangeDetector(0.0); } catch (IllegalArgumentException e) { threw = true; }
        boolean threw2 = false;
        try { new ADWINChangeDetector(1.0); } catch (IllegalArgumentException e) { threw2 = true; }
        report("ADWIN rejects invalid delta", threw && threw2);
    }

    private static void testKswinSingleDetectsDistributionShift() {
        KSWINSingleFeature k = new KSWINSingleFeature(100, 0.005);
        Random r = new Random(3);
        for (int i = 0; i < 100; i++) k.update(r.nextGaussian());
        for (int i = 0; i < 100; i++) k.update(r.nextGaussian());
        boolean before = k.testDrift();
        for (int i = 0; i < 100; i++) k.update(r.nextGaussian() + 3.0);
        boolean after = k.testDrift();
        report("KSWINSingle drift after shift (before=" + before + ", after=" + after + ")",
                !before && after);
    }

    private static void testKswinSingleNoDriftOnStationary() {
        KSWINSingleFeature k = new KSWINSingleFeature(100, 0.001);
        Random r = new Random(4);
        for (int i = 0; i < 100; i++) k.update(r.nextGaussian());
        int flags = 0;
        for (int i = 0; i < 1000; i++) {
            k.update(r.nextGaussian());
            if (k.testDrift()) flags++;
        }
        report("KSWINSingle few false alarms on stationary (got " + flags + ")", flags <= 5);
    }

    private static void testKswinSingleNotReadyBeforeFill() {
        KSWINSingleFeature k = new KSWINSingleFeature(50, 0.05);
        for (int i = 0; i < 30; i++) k.update(i);
        boolean notReady = !k.isReady() && !k.testDrift();
        for (int i = 0; i < 70; i++) k.update(i);
        boolean ready = k.isReady();
        report("KSWINSingle isReady transitions correctly", notReady && ready);
    }

    private static void testKswinSingleRejectsBadParams() {
        boolean t1 = false, t2 = false, t3 = false;
        try { new KSWINSingleFeature(5, 0.05); } catch (IllegalArgumentException e) { t1 = true; }
        try { new KSWINSingleFeature(100, 0.0); } catch (IllegalArgumentException e) { t2 = true; }
        try { new KSWINSingleFeature(100, 1.0); } catch (IllegalArgumentException e) { t3 = true; }
        report("KSWINSingle rejects invalid params", t1 && t2 && t3);
    }

    private static void testKswinSinglePromoteCurrentToReference() {
        KSWINSingleFeature k = new KSWINSingleFeature(50, 0.05);
        Random r = new Random(5);
        for (int i = 0; i < 50; i++) k.update(r.nextGaussian());
        for (int i = 0; i < 50; i++) k.update(r.nextGaussian() + 5.0);
        k.testDrift();
        k.promoteCurrentToReference();
        boolean cleared = !k.isDrift() && k.getPValue() == 1.0 && !k.isReady();
        report("KSWINSingle promoteCurrentToReference resets state", cleared);
    }

    private static void testPerFeatureKswinFlagsOnlyDriftingFeatures() {
        int F = 10;
        PerFeatureKSWIN det = new PerFeatureKSWIN(F, 0.001, 100, 0.05);
        Random r = new Random(6);
        for (int t = 0; t < 200; t++) {
            double[] v = new double[F];
            for (int i = 0; i < F; i++) v[i] = r.nextGaussian();
            det.update(v);
        }
        for (int t = 0; t < 200; t++) {
            double[] v = new double[F];
            for (int i = 0; i < F; i++) {
                v[i] = (i == 2 || i == 7) ? r.nextGaussian() + 4.0 : r.nextGaussian();
            }
            det.update(v);
        }
        Set<Integer> flagged = det.getDriftingFeatures();
        boolean hits = flagged.contains(2) && flagged.contains(7);
        long fp = flagged.stream().filter(i -> i != 2 && i != 7).count();
        report("PerFeatureKSWIN flags drifting features (flagged=" + flagged + ")",
                hits && fp <= 1);
    }

    private static void testPerFeatureKswinNoFlagsOnStationary() {
        int F = 8;
        PerFeatureKSWIN det = new PerFeatureKSWIN(F, 0.001, 100, 0.05);
        Random r = new Random(7);
        for (int t = 0; t < 1000; t++) {
            double[] v = new double[F];
            for (int i = 0; i < F; i++) v[i] = r.nextGaussian();
            det.update(v);
        }
        Set<Integer> flagged = det.getDriftingFeatures();
        report("PerFeatureKSWIN no flags on stationary (flagged=" + flagged + ")",
                flagged.isEmpty());
    }

    private static void testPerFeatureKswinFdrControlsFalsePositives() {
        int F = 20;
        PerFeatureKSWIN det = new PerFeatureKSWIN(F, 0.05, 100, 0.05);
        Random r = new Random(8);
        for (int t = 0; t < 400; t++) {
            double[] v = new double[F];
            for (int i = 0; i < F; i++) v[i] = r.nextGaussian();
            det.update(v);
        }
        Set<Integer> raw = det.getRawDriftingFeatures();
        Set<Integer> bh  = det.getDriftingFeatures();
        report("PerFeatureKSWIN BH-FDR <= raw (raw=" + raw.size() + ", bh=" + bh.size() + ")",
                bh.size() <= raw.size());
    }

    private static void testPerFeatureKswinResetFeature() {
        int F = 4;
        PerFeatureKSWIN det = new PerFeatureKSWIN(F, 0.05, 50, 0.1);
        Random r = new Random(9);
        for (int t = 0; t < 200; t++) {
            double[] v = new double[F];
            for (int i = 0; i < F; i++) v[i] = r.nextGaussian();
            det.update(v);
        }
        det.resetFeature(1);
        boolean ok = !det.getDetectors()[1].isReady() && det.getDetectors()[0].isReady();
        report("PerFeatureKSWIN resetFeature only resets one", ok);
    }

    private static void testPerFeatureKswinRejectsWrongVectorLength() {
        PerFeatureKSWIN det = new PerFeatureKSWIN(5, 0.05, 50);
        boolean threw = false;
        try { det.update(new double[3]); } catch (IllegalArgumentException e) { threw = true; }
        boolean threw2 = false;
        try { new PerFeatureKSWIN(0, 0.05, 50); } catch (IllegalArgumentException e) { threw2 = true; }
        boolean threw3 = false;
        try { new PerFeatureKSWIN(5, 0.05, 50, 0.0); } catch (IllegalArgumentException e) { threw3 = true; }
        report("PerFeatureKSWIN rejects bad inputs", threw && threw2 && threw3);
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