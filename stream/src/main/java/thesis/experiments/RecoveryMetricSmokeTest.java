package thesis.experiments;

import thesis.evaluation.MetricsCollector;

import java.util.List;
import java.util.Set;

/**
 * Drives {@link RunDetailedRecorder} through controlled accuracy trajectories to pin down the
 * recovery-episode semantics.
 *
 * <p>These cases exist because the previous implementation reported {@code recovery_length == 1}
 * for 64–90 % of all episodes: it sampled the baseline at alarm time (already degraded), only
 * evaluated at window boundaries (so "1 window" was the shortest representable answer), and
 * declared recovery on the first boundary without ever requiring a drop to have happened.
 */
public final class RecoveryMetricSmokeTest {

    private static final int WINDOW = 100;
    private static int passed;
    private static int failed;

    public static void main(String[] args) {
        testNoDropIsNotReportedAsInstantRecovery();
        testRealDropRecoversAndIsMeasuredInInstances();
        testNewAlarmCancelsOpenEpisode();
        testUnrecoveredIsDistinctFromNoDrop();

        System.out.println("======================================================================");
        System.out.println("RESULT: " + passed + " passed, " + failed + " failed");
        System.out.println("======================================================================");
        if (failed > 0) System.exit(1);
    }

    /** A harness pairing a recorder with a metrics collector, fed one instance at a time. */
    private static final class Harness {
        final RunDetailedRecorder rec = new RunDetailedRecorder(
                "T", "ds", "var", "model", "S1", "ADWIN", 1, 3, WINDOW);
        final MetricsCollector metrics = new MetricsCollector(2, WINDOW, 0, 1000);
        final int[] selection = {0, 1};
        long n;

        Harness() { rec.onInitialSelection(0, selection); }

        /** Feed {@code count} instances; {@code correct} controls whether the prediction matches. */
        void feed(int count, boolean correct) { feed(count, correct, false); }

        void feed(int count, boolean correct, boolean alarmOnFirst) {
            for (int i = 0; i < count; i++) {
                int yTrue = 1;
                int yPred = correct ? 1 : 0;
                boolean alarm = alarmOnFirst && i == 0;
                n++;
                metrics.update(yTrue, yPred, 1000L);   // runner order: metrics first, then recorder
                rec.onInstance(n, yTrue, yPred, selection, alarm, Set.of(), metrics);
            }
        }

        List<RunDetailedRecorder.RecoveryRow> episodes() { return rec.recoveries; }
    }

    private static void testNoDropIsNotReportedAsInstantRecovery() {
        Harness h = new Harness();
        h.feed(300, true);                 // steady state, accuracy 1.0
        h.feed(1, true, /*alarm=*/true);   // alarm fires but nothing actually degrades
        h.feed(400, true);

        List<RunDetailedRecorder.RecoveryRow> eps = h.episodes();
        boolean ok = eps.size() == 1
                && eps.get(0).outcome == RunDetailedRecorder.RecoveryOutcome.NO_DROP
                && eps.get(0).recoveryLength == -1;
        report("alarm without degradation -> NO_DROP (not 'recovered in 1 window')"
                + detail(eps), ok);
    }

    private static void testRealDropRecoversAndIsMeasuredInInstances() {
        Harness h = new Harness();
        h.feed(300, true);                 // baseline accuracy 1.0
        h.feed(1, true, /*alarm=*/true);
        h.feed(20, false);                 // real degradation: 20 errors inside a 100-wide window
        h.feed(400, true);                 // errors age out, accuracy climbs back

        List<RunDetailedRecorder.RecoveryRow> eps = h.episodes();
        boolean ok = eps.size() == 1;
        if (ok) {
            RunDetailedRecorder.RecoveryRow e = eps.get(0);
            ok = e.outcome == RunDetailedRecorder.RecoveryOutcome.RECOVERED
                    && e.recoveryLength > 1          // NOT the degenerate "1"
                    && e.recoveryLength < 10_000
                    && e.instancesToDrop >= 0
                    && e.dropInstance > e.driftInstance
                    && e.maxDrop > 0.0;              // the dip is actually recorded
        }
        report("real drop -> RECOVERED, length in instances, maxDrop > 0" + detail(eps), ok);
    }

    private static void testNewAlarmCancelsOpenEpisode() {
        Harness h = new Harness();
        h.feed(300, true);
        h.feed(1, true, /*alarm=*/true);
        h.feed(30, false);                 // episode is open and degraded
        h.feed(1, false, /*alarm=*/true);  // second alarm arrives before recovery
        h.feed(300, true);

        List<RunDetailedRecorder.RecoveryRow> eps = h.episodes();
        boolean ok = eps.size() == 2
                && eps.get(0).outcome == RunDetailedRecorder.RecoveryOutcome.CANCELLED
                && eps.get(0).recoveryLength == -1;
        report("second alarm -> first episode CANCELLED, distinct from UNRECOVERED"
                + detail(eps), ok);
    }

    private static void testUnrecoveredIsDistinctFromNoDrop() {
        Harness h = new Harness();
        h.feed(300, true);
        h.feed(1, true, /*alarm=*/true);
        h.feed(12_000, false);             // never comes back within the 10k-instance budget
        h.rec.finalizeAtEnd(h.n, h.metrics);

        List<RunDetailedRecorder.RecoveryRow> eps = h.episodes();
        boolean ok = !eps.isEmpty()
                && eps.get(0).outcome == RunDetailedRecorder.RecoveryOutcome.UNRECOVERED
                && eps.get(0).maxDrop > 0.5;
        report("permanent collapse -> UNRECOVERED with a large maxDrop" + detail(eps), ok);
    }

    private static String detail(List<RunDetailedRecorder.RecoveryRow> eps) {
        if (eps.isEmpty()) return " [brak epizodow]";
        RunDetailedRecorder.RecoveryRow e = eps.get(0);
        return String.format(" [n=%d, outcome=%s, len=%d, toDrop=%d, maxDrop=%.4f]",
                eps.size(), e.outcome, e.recoveryLength, e.instancesToDrop, e.maxDrop);
    }

    private static void report(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  [PASSED] " + name); }
        else    { failed++; System.out.println("  [FAILED] " + name); }
    }
}
