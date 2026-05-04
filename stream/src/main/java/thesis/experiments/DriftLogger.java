package thesis.experiments;

import thesis.detection.TwoLevelDriftDetector;
import thesis.evaluation.MetricsCollector;

import java.io.PrintWriter;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Locale;

public final class DriftLogger {

    private final PrintWriter out;
    private final String dataset;
    private final String variant;
    private final int seed;
    private final Deque<double[]> recentKappa = new ArrayDeque<>();
    private final int beforeWindow;
    private final int afterWindow;

    private static final class Pending {
        long alarmAt;
        double kappaBefore;
        String type;
        double startKappa;
        long deadline;
    }
    private final Deque<Pending> pending = new ArrayDeque<>();

    public DriftLogger(PrintWriter out, String dataset, String variant, int seed,
                       int beforeWindow, int afterWindow) {
        this.out = out;
        this.dataset = dataset;
        this.variant = variant;
        this.seed = seed;
        this.beforeWindow = beforeWindow;
        this.afterWindow = afterWindow;
    }

    public void tick(long n, double kappaNow) {
        recentKappa.addLast(new double[]{n, kappaNow});
        while (recentKappa.size() > beforeWindow) recentKappa.removeFirst();
        Pending head = pending.peekFirst();
        while (head != null && n >= head.deadline) {
            double recovered = -1;
            for (double[] p : recentKappa) {
                if (p[0] >= head.alarmAt && p[1] >= head.startKappa - 0.02) {
                    recovered = p[0] - head.alarmAt;
                    break;
                }
            }
            out.printf(Locale.ROOT, "%s,%s,%d,%d,%.4f,%.4f,%.0f,%s%n",
                    dataset, variant, seed,
                    head.alarmAt, head.kappaBefore, kappaNow,
                    recovered, head.type);
            out.flush();
            pending.removeFirst();
            head = pending.peekFirst();
        }
    }

    public void onAlarm(long n, MetricsCollector metrics, boolean featureLevel) {
        Pending p = new Pending();
        p.alarmAt = n;
        p.kappaBefore = avgRecent();
        p.type = featureLevel ? "FEATURE" : "GLOBAL";
        p.startKappa = metrics.snapshot().kappa;
        p.deadline = n + afterWindow;
        pending.addLast(p);
    }

    private double avgRecent() {
        if (recentKappa.isEmpty()) return Double.NaN;
        double s = 0;
        for (double[] p : recentKappa) s += p[1];
        return s / recentKappa.size();
    }

    public void flushPending(long lastN, double lastKappa) {
        for (Pending p : pending) {
            out.printf(Locale.ROOT, "%s,%s,%d,%d,%.4f,%.4f,%.0f,%s%n",
                    dataset, variant, seed,
                    p.alarmAt, p.kappaBefore, lastKappa, -1.0, p.type);
        }
        pending.clear();
        out.flush();
    }
}
