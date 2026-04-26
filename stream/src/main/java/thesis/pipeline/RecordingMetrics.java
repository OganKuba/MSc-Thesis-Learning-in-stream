package thesis.pipeline;

import thesis.detection.TwoLevelDriftDetector;
import thesis.selection.FeatureSelector;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Arrays;

public class RecordingMetrics extends StreamMetrics {

    private final BufferedWriter out;
    private final int sampleEvery;
    private final FeatureSelector selector;

    private StreamPipeline pipeline;
    private TwoLevelDriftDetector detector;

    private long n;
    private long correct;
    private long correctNoChange;        // baseline: predict previous label
    private int  prevY = Integer.MIN_VALUE;
    private long noChangeDenom;

    // per-class counts for kappa
    private long[] confTrue  = new long[0];
    private long[] confPred  = new long[0];

    // RAM-hours: ∫ memBytes dt   (bytes·hours)
    private long lastSampleNanos = -1;
    private double ramByteHours;

    // feature stability
    private int[] prevSelection;
    private long stableSamples;
    private long totalSelectionSamples;

    // drift / recovery
    private long driftCount;
    private long lastDriftInstance = -1;
    private long lastRecoveryInstance = -1;
    private double accAtDrift;
    private static final double RECOVERY_THRESHOLD = 0.01; // within 1% of pre-drift acc

    public RecordingMetrics(BufferedWriter out, int sampleEvery, FeatureSelector selector) {
        this.out = out;
        this.sampleEvery = Math.max(1, sampleEvery);
        this.selector = selector;
    }

    void bindPipeline(StreamPipeline p, TwoLevelDriftDetector d) {
        this.pipeline = p;
        this.detector = d;
    }

    @Override
    public void update(int y, int yhat, long elapsedNanos) {
        super.update(y, yhat, elapsedNanos);
        n++;

        // accuracy
        if (y == yhat) correct++;

        // confusion stats for Cohen's kappa
        int maxC = Math.max(y, yhat) + 1;
        if (maxC > confTrue.length) {
            confTrue = Arrays.copyOf(confTrue, maxC);
            confPred = Arrays.copyOf(confPred, maxC);
        }
        confTrue[y]++;
        confPred[yhat]++;

        // kappa-temporal baseline (no-change classifier)
        if (prevY != Integer.MIN_VALUE) {
            if (y == prevY) correctNoChange++;
            noChangeDenom++;
        }
        prevY = y;

        // drift bookkeeping
        if (detector != null && detector.isGlobalDriftDetected()) {
            driftCount++;
            lastDriftInstance = n;
            accAtDrift = accuracy();
            lastRecoveryInstance = -1;
        } else if (lastDriftInstance > 0 && lastRecoveryInstance < 0
                && accuracy() >= accAtDrift - RECOVERY_THRESHOLD) {
            lastRecoveryInstance = n;
        }

        if (n % sampleEvery == 0) writeRow();
    }

    void flushFinalRow() {
        if (n % sampleEvery != 0) writeRow();
    }

    private void writeRow() {
        // RAM-hours integration since previous sample
        long now = System.nanoTime();
        Runtime r = Runtime.getRuntime();
        long mem = r.totalMemory() - r.freeMemory();
        recordMemory(mem);
        if (lastSampleNanos > 0) {
            double hours = (now - lastSampleNanos) / 3_600_000_000_000.0;
            ramByteHours += mem * hours;
        }
        lastSampleNanos = now;

        // feature-stability ratio: fraction of samples where selection == previous
        int[] sel = selector.getCurrentSelection();
        if (prevSelection != null) {
            totalSelectionSamples++;
            if (Arrays.equals(sel, prevSelection)) stableSamples++;
        }
        prevSelection = (sel == null) ? null : sel.clone();
        double stability = totalSelectionSamples == 0
                ? 1.0 : (double) stableSamples / totalSelectionSamples;

        double recovery = (lastDriftInstance > 0 && lastRecoveryInstance > 0)
                ? (lastRecoveryInstance - lastDriftInstance) : Double.NaN;

        try {
            out.write(String.format(
                    "%d,%.6f,%.6f,%.6f,%.6e,%.6f,%d,%s",
                    n,
                    cohenKappa(),
                    kappaTemporal(),
                    accuracy(),
                    ramByteHours / (1024.0 * 1024.0 * 1024.0), // GB·h
                    stability,
                    driftCount,
                    Double.isNaN(recovery) ? "" : Long.toString((long) recovery)));
            out.newLine();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /* ---- metric helpers ---- */

    private double accuracy() { return n == 0 ? 0.0 : (double) correct / n; }

    private double cohenKappa() {
        if (n == 0) return 0.0;
        double pe = 0.0;
        int k = Math.min(confTrue.length, confPred.length);
        for (int i = 0; i < k; i++) pe += (confTrue[i] / (double) n) * (confPred[i] / (double) n);
        double po = accuracy();
        return (pe == 1.0) ? 0.0 : (po - pe) / (1.0 - pe);
    }

    /** "kappa_per" interpreted as kappa-temporal vs. no-change classifier. */
    private double kappaTemporal() {
        if (noChangeDenom == 0) return 0.0;
        double pNoChange = (double) correctNoChange / noChangeDenom;
        if (pNoChange == 1.0) return 0.0;
        return (accuracy() - pNoChange) / (1.0 - pNoChange);
    }
}