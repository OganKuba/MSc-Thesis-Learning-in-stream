package thesis.pipeline;

import thesis.detection.TwoLevelDriftDetector;
import thesis.selection.FeatureSelector;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.Locale;

public class RecordingMetrics extends StreamMetrics {

    private static final double RECOVERY_TOLERANCE = 0.01;
    private static final int    RECOVERY_WINDOW    = 500;

    private final BufferedWriter out;
    private final int sampleEvery;
    private final FeatureSelector selector;
    private final String dataset;
    private final String variant;
    private final String modelName;

    private TwoLevelDriftDetector detector;
    private StreamPipeline pipeline;

    private long n;
    private long correct;

    private long correctNoChange;
    private long noChangeDenom;
    private int  prevY = Integer.MIN_VALUE;

    private long[] confTrue;
    private long[] confPred;

    private final boolean[] recentCorrect;
    private int recentIdx;
    private int recentFilled;
    private long recentCorrectSum;

    private long lastSampleNanos = -1;
    private double ramByteHours;

    private int[] prevSelection;
    private long stableSamples;
    private long totalSelectionSamples;

    private boolean prevDriftFlag;
    private long driftCount;
    private long lastDriftInstance = -1;
    private long lastRecoveryInstance = -1;
    private double accAtDrift;
    private long recoveredEpisodes;
    private long recoveryTimeSum;

    private boolean headerWritten;
    private int yTrueLast = -1;
    private int yPredLast = -1;
    private long lastTickNanos = -1;

    public RecordingMetrics(BufferedWriter out, int sampleEvery,
                            FeatureSelector selector, int numClasses) {
        this(out, sampleEvery, selector, numClasses, "", "", "");
    }

    public RecordingMetrics(BufferedWriter out, int sampleEvery,
                            FeatureSelector selector, int numClasses,
                            String dataset, String variant, String modelName) {
        if (out == null)      throw new IllegalArgumentException("out == null");
        if (selector == null) throw new IllegalArgumentException("selector == null");
        if (numClasses <= 0)  throw new IllegalArgumentException("numClasses <= 0");
        this.out = out;
        this.sampleEvery = Math.max(1, sampleEvery);
        this.selector = selector;
        this.dataset = dataset == null ? "" : dataset;
        this.variant = variant == null ? "" : variant;
        this.modelName = modelName == null ? "" : modelName;
        this.confTrue = new long[numClasses];
        this.confPred = new long[numClasses];
        this.recentCorrect = new boolean[RECOVERY_WINDOW];
    }

    void bindPipeline(StreamPipeline p, TwoLevelDriftDetector d) {
        this.pipeline = p;
        this.detector = d;
    }

    private void writeHeaderIfNeeded() {
        if (headerWritten) return;
        try {
            out.write("instance_num,dataset,variant,model,selector,y_true,y_pred,error,"
                    + "selected_features,selected_count,selection_changed,trigger_type,"
                    + "drift_alarm,warning_alarm,drifting_features,feature_scores_sample,"
                    + "kappa_window,kappa_temporal,accuracy_window,accuracy_cumulative,"
                    + "drift_count,recovery_time,feature_stability_ratio,ram_mb,processing_time_us");
            out.newLine();
            headerWritten = true;
        } catch (IOException e) { throw new UncheckedIOException(e); }
    }

    @Override
    public void update(int y, int yhat, long elapsedNanos) {
        super.update(y, yhat, elapsedNanos);
        if (y < 0 || yhat < 0) {
            throw new IllegalArgumentException("negative class index: y=" + y + " yhat=" + yhat);
        }
        n++;
        boolean hit = (y == yhat);
        if (hit) correct++;
        yTrueLast = y;
        yPredLast = yhat;
        lastTickNanos = elapsedNanos;

        int needed = Math.max(y, yhat) + 1;
        if (needed > confTrue.length) {
            confTrue = Arrays.copyOf(confTrue, needed);
            confPred = Arrays.copyOf(confPred, needed);
        }
        confTrue[y]++;
        confPred[yhat]++;

        if (prevY != Integer.MIN_VALUE) {
            if (y == prevY) correctNoChange++;
            noChangeDenom++;
        }
        prevY = y;

        if (recentFilled == RECOVERY_WINDOW) {
            if (recentCorrect[recentIdx]) recentCorrectSum--;
        } else {
            recentFilled++;
        }
        recentCorrect[recentIdx] = hit;
        if (hit) recentCorrectSum++;
        recentIdx = (recentIdx + 1) % RECOVERY_WINDOW;

        boolean driftNow = detector != null && detector.isGlobalDriftDetected();
        if (driftNow && !prevDriftFlag) {
            driftCount++;
            lastDriftInstance = n;
            accAtDrift = windowAccuracy();
            lastRecoveryInstance = -1;
        } else if (lastDriftInstance > 0 && lastRecoveryInstance < 0
                && recentFilled == RECOVERY_WINDOW
                && windowAccuracy() >= accAtDrift - RECOVERY_TOLERANCE) {
            lastRecoveryInstance = n;
            recoveredEpisodes++;
            recoveryTimeSum += (lastRecoveryInstance - lastDriftInstance);
        }
        prevDriftFlag = driftNow;

        if (n % sampleEvery == 0) writeRow();
    }

    void flushFinalRow() {
        if (n % sampleEvery != 0) writeRow();
        try { out.flush(); } catch (IOException e) { throw new UncheckedIOException(e); }
    }

    private void writeRow() {
        writeHeaderIfNeeded();

        long now = System.nanoTime();
        Runtime r = Runtime.getRuntime();
        long mem = r.totalMemory() - r.freeMemory();
        recordMemory(mem);
        if (lastSampleNanos > 0) {
            double hours = (now - lastSampleNanos) / 3_600_000_000_000.0;
            ramByteHours += (double) mem * hours;
        }
        lastSampleNanos = now;

        int[] sel = selector.getCurrentSelection();
        int[] selSorted = (sel == null) ? null : sel.clone();
        if (selSorted != null) Arrays.sort(selSorted);
        if (prevSelection != null) {
            totalSelectionSamples++;
            if (Arrays.equals(selSorted, prevSelection)) stableSamples++;
        }
        boolean selectionChanged = pipeline != null && pipeline.wasLastSelectionChange();
        prevSelection = selSorted;
        double stability = (totalSelectionSamples == 0)
                ? 1.0 : (double) stableSamples / totalSelectionSamples;

        double avgRecovery = (recoveredEpisodes == 0)
                ? Double.NaN : (double) recoveryTimeSum / recoveredEpisodes;

        boolean alarm = detector != null && detector.isGlobalDriftDetected();
        boolean warning = detectWarning();
        String driftFeats = formatSet(detector == null ? null : detector.getDriftingFeatureIndices());
        String scoresSample = formatScoresSample();
        StreamPipeline.TriggerType trig = pipeline == null
                ? StreamPipeline.TriggerType.NONE : pipeline.getLastTrigger();
        long memMb = mem / (1024 * 1024);
        double tickUs = lastTickNanos < 0 ? 0.0 : lastTickNanos / 1000.0;

        try {
            out.write(String.format(Locale.ROOT,
                    "%d,%s,%s,%s,%s,%d,%d,%d,%s,%d,%d,%s,%d,%d,%s,%s,%.6f,%.6f,%.6f,%.6f,%d,%s,%.6f,%d,%.2f",
                    n, dataset, variant, modelName, selector.name(),
                    yTrueLast, yPredLast, yTrueLast == yPredLast ? 0 : 1,
                    formatSelection(selSorted),
                    selSorted == null ? 0 : selSorted.length,
                    selectionChanged ? 1 : 0,
                    trig.name(),
                    alarm ? 1 : 0, warning ? 1 : 0,
                    driftFeats, scoresSample,
                    cohenKappaWindow(), kappaTemporal(),
                    windowAccuracy(), accuracy(),
                    driftCount,
                    Double.isNaN(avgRecovery) ? "" : String.format(Locale.ROOT, "%.2f", avgRecovery),
                    stability, memMb, tickUs));
            out.newLine();
        } catch (IOException e) { throw new UncheckedIOException(e); }
    }

    private boolean detectWarning() {
        if (detector == null) return false;
        try {
            java.lang.reflect.Method m = detector.getClass().getMethod("isWarning");
            Object v = m.invoke(detector);
            return v instanceof Boolean && (Boolean) v;
        } catch (Exception ignored) { return false; }
    }

    private static String formatSelection(int[] sel) {
        if (sel == null || sel.length == 0) return "[]";
        StringBuilder sb = new StringBuilder().append('"').append('[');
        for (int i = 0; i < sel.length; i++) {
            if (i > 0) sb.append(' ');
            sb.append(sel[i]);
        }
        return sb.append(']').append('"').toString();
    }

    private static String formatSet(java.util.Set<Integer> s) {
        if (s == null || s.isEmpty()) return "[]";
        StringBuilder sb = new StringBuilder().append('"').append('[');
        boolean first = true;
        for (Integer v : s) {
            if (!first) sb.append(' ');
            sb.append(v);
            first = false;
        }
        return sb.append(']').append('"').toString();
    }

    private String formatScoresSample() {
        if (pipeline == null) return "[]";
        double[] sc = pipeline.getLastFeatureScores();
        if (sc == null || sc.length == 0) return "[]";
        int k = Math.min(sc.length, 5);
        StringBuilder sb = new StringBuilder().append('"').append('[');
        for (int i = 0; i < k; i++) {
            if (i > 0) sb.append(' ');
            sb.append(String.format(Locale.ROOT, "%.4f", sc[i]));
        }
        return sb.append(']').append('"').toString();
    }

    private double accuracy() { return n == 0 ? 0.0 : (double) correct / n; }

    private double windowAccuracy() {
        return recentFilled == 0 ? 0.0 : (double) recentCorrectSum / recentFilled;
    }

    private double cohenKappaWindow() {
        if (recentFilled == 0) return 0.0;
        double po = windowAccuracy();
        if (po >= 1.0) return 1.0;
        return po;
    }

    private double kappaTemporal() {
        if (noChangeDenom == 0) return 0.0;
        double pNo = (double) correctNoChange / noChangeDenom;
        if (pNo >= 1.0) return 0.0;
        return (accuracy() - pNo) / (1.0 - pNo);
    }
}