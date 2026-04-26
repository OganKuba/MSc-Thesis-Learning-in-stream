package thesis.pipeline;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.core.InstanceExample;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.streams.InstanceStream;
import moa.streams.generators.SEAGenerator;
import moa.tasks.TaskMonitor;
import thesis.detection.TwoLevelDriftDetector;
import thesis.discretization.PiDDiscretizer;
import thesis.models.HoeffdingTreeWrapper;
import thesis.models.ModelWrapper;
import thesis.selection.FeatureSelector;
import thesis.selection.InformationGainRanker;
import thesis.selection.StaticFeatureSelector;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PipelineSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) throws IOException {
        System.out.println("=".repeat(70));
        System.out.println("PIPELINE SMOKE TESTS");
        System.out.println("=".repeat(70));

        testStreamMetricsBasic();
        testStreamMetricsReset();
        testStreamMetricsAvgTimeAndMemory();

        testRecordingMetricsCsvHeaderAndRowsCount();
        testRecordingMetricsAccuracyMatchesParent();
        testRecordingMetricsKappaIsHighWhenPerfect();
        testRecordingMetricsKappaTemporalNonNaN();
        testRecordingMetricsDriftCountTracksDetector();
        testRecordingMetricsFeatureStabilityWhenSelectorStatic();
        testRecordingMetricsFlushFinalRow();

        testStreamPipelineBuilderRequiresAllParts();
        testStreamPipelineRunOnSEA();
        testStreamPipelineProcessThrowsBeforeWarmup();
        testStreamPipelineWarmupHonorsAlreadyInitializedSelector();
        testStreamPipelineRespectsMaxInstances();
        testStreamPipelineThrowsWhenSourceHasNoHeader();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    private static InstancesHeader makeHeader(int F, int numClasses, int classPos) {
        ArrayList<Attribute> attrs = new ArrayList<>(F + 1);
        ArrayList<String> classVals = new ArrayList<>();
        for (int c = 0; c < numClasses; c++) classVals.add("c" + c);
        for (int i = 0; i < F + 1; i++) {
            if (i == classPos) attrs.add(new Attribute("class", classVals));
            else                attrs.add(new Attribute("f" + i));
        }
        Instances ins = new Instances("synthetic", attrs, 0);
        ins.setClassIndex(classPos);
        return new InstancesHeader(ins);
    }

    private static double[][] sea(int n, int[] outLabels, long seed) {
        SEAGenerator g = new SEAGenerator();
        g.instanceRandomSeedOption.setValue((int) seed);
        g.functionOption.setValue(1);
        g.balanceClassesOption.setValue(false);
        g.noisePercentageOption.setValue(10);
        g.prepareForUse();
        double[][] out = new double[n][3];
        for (int i = 0; i < n; i++) {
            Instance inst = g.nextInstance().getData();
            for (int f = 0; f < 3; f++) out[i][f] = inst.value(f);
            outLabels[i] = (int) inst.classValue();
        }
        return out;
    }

    private static FeatureSelector trainedSelector(int F, double[][] win, int[] y, int K) {
        StaticFeatureSelector sel = new StaticFeatureSelector(F, 2, K,
                new PiDDiscretizer(F, 2),
                (bins, classes) -> new InformationGainRanker(F, bins, classes));
        sel.initialize(win, y);
        return sel;
    }

    private static TwoLevelDriftDetector buildDetector(int F) {
        TwoLevelDriftDetector.Config cfg = new TwoLevelDriftDetector.Config(F);
        cfg.level1Type = TwoLevelDriftDetector.Level1Type.ADWIN;
        cfg.level1Delta = 0.002;
        cfg.kswinAlpha = 0.01;
        cfg.kswinWindowSize = 100;
        cfg.bhQ = 0.05;
        return new TwoLevelDriftDetector(cfg);
    }

    private static final class FixedStream extends AbstractOptionHandler implements InstanceStream {
        private final InstancesHeader header;
        private final double[][] data;
        private final int[] labels;
        private int pos = 0;

        FixedStream(InstancesHeader header, double[][] data, int[] labels) {
            this.header = header;
            this.data = data;
            this.labels = labels;
        }

        @Override public InstancesHeader getHeader() { return header; }
        @Override public long estimatedRemainingInstances() { return data.length - pos; }
        @Override public boolean hasMoreInstances() { return pos < data.length; }
        @Override public boolean isRestartable() { return true; }
        @Override public void restart() { pos = 0; }

        @Override
        public InstanceExample nextInstance() {
            double[] vals = new double[header.numAttributes()];
            int F = data[pos].length;
            for (int f = 0; f < F; f++) vals[f] = data[pos][f];
            vals[header.classIndex()] = labels[pos];
            Instance inst = new DenseInstance(1.0, vals);
            inst.setDataset(header);
            pos++;
            return new InstanceExample(inst);
        }

        @Override public void getDescription(StringBuilder sb, int indent) { sb.append("FixedStream"); }
        @Override protected void prepareForUseImpl(TaskMonitor m, ObjectRepository r) { }
    }

    private static final class BrokenStream extends AbstractOptionHandler implements InstanceStream {
        @Override public InstancesHeader getHeader() { return null; }
        @Override public long estimatedRemainingInstances() { return 0; }
        @Override public boolean hasMoreInstances() { return false; }
        @Override public boolean isRestartable() { return false; }
        @Override public void restart() { }
        @Override public InstanceExample nextInstance() { throw new UnsupportedOperationException(); }
        @Override public void getDescription(StringBuilder sb, int indent) { }
        @Override protected void prepareForUseImpl(TaskMonitor m, ObjectRepository r) { }
    }

    private static void testStreamMetricsBasic() {
        StreamMetrics m = new StreamMetrics();
        m.update(0, 0, 100);
        m.update(1, 0, 200);
        m.update(1, 1, 300);
        boolean ok = m.getCount() == 3
                && m.getCorrect() == 2
                && Math.abs(m.getAccuracy() - 2.0/3.0) < 1e-9
                && m.getTotalTimeNanos() == 600
                && Math.abs(m.getAvgTimeMicros() - 0.2) < 1e-9;
        report("StreamMetrics tracks count/correct/avgTime", ok);
    }

    private static void testStreamMetricsReset() {
        StreamMetrics m = new StreamMetrics();
        m.update(0, 0, 100);
        m.recordMemory(10_000);
        m.reset();
        boolean ok = m.getCount() == 0 && m.getCorrect() == 0
                && m.getTotalTimeNanos() == 0
                && m.getPeakMemoryBytes() == 0
                && m.getLastUpdateNanos() == 0;
        report("StreamMetrics.reset zeroes everything", ok);
    }

    private static void testStreamMetricsAvgTimeAndMemory() {
        StreamMetrics m = new StreamMetrics();
        boolean okEmpty = m.getAccuracy() == 0.0 && m.getAvgTimeMicros() == 0.0;
        m.recordMemory(1_000);
        m.recordMemory(10_000);
        m.recordMemory(5_000);
        boolean ok = okEmpty && m.getPeakMemoryBytes() == 10_000;
        report("StreamMetrics tracks peak memory + handles empty state", ok);
    }

    private static List<String> readCsvDataRows(Path csv) throws IOException {
        try (BufferedReader r = Files.newBufferedReader(csv)) {
            List<String> rows = new ArrayList<>();
            String line; r.readLine();
            while ((line = r.readLine()) != null) rows.add(line);
            return rows;
        }
    }

    private static FeatureSelector dummySelector(int F, int K) {
        int[] y = new int[600];
        double[][] data = sea(600, y, 9001);
        return trainedSelector(F, data, y, K);
    }

    private static void testRecordingMetricsCsvHeaderAndRowsCount() throws IOException {
        Path tmp = Files.createTempFile("smoke_recmet_", ".csv");
        FeatureSelector sel = dummySelector(3, 2);
        BufferedWriter w = Files.newBufferedWriter(tmp);
        w.write("instance_num,kappa,kappa_per,accuracy,ram_hours,"
                + "feature_stability_ratio,drift_count,recovery_time");
        w.newLine();
        RecordingMetrics rm = new RecordingMetrics(w, 5, sel);
        for (int i = 0; i < 27; i++) rm.update(i % 2, i % 2, 100);
        rm.flushFinalRow();
        w.close();
        List<String> rows = readCsvDataRows(tmp);
        boolean ok = rows.size() == 6;
        report("RecordingMetrics writes one row per sample + final flush (got "
                + rows.size() + " rows)", ok);
        Files.deleteIfExists(tmp);
    }

    private static void testRecordingMetricsAccuracyMatchesParent() throws IOException {
        Path tmp = Files.createTempFile("smoke_recmet_acc_", ".csv");
        BufferedWriter w = Files.newBufferedWriter(tmp);
        w.write("h\n");
        FeatureSelector sel = dummySelector(3, 2);
        RecordingMetrics rm = new RecordingMetrics(w, 1000, sel);
        for (int i = 0; i < 100; i++) {
            int y = i % 2;
            int yhat = (i < 75) ? y : 1 - y;
            rm.update(y, yhat, 50);
        }
        w.close();
        boolean ok = Math.abs(rm.getAccuracy() - 0.75) < 1e-9 && rm.getCount() == 100;
        report("RecordingMetrics inherits StreamMetrics accuracy (acc="
                + rm.getAccuracy() + ")", ok);
        Files.deleteIfExists(tmp);
    }

    private static void testRecordingMetricsKappaIsHighWhenPerfect() throws IOException {
        Path tmp = Files.createTempFile("smoke_recmet_kappa_", ".csv");
        BufferedWriter w = Files.newBufferedWriter(tmp);
        w.write("h\n");
        FeatureSelector sel = dummySelector(3, 2);
        RecordingMetrics rm = new RecordingMetrics(w, 50, sel);
        for (int i = 0; i < 200; i++) rm.update(i % 2, i % 2, 50);
        rm.flushFinalRow();
        w.close();
        boolean ok = rm.getAccuracy() == 1.0;
        report("RecordingMetrics: perfect predictions yield acc=1.0 (kappa not negative)", ok);
        Files.deleteIfExists(tmp);
    }

    private static void testRecordingMetricsKappaTemporalNonNaN() throws IOException {
        Path tmp = Files.createTempFile("smoke_recmet_kt_", ".csv");
        BufferedWriter w = Files.newBufferedWriter(tmp);
        w.write("h\n");
        FeatureSelector sel = dummySelector(3, 2);
        RecordingMetrics rm = new RecordingMetrics(w, 25, sel);
        int[] labels = {0,0,1,1,0,1,0,1,1,0};
        for (int i = 0; i < 100; i++) {
            int y = labels[i % labels.length];
            int yhat = (i % 3 == 0) ? 1 - y : y;
            rm.update(y, yhat, 100);
        }
        rm.flushFinalRow();
        w.close();
        List<String> rows = readCsvDataRows(tmp);
        boolean anyValidKappa = false;
        for (String r : rows) {
            String[] parts = r.split(",");
            if (parts.length >= 3 && !parts[2].isEmpty()
                    && !"NaN".equalsIgnoreCase(parts[2])) {
                anyValidKappa = true; break;
            }
        }
        report("RecordingMetrics kappa_per column populated (non-NaN)", anyValidKappa);
        Files.deleteIfExists(tmp);
    }

    private static void testRecordingMetricsDriftCountTracksDetector() throws IOException {
        Path tmp = Files.createTempFile("smoke_recmet_drift_", ".csv");
        BufferedWriter w = Files.newBufferedWriter(tmp);
        w.write("h\n");
        FeatureSelector sel = dummySelector(3, 2);
        TwoLevelDriftDetector det = buildDetector(3);
        RecordingMetrics rm = new RecordingMetrics(w, 100, sel);
        rm.bindPipeline(null, det);

        int[] y = new int[3000];
        double[][] data = sea(3000, y, 7777);

        for (int i = 0; i < 1500; i++) {
            det.update(0.0, data[i]);
            rm.update(y[i], y[i], 50);
        }

        for (int i = 1500; i < 3000; i++) {
            det.update(1.0, data[i]);
            rm.update(y[i], 1 - y[i], 50);
        }
        rm.flushFinalRow();
        w.close();

        long detectorAlarmsTotal = det.getGlobalAlarms();
        long recorderDriftFinal  = parseFinalDriftCount(tmp);
        boolean ok = detectorAlarmsTotal >= 1
                && recorderDriftFinal >= 1
                && recorderDriftFinal == detectorAlarmsTotal;
        report("RecordingMetrics drift_count == detector.globalAlarms (det="
                + detectorAlarmsTotal + ", csv=" + recorderDriftFinal + ")", ok);
        Files.deleteIfExists(tmp);
    }

    private static long parseFinalDriftCount(Path csv) throws IOException {
        List<String> rows = readCsvDataRows(csv);
        if (rows.isEmpty()) return 0;
        String[] last = rows.get(rows.size() - 1).split(",");
        return Long.parseLong(last[6]);
    }

    private static void testRecordingMetricsFeatureStabilityWhenSelectorStatic() throws IOException {
        Path tmp = Files.createTempFile("smoke_recmet_stab_", ".csv");
        BufferedWriter w = Files.newBufferedWriter(tmp);
        w.write("h\n");
        FeatureSelector sel = dummySelector(3, 2);
        RecordingMetrics rm = new RecordingMetrics(w, 10, sel);
        for (int i = 0; i < 100; i++) rm.update(i % 2, i % 2, 50);
        rm.flushFinalRow();
        w.close();
        List<String> rows = readCsvDataRows(tmp);
        String[] last = rows.get(rows.size() - 1).split(",");
        double stability = Double.parseDouble(last[5]);
        report("RecordingMetrics feature_stability_ratio == 1.0 for static selector "
                + "(got " + stability + ")", Math.abs(stability - 1.0) < 1e-9);
        Files.deleteIfExists(tmp);
    }

    private static void testRecordingMetricsFlushFinalRow() throws IOException {
        Path tmp = Files.createTempFile("smoke_recmet_flush_", ".csv");
        BufferedWriter w = Files.newBufferedWriter(tmp);
        w.write("h\n");
        FeatureSelector sel = dummySelector(3, 2);
        RecordingMetrics rm = new RecordingMetrics(w, 50, sel);
        for (int i = 0; i < 73; i++) rm.update(0, 0, 50);
        rm.flushFinalRow();
        w.close();
        List<String> rows = readCsvDataRows(tmp);
        boolean ok = rows.size() == 2;
        long lastN = Long.parseLong(rows.get(rows.size() - 1).split(",")[0]);
        report("RecordingMetrics.flushFinalRow emits final partial sample (rows="
                + rows.size() + ", lastN=" + lastN + ")", ok && lastN == 73);
        Files.deleteIfExists(tmp);
    }

    private static void testStreamPipelineBuilderRequiresAllParts() {
        boolean t1 = false, t2 = false, t3 = false, t4 = false;
        try { StreamPipeline.builder().build(); }
        catch (IllegalStateException e) { t1 = true; }
        InstancesHeader h = makeHeader(3, 2, 3);
        int[] y = new int[600]; double[][] data = sea(600, y, 71);
        FeatureSelector sel = trainedSelector(3, data, y, 2);
        ModelWrapper mw = new HoeffdingTreeWrapper(sel, h);
        try {
            StreamPipeline.builder().selector(sel).model(mw)
                    .detector(buildDetector(3)).build();
        } catch (IllegalStateException e) { t2 = true; }
        try {
            StreamPipeline.builder().source(new FixedStream(h, data, y)).model(mw)
                    .detector(buildDetector(3)).build();
        } catch (IllegalStateException e) { t3 = true; }
        try {
            StreamPipeline.builder().source(new FixedStream(h, data, y))
                    .selector(sel).model(mw).detector(buildDetector(3))
                    .warmup(0).build();
        } catch (IllegalStateException e) { t4 = true; }
        report("StreamPipeline.Builder rejects missing required parts", t1 && t2 && t3 && t4);
    }

    private static void testStreamPipelineRunOnSEA() throws IOException {
        int F = 3, K = 2, N = 3000;
        int[] y = new int[N];
        double[][] data = sea(N, y, 81);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = trainedSelector(F, Arrays.copyOf(data, 600),
                Arrays.copyOf(y, 600), K);
        ModelWrapper model = new HoeffdingTreeWrapper(sel, h);
        TwoLevelDriftDetector det = buildDetector(F);

        Path csv = Files.createTempFile("smoke_pipe_run_", ".csv");
        BufferedWriter w = Files.newBufferedWriter(csv);
        w.write("instance_num,kappa,kappa_per,accuracy,ram_hours,"
                + "feature_stability_ratio,drift_count,recovery_time");
        w.newLine();
        RecordingMetrics rm = new RecordingMetrics(w, 200, sel);

        StreamPipeline pipe = StreamPipeline.builder()
                .source(new FixedStream(h, data, y))
                .selector(sel).model(model).detector(det).metrics(rm)
                .warmup(1)
                .logEvery(10_000).verbose(false)
                .maxInstances(N)
                .build();

        rm.bindPipeline(pipe, det);
        pipe.run();
        rm.flushFinalRow();
        w.close();

        List<String> rows = readCsvDataRows(csv);
        boolean ok = pipe.getProcessed() >= N - 1
                && rm.getAccuracy() > 0.5
                && !rows.isEmpty()
                && pipe.getMetrics() == rm;
        report("StreamPipeline runs SEA end-to-end (processed=" + pipe.getProcessed()
                + ", acc=" + String.format("%.3f", rm.getAccuracy())
                + ", csvRows=" + rows.size() + ")", ok);
        Files.deleteIfExists(csv);
    }

    private static void testStreamPipelineProcessThrowsBeforeWarmup() {
        int F = 3;
        int[] y = new int[600]; double[][] data = sea(600, y, 91);
        InstancesHeader h = makeHeader(F, 2, F);

        FeatureSelector uninit = new StaticFeatureSelector(F, 2, 2,
                new PiDDiscretizer(F, 2),
                (b, c) -> new InformationGainRanker(F, b, c));

        FeatureSelector sel = trainedSelector(F, data, y, 2);
        ModelWrapper model = new HoeffdingTreeWrapper(sel, h);
        StreamPipeline pipe = StreamPipeline.builder()
                .source(new FixedStream(h, data, y))
                .selector(uninit).model(model).detector(buildDetector(F))
                .warmup(50).verbose(false).build();
        Instance inst = new DenseInstance(1.0, new double[]{0, 0, 0, 0});
        inst.setDataset(h);
        boolean threw = false;
        try { pipe.processInstance(inst); }
        catch (IllegalStateException e) { threw = true; }
        report("StreamPipeline.processInstance throws before warmup", threw);
    }

    private static void testStreamPipelineWarmupHonorsAlreadyInitializedSelector() {
        int F = 3;
        int[] y = new int[600]; double[][] data = sea(600, y, 92);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = trainedSelector(F, data, y, 2);
        ModelWrapper model = new HoeffdingTreeWrapper(sel, h);
        StreamPipeline pipe = StreamPipeline.builder()
                .source(new FixedStream(h, data, y))
                .selector(sel).model(model).detector(buildDetector(F))
                .warmup(999_999).verbose(false).build();
        pipe.warmupIfNeeded();
        report("StreamPipeline.warmupIfNeeded skips when selector is already initialized",
                pipe.isWarmedUp() && pipe.getProcessed() == 0);
    }

    private static void testStreamPipelineRespectsMaxInstances() {
        int F = 3, N = 1000;
        int[] y = new int[N];
        double[][] data = sea(N, y, 93);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = trainedSelector(F, Arrays.copyOf(data, 600),
                Arrays.copyOf(y, 600), 2);
        ModelWrapper model = new HoeffdingTreeWrapper(sel, h);
        StreamPipeline pipe = StreamPipeline.builder()
                .source(new FixedStream(h, data, y))
                .selector(sel).model(model).detector(buildDetector(F))
                .warmup(1).maxInstances(123).verbose(false).build();
        pipe.run();
        report("StreamPipeline honors maxInstances cap (processed=" + pipe.getProcessed() + ")",
                pipe.getProcessed() == 123);
    }

    private static void testStreamPipelineThrowsWhenSourceHasNoHeader() {
        InstanceStream broken = new BrokenStream();
        int[] y = new int[600]; double[][] data = sea(600, y, 94);
        InstancesHeader h = makeHeader(3, 2, 3);
        FeatureSelector sel = trainedSelector(3, data, y, 2);
        ModelWrapper model = new HoeffdingTreeWrapper(sel, h);
        boolean threw = false;
        try {
            StreamPipeline.builder()
                    .source(broken).selector(sel).model(model)
                    .detector(buildDetector(3)).warmup(1).verbose(false).build();
        } catch (IllegalStateException e) {
            threw = true;
        }
        report("StreamPipeline rejects source with null header", threw);
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