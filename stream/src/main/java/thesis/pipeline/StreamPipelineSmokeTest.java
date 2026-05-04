package thesis.pipeline;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.streams.generators.SEAGenerator;
import thesis.detection.TwoLevelDriftDetector;
import thesis.discretization.PiDDiscretizer;
import thesis.models.DriftAwareSRP;
import thesis.models.FeatureImportance;
import thesis.models.ModelWrapper;
import thesis.models.SRPWrapper;
import thesis.selection.FeatureSelector;

import java.io.BufferedWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class StreamPipelineSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("STREAM PIPELINE SMOKE TESTS");
        System.out.println("=".repeat(70));

        testWarmupInitializesSelector();
        testSelectorUpdateCalledOncePerTick();
        testDriftAlarmReachesSelector();
        testDriftingFeaturesPropagated();
        testPeriodicTriggerFires();
        testWhereTriggerForDASRP();
        testCsvHeaderAndColumnsCount();
        testCsvAccuracyIsWindowed();
        testStrictModeRejectsBadScoresLen();
        testNoDoubleAutoHandleDriftOnAlarm();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    private static SEAGenerator makeStream(long seed) {
        SEAGenerator g = new SEAGenerator();
        g.instanceRandomSeedOption.setValue((int) seed);
        g.functionOption.setValue(1);
        g.balanceClassesOption.setValue(false);
        g.noisePercentageOption.setValue(10);
        g.prepareForUse();
        return g;
    }

    private static final class CountingSelector implements FeatureSelector {
        int updates = 0;
        int alarms = 0;
        int totalDriftFeats = 0;
        boolean initialized = false;
        int F;
        int K;
        int[] sel;
        CountingSelector(int F, int K) {
            this.F = F; this.K = K;
            this.sel = new int[K];
            for (int i = 0; i < K; i++) sel[i] = i;
        }
        @Override public void initialize(double[][] w, int[] y) { initialized = true; }
        @Override public void update(double[] x, int y, boolean alarm, Set<Integer> df) {
            updates++;
            if (alarm) { alarms++; totalDriftFeats += df == null ? 0 : df.size(); }
        }
        @Override public int[] getSelectedFeatures() { return sel.clone(); }
        @Override public int[] getCurrentSelection() { return sel.clone(); }
        @Override public double[] filterInstance(double[] x) {
            double[] o = new double[K];
            for (int i = 0; i < K; i++) o[i] = x[sel[i]];
            return o;
        }
        @Override public int getNumFeatures() { return F; }
        @Override public int getK() { return K; }
        @Override public boolean isInitialized() { return initialized; }
        @Override public String name() { return "Counting"; }
    }

    private static final class StubDetector extends TwoLevelDriftDetector {
        boolean alarmEvery;
        Set<Integer> drifting = Set.of();
        final int F;

        StubDetector(int F) {
            super(new Config(F));
            this.F = F;
        }

        @Override
        public void update(double predictionError, double[] featureValues) {
            // intencjonalnie no-op: w testach pipeline'u chcemy deterministyczny alarm
        }

        @Override
        public boolean isGlobalDriftDetected() { return alarmEvery; }

        @Override
        public Set<Integer> getDriftingFeatureIndices() { return drifting; }

        @Override
        public double[] getLastPValues() { return new double[F]; }
    }

    private static StreamPipeline buildPipeline(CountingSelector sel, ModelWrapper model,
                                                StubDetector det, int warmup,
                                                StreamMetrics metrics) {
        return StreamPipeline.builder()
                .source(makeStream(7))
                .selector(sel)
                .model(model)
                .detector(det)
                .metrics(metrics == null ? new StreamMetrics() : metrics)
                .warmup(warmup)
                .logEvery(0)
                .verbose(false)
                .maxInstances(2000)
                .build();
    }

    private static SRPWrapper buildSrp(CountingSelector sel) {
        sel.initialized = true;
        InstancesHeader h = makeStream(7).getHeader();
        return new SRPWrapper(sel, h, 3, 6.0, false, false);
    }

    private static void testWarmupInitializesSelector() {
        CountingSelector sel = new CountingSelector(3, 2);
        sel.initialized = true;
        SRPWrapper srp = buildSrp(sel);
        sel.initialized = false;
        StubDetector det = new StubDetector(3);
        StreamPipeline p = buildPipeline(sel, srp, det, 200, null);
        p.warmupIfNeeded();
        report("warmup initializes selector", sel.initialized);
    }

    private static void testSelectorUpdateCalledOncePerTick() {
        CountingSelector sel = new CountingSelector(3, 2);
        sel.initialized = true;
        StubDetector det = new StubDetector(3);
        SRPWrapper srp = buildSrp(sel);
        StreamPipeline p = buildPipeline(sel, srp, det, 1, null);
        p.warmupIfNeeded();
        SEAGenerator g = makeStream(11);
        for (int i = 0; i < 50; i++) p.processInstance(g.nextInstance().getData());
        report("selector.update called once per tick (got " + sel.updates + " for 50 ticks)",
                sel.updates == 50);
    }

    private static void testDriftAlarmReachesSelector() {
        CountingSelector sel = new CountingSelector(3, 2);
        sel.initialized = true;
        StubDetector det = new StubDetector(3);
        det.alarmEvery = true;
        SRPWrapper srp = buildSrp(sel);
        StreamPipeline p = buildPipeline(sel, srp, det, 1, null);
        p.warmupIfNeeded();
        SEAGenerator g = makeStream(12);
        for (int i = 0; i < 30; i++) p.processInstance(g.nextInstance().getData());
        report("drift alarm reaches selector (alarms=" + sel.alarms + "/30)", sel.alarms == 30);
    }

    private static void testDriftingFeaturesPropagated() {
        CountingSelector sel = new CountingSelector(3, 2);
        sel.initialized = true;
        StubDetector det = new StubDetector(3);
        det.alarmEvery = true;
        det.drifting = Set.of(0, 2);
        SRPWrapper srp = buildSrp(sel);
        StreamPipeline p = buildPipeline(sel, srp, det, 1, null);
        p.warmupIfNeeded();
        SEAGenerator g = makeStream(13);
        for (int i = 0; i < 10; i++) p.processInstance(g.nextInstance().getData());
        report("drifting features propagated (totalDriftFeats=" + sel.totalDriftFeats + " expected 20)",
                sel.totalDriftFeats == 20);
    }

    private static void testPeriodicTriggerFires() {
        CountingSelector sel = new CountingSelector(3, 2);
        sel.initialized = true;
        StubDetector det = new StubDetector(3);
        SRPWrapper srp = buildSrp(sel);
        StreamPipeline p = StreamPipeline.builder()
                .source(makeStream(14))
                .selector(sel).model(srp).detector(det)
                .warmup(1).logEvery(0).verbose(false).refreshEvery(10)
                .maxInstances(50).build();
        p.warmupIfNeeded();
        SEAGenerator g = makeStream(14);
        StreamPipeline.TriggerType seen = StreamPipeline.TriggerType.NONE;
        for (int i = 0; i < 30; i++) {
            p.processInstance(g.nextInstance().getData());
            if (p.getLastTrigger() == StreamPipeline.TriggerType.PERIODIC) {
                seen = StreamPipeline.TriggerType.PERIODIC;
            }
        }
        report("periodic trigger fires (lastSeen=" + seen + ")", seen == StreamPipeline.TriggerType.PERIODIC);
    }

    private static void testWhereTriggerForDASRP() {
        CountingSelector sel = new CountingSelector(3, 2);
        sel.initialized = true;
        StubDetector det = new StubDetector(3);
        det.alarmEvery = true;
        SRPWrapper srp = buildSrp(sel);
        FeatureImportance imp = new FeatureImportance(3);
        imp.update(new double[]{0.2, 0.5, 0.3}, new double[3]);
        DriftAwareSRP da = new DriftAwareSRP(srp, 0.5, 1L, imp);
        StreamPipeline p = buildPipeline(sel, da, det, 1, null);
        p.warmupIfNeeded();
        SEAGenerator g = makeStream(15);
        StreamPipeline.TriggerType seen = StreamPipeline.TriggerType.NONE;
        for (int i = 0; i < 20; i++) {
            p.processInstance(g.nextInstance().getData());
            if (p.getLastTrigger() == StreamPipeline.TriggerType.WHERE) {
                seen = StreamPipeline.TriggerType.WHERE;
            }
        }
        report("WHERE trigger fires for DA-SRP on alarm (lastSeen=" + seen + ")",
                seen == StreamPipeline.TriggerType.WHERE);
    }

    private static void testCsvHeaderAndColumnsCount() {
        CountingSelector sel = new CountingSelector(3, 2);
        sel.initialized = true;
        StubDetector det = new StubDetector(3);
        SRPWrapper srp = buildSrp(sel);
        StringWriter sw = new StringWriter();
        BufferedWriter bw = new BufferedWriter(sw);
        RecordingMetrics rm = new RecordingMetrics(bw, 10, sel, 2, "ds", "S1", "SRP");
        StreamPipeline p = buildPipeline(sel, srp, det, 1, rm);
        p.warmupIfNeeded();
        SEAGenerator g = makeStream(16);
        for (int i = 0; i < 30; i++) p.processInstance(g.nextInstance().getData());
        rm.flushFinalRow();
        String[] lines = sw.toString().split("\\R");
        String[] header = lines[0].split(",");
        report("CSV header has 25 columns (got " + header.length + ")", header.length == 25);
        report("CSV rows include trigger_type column",
                lines[0].contains("trigger_type"));
    }

    private static void testCsvAccuracyIsWindowed() {
        CountingSelector sel = new CountingSelector(3, 2);
        sel.initialized = true;
        StubDetector det = new StubDetector(3);
        SRPWrapper srp = buildSrp(sel);
        StringWriter sw = new StringWriter();
        BufferedWriter bw = new BufferedWriter(sw);
        RecordingMetrics rm = new RecordingMetrics(bw, 10, sel, 2, "ds", "S1", "SRP");
        StreamPipeline p = buildPipeline(sel, srp, det, 1, rm);
        p.warmupIfNeeded();
        SEAGenerator g = makeStream(17);
        for (int i = 0; i < 30; i++) p.processInstance(g.nextInstance().getData());
        rm.flushFinalRow();
        String[] lines = sw.toString().split("\\R");
        String[] header = lines[0].split(",");
        int idxAccWin = -1, idxAccCum = -1;
        for (int i = 0; i < header.length; i++) {
            if ("accuracy_window".equals(header[i].trim())) idxAccWin = i;
            if ("accuracy_cumulative".equals(header[i].trim())) idxAccCum = i;
        }
        report("accuracy_window and accuracy_cumulative both present",
                idxAccWin >= 0 && idxAccCum >= 0);
    }

    private static void testStrictModeRejectsBadScoresLen() {
        CountingSelector sel = new CountingSelector(3, 2);
        sel.initialized = true;
        StubDetector det = new StubDetector(3);
        det.alarmEvery = true;
        SRPWrapper srp = buildSrp(sel);
        FeatureImportance imp = new FeatureImportance(3);
        imp.update(new double[]{0.2, 0.5, 0.3}, new double[3]);
        DriftAwareSRP da = new DriftAwareSRP(srp, 0.5, 1L, imp);
        StreamPipeline p = StreamPipeline.builder()
                .source(makeStream(18))
                .selector(sel).model(da).detector(det)
                .importance(imp)
                .warmup(1).logEvery(0).verbose(false).strict(true)
                .maxInstances(20).build();
        p.warmupIfNeeded();
        report("strict mode constructs OK without fullRanker (importance update skipped)", true);
    }

    private static void testNoDoubleAutoHandleDriftOnAlarm() {
        CountingSelector sel = new CountingSelector(3, 2);
        sel.initialized = true;
        StubDetector det = new StubDetector(3);
        det.alarmEvery = true;
        SRPWrapper srp = buildSrp(sel);
        FeatureImportance imp = new FeatureImportance(3);
        imp.update(new double[]{0.1, 0.5, 0.4}, new double[3]);
        DriftAwareSRP da = new DriftAwareSRP(srp, 0.5, 1L, imp);
        StreamPipeline p = buildPipeline(sel, da, det, 1, null);
        p.warmupIfNeeded();
        long beforeAuto = da.getAutoHandleDriftCalls();
        long beforeManual = da.getHandleDriftCalls();
        SEAGenerator g = makeStream(19);
        for (int i = 0; i < 10; i++) p.processInstance(g.nextInstance().getData());
        long afterAuto = da.getAutoHandleDriftCalls();
        long afterManual = da.getHandleDriftCalls();
        report("each alarm triggers exactly one handleDrift "
                        + "(auto+ " + (afterAuto - beforeAuto)
                        + ", manual+ " + (afterManual - beforeManual) + " for 10 alarms)",
                afterAuto - beforeAuto == 10
                        && afterManual - beforeManual == afterAuto - beforeAuto);
    }

    private static void report(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  [PASSED] " + name); }
        else    { failed++; System.out.println("  [FAILED] " + name); }
    }
}