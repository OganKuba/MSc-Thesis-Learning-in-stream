package thesis.experiments;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.options.OptionHandler;
import moa.streams.InstanceStream;
import thesis.detection.TwoLevelDriftDetector;
import thesis.evaluation.MetricsCollector;
import thesis.experiments.E2AdaptiveFS.Cfg;
import thesis.experiments.E2AdaptiveFS.DatasetSpec;
import thesis.experiments.E2AdaptiveFS.Variant;
import thesis.models.FeatureSpace;
import thesis.models.MajorityClassWrapper;
import thesis.models.ModelWrapper;
import thesis.models.NoChangeWrapper;
import thesis.selection.FeatureSelector;
import thesis.selection.StaticFeatureSelector;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public class E2PrototypeRunner {

    static final class Result {
        String dataset, variant, model, selector, detector;
        int seed, k, d, periodicInterval;
        long instances, driftCount, periodicTriggers, reSelections, selectionChangeCount;
        double accuracy, kappa, kappaPer, accMajority, accNoChange, ramHours, avgStability;
        int[] finalSelection;
        boolean s1IdenticalSelection;
        boolean pass;
        String reason = "";
    }

    public static void main(String[] args) throws Exception {

        Path configPath = args.length > 0
                ? Paths.get(args[0])
                : Paths.get("src/main/java/thesis/experiments/e2_prototype.json");

        if (!Files.exists(configPath)) {
            throw new RuntimeException(
                    "Config not found: " + configPath.toAbsolutePath() +
                            "\nWorking dir: " + Paths.get(".").toAbsolutePath()
            );
        }

        Cfg cfg = Cfg.load(configPath);

        new E2PrototypeRunner().run(cfg);
    }

    public void run(Cfg cfg) throws Exception {
        Files.createDirectories(Paths.get(cfg.outputDir));
        Path csvPath = Paths.get(cfg.outputDir, "E2_prototype.csv");
        List<Result> all = new ArrayList<>();

        try (PrintWriter csv = new PrintWriter(new FileWriter(csvPath.toFile()))) {
            csv.println(E2AdaptiveFS.windowHeader());
            for (DatasetSpec ds : cfg.datasets) {
                if ("arff".equalsIgnoreCase(ds.type)
                        && (ds.path == null || !Files.exists(Paths.get(ds.path)))) {
                    System.err.printf("[proto] skip arff %s%n", ds.name);
                    if (cfg.skipMissingArff) continue;
                    throw new java.io.FileNotFoundException(String.valueOf(ds.path));
                }
                for (int seed : cfg.seeds) {
                    for (Variant v : cfg.variants) {
                        try {
                            all.add(runOne(cfg, ds, v, seed, csv));
                        } catch (Exception e) {
                            System.err.printf("[proto][FAIL] %s|%s|seed=%d -> %s%n",
                                    ds.name, v.name, seed, e);
                            e.printStackTrace(System.err);
                        }
                    }
                }
            }
        }
        validateAndSummarize(all, cfg);
    }

    private Result runOne(Cfg cfg, DatasetSpec ds, Variant v, int seed,
                          PrintWriter csv) throws Exception {
        InstanceStream stream = E2AdaptiveFS.buildStream(ds, seed);
        if (stream instanceof OptionHandler) ((OptionHandler) stream).prepareForUse();
        InstancesHeader header = stream.getHeader();
        int d = header.numAttributes() - 1;
        int C = header.numClasses();
        int K = StaticFeatureSelector.defaultK(d);
        FeatureSpace space = new FeatureSpace(header);

        double[][] window = new double[cfg.warmup][];
        int[] labels = new int[cfg.warmup];
        int collected = 0;
        while (collected < cfg.warmup && stream.hasMoreInstances()) {
            Instance x = stream.nextInstance().getData();
            window[collected] = space.extractFeatures(x);
            labels[collected] = (int) x.classValue();
            collected++;
        }
        if (collected == 0) throw new IllegalStateException("no warmup instances");
        if (collected < cfg.warmup) {
            window = Arrays.copyOf(window, collected);
            labels = Arrays.copyOf(labels, collected);
        }

        FeatureSelector selector = E2AdaptiveFS.buildSelector(v, d, C, K);
        selector.initialize(window, labels);

        AtomicInteger triggerCode = new AtomicInteger(0);
        AtomicLong periodicTriggers = new AtomicLong(0);
        AtomicLong reSelections = new AtomicLong(0);
        E2AdaptiveFS.attachListeners(selector, triggerCode, periodicTriggers, reSelections);

        ModelWrapper model    = E2AdaptiveFS.buildModel(v.model, selector, header);
        ModelWrapper majority = new MajorityClassWrapper(selector, C);
        ModelWrapper noChange = new NoChangeWrapper(selector, C);
        TwoLevelDriftDetector detector = E2AdaptiveFS.buildDetector(v.detector, d, cfg.detectorDelta);

        for (int i = 0; i < collected; i++) {
            majority.train(null, labels[i]);
            noChange.train(null, labels[i]);
        }

        MetricsCollector mc = new MetricsCollector(C, cfg.windowSize, cfg.logEvery, cfg.ramSampleEvery);
        MetricsCollector mM = new MetricsCollector(C, cfg.windowSize, cfg.logEvery, cfg.ramSampleEvery);
        MetricsCollector mN = new MetricsCollector(C, cfg.windowSize, cfg.logEvery, cfg.ramSampleEvery);

        int[] lastSel = selector.getCurrentSelection().clone();
        long n = collected;
        long driftCount = 0;
        long selectionChangeCount = 0;
        boolean selChangedSinceLog = false;

        while (stream.hasMoreInstances() && n < cfg.maxInstances) {
            Instance raw = stream.nextInstance().getData();
            int y = (int) raw.classValue();
            double[] x = space.extractFeatures(raw);

            triggerCode.set(0);

            long t0 = System.nanoTime();
            int yhat = model.predict(raw);
            long elapsed = System.nanoTime() - t0;
            int yMaj = majority.predict(raw);
            int yNC  = noChange.predict(raw);

            double err = (yhat == y) ? 0.0 : 1.0;
            detector.update(err, x);
            boolean alarm = detector.isGlobalDriftDetected();
            Set<Integer> drifting = alarm ? detector.getDriftingFeatureIndices() : Set.of();

            mc.update(y, yhat, elapsed);
            mM.update(y, yMaj, 0);
            mN.update(y, yNC, 0);
            if (alarm) { mc.onDriftAlarm(); driftCount++; }

            selector.update(x, y, alarm, drifting);
            model.train(raw, y, alarm, drifting);
            majority.train(raw, y);
            noChange.train(raw, y);

            int[] sel = selector.getCurrentSelection();
            mc.onSelectionChanged(sel);
            if (!Arrays.equals(sel, lastSel)) {
                selChangedSinceLog = true;
                selectionChangeCount++;
                lastSel = sel.clone();
            }

            n++;
            if (cfg.logEvery > 0 && n % cfg.logEvery == 0) {
                MetricsCollector.Snapshot s = mc.snapshot();
                String trig = triggerCode.get() == 1 ? "ALARM"
                        : triggerCode.get() == 2 ? "PERIODIC" : "NONE";
                csv.printf(Locale.ROOT,
                        "%d,%s,%s,%s,%s,%s,%d,%d,\"%s\",%d,%s,%s,%s,%d,%.4f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%.6f%n",
                        n, ds.name, v.name, v.model, v.selector, v.detector,
                        v.periodicInterval, seed,
                        E2AdaptiveFS.joinSel(sel),
                        sel.length,
                        selChangedSinceLog ? "true" : "false",
                        trig,
                        alarm ? "true" : "false",
                        driftCount,
                        Double.isNaN(s.featureStabilityRatio) ? 1.0 : s.featureStabilityRatio,
                        model.getCurrentSelection().length + 1,
                        s.accuracyWindow, s.kappa, s.kappaPer,
                        mM.snapshot().accuracyWindow, mN.snapshot().accuracyWindow,
                        s.lastRecoveryTime, s.ramHoursGB);
                csv.flush();
                selChangedSinceLog = false;
            }
        }

        MetricsCollector.Snapshot s = mc.snapshot();
        Result r = new Result();
        r.dataset = ds.name; r.variant = v.name; r.model = v.model;
        r.selector = v.selector; r.detector = v.detector;
        r.periodicInterval = v.periodicInterval; r.seed = seed; r.k = K; r.d = d;
        r.instances = n;
        r.driftCount = driftCount;
        r.periodicTriggers = periodicTriggers.get();
        r.reSelections = reSelections.get();
        r.selectionChangeCount = selectionChangeCount;
        r.accuracy = s.accuracyWindow; r.kappa = s.kappa; r.kappaPer = s.kappaPer;
        r.accMajority = mM.snapshot().accuracyWindow;
        r.accNoChange = mN.snapshot().accuracyWindow;
        r.ramHours = s.ramHoursGB;
        r.avgStability = Double.isNaN(s.featureStabilityRatio) ? 1.0 : s.featureStabilityRatio;
        r.finalSelection = selector.getCurrentSelection();

        System.out.printf(Locale.ROOT,
                "[proto] %-14s %-22s seed=%d  d=%d K=%d  acc=%.3f k=%.3f  drift=%d periodic=%d reSel=%d selChg=%d  maj=%.3f nc=%.3f%n",
                ds.name, v.name, seed, d, K,
                r.accuracy, r.kappa,
                r.driftCount, r.periodicTriggers, r.reSelections, r.selectionChangeCount,
                r.accMajority, r.accNoChange);
        return r;
    }

    private void validateAndSummarize(List<Result> all, Cfg cfg) throws Exception {
        Map<String, int[]> baselineSelByDsSeed = new HashMap<>();
        for (Result r : all) {
            if ("S1".equalsIgnoreCase(r.selector)) {
                baselineSelByDsSeed.put(r.dataset + "#" + r.seed + "#" + r.model, r.finalSelection);
            }
        }
        Map<String, Double> baselineKappa = new HashMap<>();
        for (Result r : all) {
            if ("S1".equalsIgnoreCase(r.selector)) {
                baselineKappa.put(r.dataset + "#" + r.seed + "#" + r.model, r.kappa);
            }
        }

        try (PrintWriter w = new PrintWriter(new FileWriter(
                Paths.get(cfg.outputDir, "E2_prototype_summary.txt").toFile()))) {
            w.println("=== E2 Prototype Summary ===");
            int pass = 0, fail = 0, warn = 0;

            boolean s2EverDrifted = false;
            boolean s3EverPeriodic = false;
            boolean anyAdaptiveSelChanged = false;

            for (Result r : all) {
                int[] s1Sel = baselineSelByDsSeed.get(r.dataset + "#" + r.seed + "#" + r.model);
                Double s1Kappa = baselineKappa.get(r.dataset + "#" + r.seed + "#" + r.model);
                r.s1IdenticalSelection = s1Sel != null && Arrays.equals(s1Sel, r.finalSelection);

                StringBuilder why = new StringBuilder();
                boolean isS1 = "S1".equalsIgnoreCase(r.selector);
                boolean isS2 = "S2".equalsIgnoreCase(r.selector);
                boolean isS3 = "S3".equalsIgnoreCase(r.selector);
                boolean isAdaptive = !isS1;

                if (isS2 && r.driftCount == 0)
                    why.append("S2: no drift alarm; ");
                if (isS3 && r.periodicTriggers == 0)
                    why.append("S3: no periodic trigger; ");
                if (isAdaptive && r.selectionChangeCount == 0)
                    why.append("no selection_changed during run; ");
                if (isAdaptive && r.s1IdenticalSelection)
                    why.append("final selection == S1 baseline; ");

                r.reason = why.toString();
                r.pass = isS1 || (
                        (!isS2 || r.driftCount > 0) &&
                                (!isS3 || r.periodicTriggers > 0) &&
                                (isAdaptive ? r.selectionChangeCount > 0 : true)
                );

                if (isS2 && r.driftCount > 0) s2EverDrifted = true;
                if (isS3 && r.periodicTriggers > 0) s3EverPeriodic = true;
                if (isAdaptive && r.selectionChangeCount > 0) anyAdaptiveSelChanged = true;

                String tag = r.pass ? "PASS" : "FAIL";
                if (isAdaptive && s1Kappa != null && Math.abs(r.kappa - s1Kappa) < 1e-6) {
                    warn++;
                    tag = "WARN";
                    r.reason += "kappa==S1(" + String.format(Locale.ROOT, "%.4f", s1Kappa) + "); ";
                }

                w.printf(Locale.ROOT,
                        "%-14s %-22s seed=%d d=%d K=%d  finalSel=%s  acc=%.4f k=%.4f kPer=%.4f  drift=%d periodic=%d reSel=%d selChg=%d  maj=%.4f nc=%.4f stab=%.3f  ram=%.6f  %s %s%n",
                        r.dataset, r.variant, r.seed, r.d, r.k,
                        Arrays.toString(r.finalSelection),
                        r.accuracy, r.kappa, r.kappaPer,
                        r.driftCount, r.periodicTriggers, r.reSelections, r.selectionChangeCount,
                        r.accMajority, r.accNoChange, r.avgStability,
                        r.ramHours, tag, r.reason);
                if (r.pass) pass++; else fail++;
            }

            w.println();
            w.println("--- GLOBAL CHECKS ---");
            w.println("S2 had >=1 drift alarm anywhere : " + s2EverDrifted);
            w.println("S3 had >=1 periodic trigger     : " + s3EverPeriodic);
            w.println("Any adaptive variant changed sel: " + anyAdaptiveSelChanged);
            w.printf("--- TOTAL: %d PASS / %d FAIL / %d WARN ---%n", pass, fail, warn);

            System.out.println("------------------------------------------------------------");
            System.out.printf("[proto] TOTAL: %d PASS / %d FAIL / %d WARN%n", pass, fail, warn);
            System.out.println("[proto] s2_drifted=" + s2EverDrifted
                    + "  s3_periodic=" + s3EverPeriodic
                    + "  any_sel_changed=" + anyAdaptiveSelChanged);
            System.out.println("[proto] details -> " + cfg.outputDir + "/E2_prototype_summary.txt");
        }
    }
}
