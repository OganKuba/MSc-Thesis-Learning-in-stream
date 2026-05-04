package thesis.experiments;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.options.OptionHandler;
import moa.streams.ArffFileStream;
import moa.streams.InstanceStream;
import thesis.detection.TwoLevelDriftDetector;
import thesis.discretization.PiDDiscretizer;
import thesis.evaluation.MetricsCollector;
import thesis.models.ARFWrapper;
import thesis.models.FeatureSpace;
import thesis.models.MajorityClassWrapper;
import thesis.models.ModelWrapper;
import thesis.models.NoChangeWrapper;
import thesis.models.SRPWrapper;
import thesis.pipeline.SyntheticStreamFactory;
import thesis.selection.AlarmTriggeredSelector;
import thesis.selection.DriftAwareSelector;
import thesis.selection.FeatureSelector;
import thesis.selection.InformationGainRanker;
import thesis.selection.PeriodicSelector;
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

public class E2AdaptiveFS {

    public enum TriggerType { NONE, ALARM, PERIODIC }

    public static final class DatasetSpec {
        public String name;
        public String type = "synthetic";
        public String generator;
        public String path;
        public int n = 100_000;
        public int noiseFeatures = 0;
        public double sigma = 0.01;
        public double speed = 0.001;
        public int driftFeatures = 5;
    }

    public static final class Variant {
        public String name;
        public String model;
        public String selector;
        public String detector = "ADWIN";
        public int periodicInterval = 1000;
        public int wPostDrift = 1000;
    }

    public static final class Cfg {
        public String experimentGroup = "E2";
        public String outputDir = "results/E2";
        public int warmup = 1500;
        public int windowSize = 1000;
        public int logEvery = 1000;
        public int ramSampleEvery = 200;
        public long maxInstances = Long.MAX_VALUE;
        public double detectorDelta = 0.002;
        public boolean skipMissingArff = true;
        public List<DatasetSpec> datasets = new ArrayList<>();
        public List<Variant> variants = new ArrayList<>();
        public List<Integer> seeds = new ArrayList<>();

        public static Cfg load(Path path) throws Exception {
            ObjectMapper m = new ObjectMapper();
            JsonNode r = m.readTree(path.toFile());
            Cfg c = new Cfg();
            c.experimentGroup = r.path("experiment_group").asText(c.experimentGroup);
            c.outputDir       = r.path("output_dir").asText(c.outputDir);
            c.warmup          = r.path("warmup").asInt(c.warmup);
            c.windowSize      = r.path("window_size").asInt(c.windowSize);
            c.logEvery        = r.path("log_every").asInt(c.logEvery);
            c.ramSampleEvery  = r.path("ram_sample_every").asInt(c.ramSampleEvery);
            c.maxInstances    = r.path("max_instances").asLong(c.maxInstances);
            c.detectorDelta   = r.path("detector_delta").asDouble(c.detectorDelta);
            c.skipMissingArff = r.path("skip_missing_arff").asBoolean(c.skipMissingArff);
            r.path("seeds").forEach(n -> c.seeds.add(n.asInt()));
            for (JsonNode d : r.path("datasets")) {
                DatasetSpec ds = new DatasetSpec();
                ds.name           = d.path("name").asText();
                ds.type           = d.path("type").asText(ds.type);
                ds.generator      = d.path("generator").asText(null);
                ds.path           = d.path("path").asText(null);
                ds.n              = d.path("n").asInt(ds.n);
                ds.noiseFeatures  = d.path("noise_features").asInt(ds.noiseFeatures);
                ds.sigma          = d.path("sigma").asDouble(ds.sigma);
                ds.speed          = d.path("speed").asDouble(ds.speed);
                ds.driftFeatures  = d.path("drift_features").asInt(ds.driftFeatures);
                c.datasets.add(ds);
            }
            for (JsonNode v : r.path("variants")) {
                Variant vv = new Variant();
                vv.name             = v.path("name").asText();
                vv.model            = v.path("model").asText();
                vv.selector         = v.path("selector").asText();
                vv.detector         = v.path("detector").asText(vv.detector);
                vv.periodicInterval = v.path("periodic_interval").asInt(vv.periodicInterval);
                vv.wPostDrift       = v.path("w_post_drift").asInt(vv.wPostDrift);
                c.variants.add(vv);
            }
            if (c.seeds.isEmpty())    throw new IllegalArgumentException("seeds empty");
            if (c.variants.isEmpty()) throw new IllegalArgumentException("variants empty");
            if (c.datasets.isEmpty()) throw new IllegalArgumentException("datasets empty");
            return c;
        }
    }

    public static final class RunSummary {
        public String dataset, variant, model, selector, detector;
        public int periodicInterval, seed, k, d;
        public long instances, driftCount, periodicTriggers, reSelections, selectionChangeCount;
        public double accuracy, kappa, kappaPer, accMajority, accNoChange;
        public double avgFeatureStability, ramHours;
    }

    public static void main(String[] args) throws Exception {
        String configPath = args.length > 0 ? args[0]
                : "stream/configs/E2_adaptive_fs.json";
        Cfg cfg = Cfg.load(Paths.get(configPath));
        new E2AdaptiveFS().run(cfg);
    }

    public void run(Cfg cfg) throws Exception {
        Files.createDirectories(Paths.get(cfg.outputDir));
        Path csvPath = Paths.get(cfg.outputDir, "E2_results.csv");
        Path summaryPath = Paths.get(cfg.outputDir, "E2_summary.csv");
        List<RunSummary> all = new ArrayList<>();

        try (PrintWriter csv = new PrintWriter(new FileWriter(csvPath.toFile()))) {
            csv.println(windowHeader());
            for (DatasetSpec ds : cfg.datasets) {
                if ("arff".equalsIgnoreCase(ds.type)
                        && (ds.path == null || !Files.exists(Paths.get(ds.path)))) {
                    System.err.printf("[E2] missing ARFF, skip: %s%n", ds.name);
                    if (cfg.skipMissingArff) continue;
                    throw new java.io.FileNotFoundException(String.valueOf(ds.path));
                }
                for (int seed : cfg.seeds) {
                    for (Variant v : cfg.variants) {
                        try {
                            RunSummary rs = runOne(cfg, ds, v, seed, csv);
                            all.add(rs);
                        } catch (Exception e) {
                            System.err.printf("[E2][FAIL] %s|%s|seed=%d -> %s%n",
                                    ds.name, v.name, seed, e);
                            e.printStackTrace(System.err);
                        }
                    }
                }
            }
        }

        writeSummary(all, summaryPath);
        writeValidation(all, Paths.get(cfg.outputDir, "validation_level1.txt"));
        System.out.println("[E2] Done -> " + cfg.outputDir);
    }

    static String windowHeader() {
        return "instance_num,dataset,variant,model,selector,detector,periodic_interval,seed,"
                + "selected_features,selected_count,selection_changed,trigger_type,drift_alarm,"
                + "drift_count,feature_stability_ratio,model_num_attributes,"
                + "accuracy_window,kappa_window,kappa_per_window,"
                + "majority_baseline_window,nochange_baseline_window,recovery_time,ram_hours";
    }

    public RunSummary runOne(Cfg cfg, DatasetSpec ds, Variant v, int seed,
                             PrintWriter csv) throws Exception {
        InstanceStream stream = buildStream(ds, seed);
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

        FeatureSelector selector = buildSelector(v, d, C, K);
        selector.initialize(window, labels);

        AtomicInteger triggerCode = new AtomicInteger(0);
        AtomicLong periodicTriggers = new AtomicLong(0);
        AtomicLong reSelections = new AtomicLong(0);
        attachListeners(selector, triggerCode, periodicTriggers, reSelections);

        ModelWrapper model    = buildModel(v.model, selector, header);
        ModelWrapper majority = new MajorityClassWrapper(selector, C);
        ModelWrapper noChange = new NoChangeWrapper(selector, C);
        TwoLevelDriftDetector detector = buildDetector(v.detector, d, cfg.detectorDelta);

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
        boolean selectionChangedSinceLastLog = false;

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
            boolean selChanged = !Arrays.equals(sel, lastSel);
            if (selChanged) {
                selectionChangedSinceLastLog = true;
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
                        joinSel(sel),
                        sel.length,
                        selectionChangedSinceLastLog ? "true" : "false",
                        trig,
                        alarm ? "true" : "false",
                        driftCount,
                        Double.isNaN(s.featureStabilityRatio) ? 1.0 : s.featureStabilityRatio,
                        model.getCurrentSelection().length + 1,
                        s.accuracyWindow, s.kappa, s.kappaPer,
                        mM.snapshot().accuracyWindow,
                        mN.snapshot().accuracyWindow,
                        s.lastRecoveryTime,
                        s.ramHoursGB);
                csv.flush();
                selectionChangedSinceLastLog = false;
            }
        }

        MetricsCollector.Snapshot s = mc.snapshot();
        RunSummary rs = new RunSummary();
        rs.dataset = ds.name; rs.variant = v.name; rs.model = v.model;
        rs.selector = v.selector; rs.detector = v.detector;
        rs.periodicInterval = v.periodicInterval; rs.seed = seed; rs.k = K; rs.d = d;
        rs.instances = n;
        rs.driftCount = driftCount;
        rs.periodicTriggers = periodicTriggers.get();
        rs.reSelections = reSelections.get();
        rs.selectionChangeCount = selectionChangeCount;
        rs.accuracy = s.accuracyWindow;
        rs.kappa = s.kappa;
        rs.kappaPer = s.kappaPer;
        rs.accMajority = mM.snapshot().accuracyWindow;
        rs.accNoChange = mN.snapshot().accuracyWindow;
        rs.avgFeatureStability = Double.isNaN(s.featureStabilityRatio) ? 1.0 : s.featureStabilityRatio;
        rs.ramHours = s.ramHoursGB;

        System.out.printf(Locale.ROOT,
                "[E2] %-14s %-22s seed=%d  d=%d K=%d  acc=%.4f k=%.4f kPer=%.4f  drift=%d periodic=%d reSel=%d selChg=%d  maj=%.4f nc=%.4f%n",
                ds.name, v.name, seed, d, K,
                rs.accuracy, rs.kappa, rs.kappaPer,
                rs.driftCount, rs.periodicTriggers, rs.reSelections, rs.selectionChangeCount,
                rs.accMajority, rs.accNoChange);
        return rs;
    }

    static void attachListeners(FeatureSelector selector,
                                AtomicInteger triggerCode,
                                AtomicLong periodicTriggers,
                                AtomicLong reSelections) {
        if (selector instanceof AlarmTriggeredSelector) {
            ((AlarmTriggeredSelector) selector).setEventListener(
                    new AlarmTriggeredSelector.EventListener() {
                        public void onAlarm(long idx, boolean accepted, Set<Integer> drift) {
                            if (accepted) triggerCode.set(1);
                        }
                        public void onCollectingStart(int w, int[] sel) {}
                        public void onReSelection(int[] o, int[] nw, double[] sc, boolean ch) {
                            reSelections.incrementAndGet();
                        }
                    });
        } else if (selector instanceof PeriodicSelector) {
            ((PeriodicSelector) selector).setEventListener(
                    new PeriodicSelector.EventListener() {
                        public void onPeriodicTick(long t, boolean triggered) {
                            if (triggered) {
                                periodicTriggers.incrementAndGet();
                                triggerCode.set(2);
                            }
                        }
                        public void onReSelection(long t, int[] o, int[] nw,
                                                  Set<Integer> ro, Set<Integer> ri,
                                                  double[] sc, long[] tn, double sr) {
                            reSelections.incrementAndGet();
                        }
                    });
        } else if (selector instanceof DriftAwareSelector) {
            ((DriftAwareSelector) selector).setEventListener(
                    new DriftAwareSelector.EventListener() {
                        public void onAlarm(long idx, boolean acc, Set<Integer> dr,
                                            Set<Integer> ds, Set<Integer> ss) {
                            if (acc) triggerCode.set(1);
                        }
                        public void onPeriodicTick(long t, boolean triggered) {
                            if (triggered) {
                                periodicTriggers.incrementAndGet();
                                if (triggerCode.get() == 0) triggerCode.set(2);
                            }
                        }
                        public void onSwap(DriftAwareSelector.TriggerType tr, long t,
                                           int[] o, int[] nw, Set<Integer> ro, Set<Integer> ri,
                                           double[] sc, long[] tn, double sr) {
                            reSelections.incrementAndGet();
                        }
                    });
        }
    }

    public static InstanceStream buildStream(DatasetSpec ds, int seed) {
        if ("arff".equalsIgnoreCase(ds.type)) {
            ArffFileStream s = new ArffFileStream(ds.path, -1);
            s.prepareForUse();
            return s;
        }
        String g = ds.generator == null ? ds.name : ds.generator;
        InstanceStream base;
        switch (g.toUpperCase(Locale.ROOT)) {
            case "SEA":        base = SyntheticStreamFactory.createSEA(seed, ds.n); break;
            case "HYPERPLANE": base = SyntheticStreamFactory.createHyperplane(seed, ds.sigma, ds.n); break;
            case "RANDOMRBF":  base = SyntheticStreamFactory.createRandomRBF(seed, ds.speed, ds.n); break;
            case "STAGGER":    base = SyntheticStreamFactory.createSTAGGER(seed, ds.n); break;
            case "CUSTOMFEATUREDRIFT":
            case "FEATUREDRIFT":
                base = SyntheticStreamFactory.createCustomFeatureDrift(seed, ds.driftFeatures, ds.sigma, ds.n);
                break;
            default: throw new IllegalArgumentException("Unknown generator: " + g);
        }
        return ds.noiseFeatures > 0
                ? SyntheticStreamFactory.addNoiseFeatures(base, ds.noiseFeatures, seed)
                : base;
    }

    public static FeatureSelector buildSelector(Variant v, int d, int C, int K) {
        switch (v.selector.toUpperCase(Locale.ROOT)) {
            case "S1":
                return new StaticFeatureSelector(d, C);
            case "S2":
                return new AlarmTriggeredSelector(
                        d, C, K, Math.max(50, v.wPostDrift),
                        new PiDDiscretizer(d, C),
                        (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc));
            case "S3":
                return new PeriodicSelector(
                        d, C, K, Math.max(100, v.periodicInterval), 100,
                        new PiDDiscretizer(d, C),
                        (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc));
            case "S4":
                return new DriftAwareSelector(
                        d, C, K, Math.max(100, v.periodicInterval), 100, Math.max(50, v.wPostDrift),
                        new PiDDiscretizer(d, C),
                        (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc));
            default:
                throw new IllegalArgumentException("Unknown selector: " + v.selector);
        }
    }

    public static ModelWrapper buildModel(String name, FeatureSelector selector,
                                          InstancesHeader header) {
        switch (name.toUpperCase(Locale.ROOT)) {
            case "ARF": return new ARFWrapper(selector, header, 10, 6.0, false, true);
            case "SRP": return new SRPWrapper(selector, header, 10, 6.0, false, true);
            default: throw new IllegalArgumentException(
                    "E2 supports only ARF/SRP; got: " + name + " (HT belongs to E1, DASRP to E3)");
        }
    }

    public static TwoLevelDriftDetector buildDetector(String name, int d, double delta) {
        TwoLevelDriftDetector.Config c = new TwoLevelDriftDetector.Config(d);
        c.level1Delta = delta;
        switch (name.toUpperCase(Locale.ROOT)) {
            case "ADWIN":  c.level1Type = TwoLevelDriftDetector.Level1Type.ADWIN;  break;
            case "HDDM_A": c.level1Type = TwoLevelDriftDetector.Level1Type.HDDM_A; break;
            case "HDDM_W": c.level1Type = TwoLevelDriftDetector.Level1Type.HDDM_W; break;
            default: throw new IllegalArgumentException("Unknown detector: " + name);
        }
        return new TwoLevelDriftDetector(c);
    }

    static String joinSel(int[] sel) {
        if (sel == null || sel.length == 0) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < sel.length; i++) {
            if (i > 0) sb.append('|');
            sb.append(sel[i]);
        }
        return sb.toString();
    }

    static void writeSummary(List<RunSummary> all, Path path) throws Exception {
        try (PrintWriter w = new PrintWriter(new FileWriter(path.toFile()))) {
            w.println("dataset,variant,model,selector,detector,periodic_interval,seed,"
                    + "instances,d,k,accuracy,kappa,kappa_per,"
                    + "drift_count,periodic_triggers,re_selections,selection_change_count,"
                    + "avg_feature_stability,acc_majority,acc_nochange,ram_hours");
            for (RunSummary s : all) {
                w.printf(Locale.ROOT,
                        "%s,%s,%s,%s,%s,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.6f%n",
                        s.dataset, s.variant, s.model, s.selector, s.detector, s.periodicInterval, s.seed,
                        s.instances, s.d, s.k, s.accuracy, s.kappa, s.kappaPer,
                        s.driftCount, s.periodicTriggers, s.reSelections, s.selectionChangeCount,
                        s.avgFeatureStability, s.accMajority, s.accNoChange, s.ramHours);
            }
        }
    }

    static void writeValidation(List<RunSummary> all, Path path) throws Exception {
        Map<String, Map<String, List<Double>>> agg = new HashMap<>();
        for (RunSummary s : all) {
            agg.computeIfAbsent(s.dataset, k -> new HashMap<>())
                    .computeIfAbsent(s.variant, k -> new ArrayList<>())
                    .add(s.kappa);
        }
        try (PrintWriter w = new PrintWriter(new FileWriter(path.toFile()))) {
            w.println("E2 validation: adaptive variants vs S1 baselines (mean kappa)");
            int pass = 0, fail = 0, warn = 0;
            for (Map.Entry<String, Map<String, List<Double>>> e : agg.entrySet()) {
                String ds = e.getKey();
                Map<String, Double> mean = new HashMap<>();
                e.getValue().forEach((var, ks) ->
                        mean.put(var, ks.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN)));
                w.println("[" + ds + "] " + mean);
                Double arfS1 = mean.get("ARF+S1");
                Double srpS1 = mean.get("SRP+S1");
                for (Map.Entry<String, Double> me : mean.entrySet()) {
                    String var = me.getKey();
                    double v = me.getValue();
                    if (var.endsWith("+S1")) continue;
                    Double base = var.startsWith("ARF") ? arfS1 : (var.startsWith("SRP") ? srpS1 : null);
                    if (base == null || base.isNaN()) continue;
                    if (Math.abs(v - base) < 1e-6) {
                        warn++;
                        w.printf(Locale.ROOT, "   WARN  %s == S1 baseline (%.4f) — selector ineffective?%n", var, v);
                    } else if (v >= base) {
                        pass++;
                        w.printf(Locale.ROOT, "   PASS  %s mean=%.4f (vs S1=%.4f)%n", var, v, base);
                    } else {
                        fail++;
                        w.printf(Locale.ROOT, "   FAIL  %s mean=%.4f < S1=%.4f%n", var, v, base);
                    }
                }
                w.println();
            }
            w.printf("TOTAL: pass=%d fail=%d warn=%d%n", pass, fail, warn);
        }
    }
}
