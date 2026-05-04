package thesis.experiments;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.options.OptionHandler;
import moa.streams.InstanceStream;
import thesis.detection.ADWINChangeDetector;
import thesis.detection.DriftDetector;
import thesis.detection.HDDMChangeDetector;
import thesis.detection.KSWINSingleFeature;
import thesis.evaluation.MetricsCollector;
import thesis.experiments.E4DriftAnalysis.AlarmEvent;
import thesis.experiments.E4DriftAnalysis.DetectionStats;
import thesis.experiments.E4DriftAnalysis.GeneratorSpec;
import thesis.experiments.E4DriftAnalysis.Magnitude;
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public class E5Detectors {

    public interface DetectorAdapter {
        void update(double err);
        boolean isAlarm();
        void reset();
        String name();
    }

    public static final class AdwinAdapter implements DetectorAdapter {
        private final ADWINChangeDetector adwin;
        private final double delta;
        private boolean lastAlarm;
        public AdwinAdapter(double delta) { this.delta = delta; this.adwin = new ADWINChangeDetector(delta); }
        public void update(double err) { adwin.update(err); lastAlarm = adwin.isChangeDetected(); if (lastAlarm) adwin.reset(); }
        public boolean isAlarm() { return lastAlarm; }
        public void reset() { adwin.reset(); lastAlarm = false; }
        public String name() { return "ADWIN(delta=" + delta + ")"; }
    }

    public static final class HddmAdapter implements DetectorAdapter {
        private final DriftDetector hddm;
        private final String tag;
        private boolean lastAlarm;
        public HddmAdapter(boolean wType, double drift, double warn, double lambda) {
            this.hddm = wType ? HDDMChangeDetector.ofW(drift, warn, lambda) : HDDMChangeDetector.ofA(drift, warn);
            this.tag = wType ? ("HDDM_W(d=" + drift + ",w=" + warn + ",l=" + lambda + ")")
                    : ("HDDM_A(d=" + drift + ",w=" + warn + ")");
        }
        public void update(double err) { hddm.update(err); lastAlarm = hddm.isChangeDetected(); if (lastAlarm) hddm.reset(); }
        public boolean isAlarm() { return lastAlarm; }
        public void reset() { hddm.reset(); lastAlarm = false; }
        public String name() { return tag; }
    }

    public static final class KswinGlobalAdapter implements DetectorAdapter {
        private final KSWINSingleFeature ks;
        private final int windowSize;
        private final double alpha;
        private boolean lastAlarm;
        private int sinceReset;
        private int testEvery;
        public KswinGlobalAdapter(int windowSize, double alpha, int testEvery) {
            this.windowSize = windowSize; this.alpha = alpha;
            this.ks = new KSWINSingleFeature(windowSize, alpha);
            this.testEvery = Math.max(1, testEvery);
        }
        public void update(double err) {
            ks.update(err); sinceReset++; lastAlarm = false;
            if (ks.isReady() && sinceReset % testEvery == 0) {
                if (ks.testDrift()) {
                    lastAlarm = true;
                    ks.promoteCurrentToReference();
                    sinceReset = 0;
                }
            }
        }
        public boolean isAlarm() { return lastAlarm; }
        public void reset() { ks.reset(); lastAlarm = false; sinceReset = 0; }
        public String name() { return "KSWIN_GLOBAL(W=" + windowSize + ",a=" + alpha + ")"; }
    }

    public static final class Cfg {
        public String experimentGroup = "E5";
        public String outputDir = "results/E5";
        public int warmup = 1500;
        public int windowSize = 1000;
        public int logEvery = 1000;
        public int ramSampleEvery = 200;
        public int toleranceWindow = 500;
        public long maxInstances = Long.MAX_VALUE;
        public double f1Threshold = 0.5;
        public double recoveryCapFraction = 0.5;
        public int methodPeriodicInterval = 500;
        public int methodWPostDrift = 300;
        public double adwinDelta = 0.002;
        public double hddmDriftConfidence = 0.001;
        public double hddmWarningConfidence = 0.005;
        public double hddmLambda = 0.05;
        public int kswinWindow = 200;
        public double kswinAlpha = 0.005;
        public int kswinTestEvery = 50;
        public int driftBeforeWindow = 500;
        public int driftAfterWindow = 500;
        public List<GeneratorSpec> generators = new ArrayList<>();
        public List<String> detectors = new ArrayList<>();
        public List<String> models = new ArrayList<>();
        public List<String> magnitudes = new ArrayList<>();
        public List<Integer> seeds = new ArrayList<>();

        public static Cfg load(Path path) throws Exception {
            ObjectMapper m = new ObjectMapper();
            JsonNode r = m.readTree(path.toFile());
            Cfg c = new Cfg();
            c.experimentGroup        = r.path("experiment_group").asText(c.experimentGroup);
            c.outputDir              = r.path("output_dir").asText(c.outputDir);
            c.warmup                 = r.path("warmup").asInt(c.warmup);
            c.windowSize             = r.path("window_size").asInt(c.windowSize);
            c.logEvery               = r.path("log_every").asInt(c.logEvery);
            c.ramSampleEvery         = r.path("ram_sample_every").asInt(c.ramSampleEvery);
            c.toleranceWindow        = r.path("tolerance_window").asInt(c.toleranceWindow);
            c.maxInstances           = r.path("max_instances").asLong(c.maxInstances);
            c.f1Threshold            = r.path("f1_threshold").asDouble(c.f1Threshold);
            c.recoveryCapFraction    = r.path("recovery_cap_fraction").asDouble(c.recoveryCapFraction);
            c.methodPeriodicInterval = r.path("method_periodic_interval").asInt(c.methodPeriodicInterval);
            c.methodWPostDrift       = r.path("method_w_post_drift").asInt(c.methodWPostDrift);
            c.adwinDelta             = r.path("adwin_delta").asDouble(c.adwinDelta);
            c.hddmDriftConfidence    = r.path("hddm_drift_confidence").asDouble(c.hddmDriftConfidence);
            c.hddmWarningConfidence  = r.path("hddm_warning_confidence").asDouble(c.hddmWarningConfidence);
            c.hddmLambda             = r.path("hddm_lambda").asDouble(c.hddmLambda);
            c.kswinWindow            = r.path("kswin_window").asInt(c.kswinWindow);
            c.kswinAlpha             = r.path("kswin_alpha").asDouble(c.kswinAlpha);
            c.kswinTestEvery         = r.path("kswin_test_every").asInt(c.kswinTestEvery);
            c.driftBeforeWindow      = r.path("drift_before_window").asInt(c.driftBeforeWindow);
            c.driftAfterWindow       = r.path("drift_after_window").asInt(c.driftAfterWindow);
            r.path("seeds").forEach(n -> c.seeds.add(n.asInt()));
            r.path("detectors").forEach(n -> c.detectors.add(n.asText().toUpperCase(Locale.ROOT)));
            r.path("models").forEach(n -> c.models.add(n.asText()));
            r.path("magnitudes").forEach(n -> c.magnitudes.add(n.asText().toUpperCase(Locale.ROOT)));
            for (JsonNode g : r.path("generators")) {
                GeneratorSpec gs = new GeneratorSpec();
                gs.name           = g.path("name").asText();
                gs.n              = g.path("n").asInt(gs.n);
                gs.noiseFeatures  = g.path("noise_features").asInt(gs.noiseFeatures);
                gs.driftFeatures  = g.path("drift_features").asInt(gs.driftFeatures);
                gs.abrupt         = isAbrupt(gs.name);
                c.generators.add(gs);
            }
            if (c.seeds.isEmpty())      throw new IllegalArgumentException("seeds empty");
            if (c.detectors.isEmpty())  throw new IllegalArgumentException("detectors empty");
            if (c.models.isEmpty())     throw new IllegalArgumentException("models empty");
            if (c.magnitudes.isEmpty()) throw new IllegalArgumentException("magnitudes empty");
            if (c.generators.isEmpty()) throw new IllegalArgumentException("generators empty");
            return c;
        }
    }

    public static final class RunResult {
        public String generator, detector, model, magnitude;
        public int seed, d;
        public long instances;
        public double accuracy, kappa, kappaPer, recoveryTime, ramHours, featureStability, lastFeatureStability;
        public double throughput, peakMB;
        public long driftCount, selectionChangeCount;
        public int[] gtPositions = new int[0];
        public List<AlarmEvent> alarms = new ArrayList<>();
        public DetectionStats detection = new DetectionStats();
        public boolean abrupt;
        public String status = "OK";
    }

    public static void main(String[] args) throws Exception {

        Path configPath = args.length > 0
                ? Paths.get(args[0])
                : findDefaultConfig();

        if (!Files.exists(configPath)) {
            throw new RuntimeException(
                    "Config not found: " + configPath.toAbsolutePath() +
                            "\nWorking dir: " + Paths.get(".").toAbsolutePath()
            );
        }

        Cfg cfg = Cfg.load(configPath);
        new E5Detectors().run(cfg);
    }

    private static Path findDefaultConfig() {
        List<Path> candidates = List.of(
                Paths.get("src/main/java/thesis/experiments/e5_detectors.json"),
                Paths.get("experiments/e5_detectors.json"),
                Paths.get("src/main/resources/e5_detectors.json")
        );
        for (Path p : candidates) {
            if (Files.exists(p)) return p;
        }
        return candidates.get(0);
    }

    public void run(Cfg cfg) throws Exception {
        Files.createDirectories(Paths.get(cfg.outputDir));
        Path winCsv    = Paths.get(cfg.outputDir, "E5_window.csv");
        Path alrCsv    = Paths.get(cfg.outputDir, "E5_alarms.csv");
        Path sumCsv    = Paths.get(cfg.outputDir, "E5_summary.csv");
        Path driftsCsv = Paths.get(cfg.outputDir, "E5_drifts.csv");

        int runTotal = cfg.generators.size() * cfg.magnitudes.size()
                * cfg.seeds.size() * cfg.models.size() * cfg.detectors.size();
        int runIdx = 0;

        List<RunResult> all = new ArrayList<>();

        try (PrintWriter w   = new PrintWriter(new FileWriter(winCsv.toFile()));
             PrintWriter a   = new PrintWriter(new FileWriter(alrCsv.toFile()));
             PrintWriter dW  = new PrintWriter(new FileWriter(driftsCsv.toFile()))) {

            w.println(windowHeader());
            a.println(alarmHeader());
            dW.println("dataset,variant,seed,alarm_at,kappa_before_500,kappa_after_500,recovery_instances,drift_type");

            for (GeneratorSpec g : cfg.generators) {
                for (String mag : cfg.magnitudes) {
                    Magnitude m = Magnitude.valueOf(mag);
                    for (int seed : cfg.seeds) {
                        for (String model : cfg.models) {
                            for (String det : cfg.detectors) {
                                runIdx++;
                                String tag = g.name + "|" + det + "|" + model + "|" + mag + "|seed=" + seed;
                                System.out.printf(Locale.ROOT,
                                        "[E5] (%d/%d) START %s | cap=%s%n",
                                        runIdx, runTotal, tag,
                                        cfg.maxInstances == Long.MAX_VALUE ? "ALL" : Long.toString(cfg.maxInstances));
                                try {
                                    RunResult r = runOne(cfg, g, m, det, model, seed, w, dW, runIdx, runTotal);
                                    writeAlarmsCsv(a, r);
                                    all.add(r);
                                } catch (Exception ex) {
                                    System.err.printf("[E5][FAIL] (%d/%d) %s -> %s%n",
                                            runIdx, runTotal, tag, ex);
                                    ex.printStackTrace(System.err);
                                    RunResult fr = new RunResult();
                                    fr.generator = g.name; fr.detector = det; fr.model = model;
                                    fr.magnitude = mag; fr.seed = seed; fr.status = "FAIL";
                                    all.add(fr);
                                }
                            }
                        }
                    }
                }
            }
        }
        writeSummary(all, sumCsv);
        writeAggregated(cfg, all);
        writeRanking(cfg, all);
        writeSanityChecks(cfg, all);
        System.out.println("[E5] Done -> " + cfg.outputDir);
    }

    public RunResult runOne(Cfg cfg, GeneratorSpec g, Magnitude mag,
                            String detName, String modelName, int seed,
                            PrintWriter wcsv, PrintWriter driftsCsv,
                            int runIdx, int runTotal) throws Exception {

        InstanceStream stream = E4DriftAnalysis.buildStreamWithMagnitude(g, mag, seed);
        if (stream instanceof OptionHandler) ((OptionHandler) stream).prepareForUse();
        InstancesHeader header = stream.getHeader();
        int d = header.numAttributes() - 1;
        int C = header.numClasses();
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

        StaticFeatureSelector baselineSel = new StaticFeatureSelector(d, C);
        baselineSel.initialize(window, labels);

        String[] parts = modelName.split("\\+");
        String mPart = parts[0];
        String sPart = parts.length > 1 ? parts[1] : "S1";
        E2AdaptiveFS.Variant v = new E2AdaptiveFS.Variant();
        v.name = modelName; v.model = mPart; v.selector = sPart;
        v.detector = detName;
        v.periodicInterval = cfg.methodPeriodicInterval;
        v.wPostDrift       = cfg.methodWPostDrift;
        int K = StaticFeatureSelector.defaultK(d);
        FeatureSelector fs = E2AdaptiveFS.buildSelector(v, d, C, K);
        fs.initialize(window, labels);
        ModelWrapper main = E2AdaptiveFS.buildModel(mPart, fs, header);

        DetectorAdapter detector = buildAdapter(detName, cfg);

        ModelWrapper majority = new MajorityClassWrapper(baselineSel, C);
        ModelWrapper noChange = new NoChangeWrapper(baselineSel, C);
        for (int i = 0; i < collected; i++) {
            majority.train(null, labels[i]);
            noChange.train(null, labels[i]);
        }

        MetricsCollector mc = new MetricsCollector(C, cfg.windowSize, cfg.logEvery, cfg.ramSampleEvery);
        MetricsCollector mM = new MetricsCollector(C, cfg.windowSize, cfg.logEvery, cfg.ramSampleEvery);
        MetricsCollector mN = new MetricsCollector(C, cfg.windowSize, cfg.logEvery, cfg.ramSampleEvery);

        String dlTag = g.name + "|" + mag.name() + "|" + detName + "|" + modelName;
        DriftLogger dl = new DriftLogger(driftsCsv, dlTag, modelName, seed,
                cfg.driftBeforeWindow, cfg.driftAfterWindow);

        long n = collected;
        long driftCount = 0, selChgCount = 0;
        int[] lastSel = main.getCurrentSelection().clone();
        List<AlarmEvent> alarms = new ArrayList<>();
        int[] gtPositions = E4DriftAnalysis.groundTruthPositions(g, cfg.warmup);

        long wallStart = System.nanoTime();
        long lastLogN = n;
        long lastLogNanos = wallStart;
        double lastThr = 0.0;

        while (stream.hasMoreInstances() && n < cfg.maxInstances) {
            Instance raw = stream.nextInstance().getData();
            int y = (int) raw.classValue();
            double[] xv = space.extractFeatures(raw);

            long t0 = System.nanoTime();
            int yhat = main.predict(raw);
            long elapsed = System.nanoTime() - t0;
            int yMaj = majority.predict(raw);
            int yNC  = noChange.predict(raw);

            double err = (yhat == y) ? 0.0 : 1.0;
            detector.update(err);
            boolean alarm = detector.isAlarm();
            Set<Integer> drifting = Set.of();

            mc.update(y, yhat, elapsed);
            mM.update(y, yMaj, 0);
            mN.update(y, yNC, 0);

            dl.tick(n, mc.snapshot().kappa);

            if (alarm) {
                mc.onDriftAlarm();
                driftCount++;
                AlarmEvent ev = new AlarmEvent();
                ev.instance = n;
                alarms.add(ev);
                dl.onAlarm(n, mc, false);
            }

            fs.update(xv, y, alarm, drifting);
            main.train(raw, y, alarm, drifting);
            majority.train(raw, y);
            noChange.train(raw, y);

            int[] sel = main.getCurrentSelection();
            mc.onSelectionChanged(sel);
            if (!Arrays.equals(sel, lastSel)) { selChgCount++; lastSel = sel.clone(); }

            n++;
            if (cfg.logEvery > 0 && n % cfg.logEvery == 0) {
                MetricsCollector.Snapshot s = mc.snapshot();
                long nowNs = System.nanoTime();
                double dtSec = (nowNs - lastLogNanos) / 1e9;
                long dN = n - lastLogN;
                double thr = (dtSec > 0) ? (dN / dtSec) : 0.0;
                lastLogN = n; lastLogNanos = nowNs; lastThr = thr;

                int trueDriftHere = 0;
                for (int p : gtPositions) {
                    if (p > n - cfg.logEvery && p <= n) { trueDriftHere = 1; break; }
                }
                wcsv.printf(Locale.ROOT,
                        "%d,%s,%s,%s,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%.4f,%.6f,%.2f,%.2f%n",
                        n, g.name, detName, modelName, mag.name(), seed,
                        trueDriftHere, alarm ? 1 : 0,
                        s.accuracyWindow, s.kappa, s.kappaPer,
                        mM.snapshot().accuracyWindow, mN.snapshot().accuracyWindow,
                        s.lastRecoveryTime, driftCount, selChgCount,
                        Double.isNaN(s.featureStabilityRatio) ? 1.0 : s.featureStabilityRatio,
                        s.ramHoursGB, thr, s.peakMB);
                wcsv.flush();

                if (cfg.maxInstances == Long.MAX_VALUE && n % (cfg.logEvery * 10L) == 0) {
                    System.out.printf(Locale.ROOT,
                            "[E5] (%d/%d)   ... n=%d  k=%.4f  drift=%d  thr=%.0f i/s  peak=%.1fMB%n",
                            runIdx, runTotal, n, s.kappa, driftCount, thr, s.peakMB);
                }
            }
        }

        MetricsCollector.Snapshot s = mc.snapshot();
        dl.flushPending(n, s.kappa);

        RunResult r = new RunResult();
        r.generator = g.name; r.detector = detName; r.model = modelName; r.magnitude = mag.name();
        r.seed = seed; r.d = d;
        r.instances = n;
        r.accuracy = s.accuracyWindow; r.kappa = s.kappa; r.kappaPer = s.kappaPer;
        r.recoveryTime = Double.isNaN(s.avgRecoveryTime) ? -1.0 : s.avgRecoveryTime;
        long monitored = n - cfg.warmup;
        if (!g.abrupt && r.recoveryTime > cfg.recoveryCapFraction * monitored) r.recoveryTime = -1.0;
        r.featureStability = Double.isNaN(s.featureStabilityRatio) ? 1.0 : s.featureStabilityRatio;
        r.lastFeatureStability = Double.isNaN(s.lastFeatureStabilityRatio) ? 1.0 : s.lastFeatureStabilityRatio;
        r.ramHours = s.ramHoursGB;
        r.peakMB   = s.peakMB;
        double elapsedSec = (System.nanoTime() - wallStart) / 1e9;
        double thrAvg = (elapsedSec > 0) ? (n - collected) / elapsedSec : lastThr;
        r.throughput = thrAvg > 0 ? thrAvg : lastThr;
        r.driftCount = driftCount;
        r.selectionChangeCount = selChgCount;
        r.gtPositions = gtPositions;
        r.alarms = alarms;
        r.abrupt = g.abrupt;
        r.detection = E4DriftAnalysis.computeDetection(gtPositions, alarms, cfg.toleranceWindow, g.abrupt, n);
        r.status = "OK";

        System.out.printf(Locale.ROOT,
                "[E5] (%d/%d) DONE  %-18s %-18s %-8s %-6s seed=%d  k=%.4f  drift=%d  TP=%d FP=%d FN=%d  P=%.3f R=%.3f F1=%.3f delay=%.1f  recov=%.1f  thr=%.0f i/s  peak=%.1fMB%n",
                runIdx, runTotal, g.name, detName, modelName, mag.name(), seed,
                r.kappa, r.driftCount,
                r.detection.tp, r.detection.fp, r.detection.fn,
                r.detection.precision, r.detection.recall, r.detection.f1,
                r.detection.meanDetectionDelay, r.recoveryTime, r.throughput, r.peakMB);
        return r;
    }

    public static DetectorAdapter buildAdapter(String detName, Cfg cfg) {
        switch (detName.toUpperCase(Locale.ROOT)) {
            case "ADWIN":        return new AdwinAdapter(cfg.adwinDelta);
            case "HDDM_A":       return new HddmAdapter(false, cfg.hddmDriftConfidence, cfg.hddmWarningConfidence, cfg.hddmLambda);
            case "HDDM_W":       return new HddmAdapter(true,  cfg.hddmDriftConfidence, cfg.hddmWarningConfidence, cfg.hddmLambda);
            case "KSWIN_GLOBAL": return new KswinGlobalAdapter(cfg.kswinWindow, cfg.kswinAlpha, cfg.kswinTestEvery);
            default: throw new IllegalArgumentException("Unknown detector: " + detName);
        }
    }

    static boolean isAbrupt(String generator) {
        String g = generator.toUpperCase(Locale.ROOT);
        return g.equals("SEA") || g.equals("STAGGER");
    }

    public static String windowHeader() {
        return "instance_num,generator,detector,model,magnitude,seed,"
                + "true_drift,detected_drift,accuracy_window,kappa_window,kappa_per_window,"
                + "majority_baseline_window,nochange_baseline_window,"
                + "recovery_time,drift_count,selection_change_count,"
                + "feature_stability,ram_hours,throughput_inst_per_sec,peak_ram_mb";
    }

    public static String alarmHeader() {
        return "generator,detector,model,magnitude,seed,instance_num,is_tp,delay,matched_gt";
    }

    public static void writeAlarmsCsv(PrintWriter w, RunResult r) {
        if ("FAIL".equals(r.status)) return;
        for (AlarmEvent a : r.alarms) {
            w.printf(Locale.ROOT, "%s,%s,%s,%s,%d,%d,%s,%d,%d%n",
                    r.generator, r.detector, r.model, r.magnitude, r.seed,
                    a.instance, a.isTP ? "TP" : "FP", a.delay, a.matchedGT);
        }
        w.flush();
    }

    public static void writeSummary(List<RunResult> all, Path path) throws Exception {
        try (PrintWriter w = new PrintWriter(new FileWriter(path.toFile()))) {
            w.println("generator,detector,model,magnitude,seed,instances,d,"
                    + "accuracy,kappa,kappa_per,recovery_time,feature_stability,last_feature_stability,ram_hours,"
                    + "throughput_inst_per_sec,peak_ram_mb,"
                    + "drift_count,selection_change_count,"
                    + "tp,fp,fn,precision,recall,f1,false_alarm_rate,mean_detection_delay,"
                    + "status,gt_positions");
            for (RunResult r : all) {
                if ("FAIL".equals(r.status)) {
                    w.printf(Locale.ROOT,
                            "%s,%s,%s,%s,%d,0,0,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,0,0,0,0,0,NaN,NaN,NaN,NaN,NaN,FAIL,\"\"%n",
                            r.generator, r.detector, r.model, r.magnitude, r.seed);
                    continue;
                }
                w.printf(Locale.ROOT,
                        "%s,%s,%s,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%.2f,%.2f,%d,%d,"
                                + "%d,%d,%d,%.4f,%.4f,%.4f,%.6f,%.2f,%s,\"%s\"%n",
                        r.generator, r.detector, r.model, r.magnitude, r.seed, r.instances, r.d,
                        r.accuracy, r.kappa, r.kappaPer, r.recoveryTime,
                        r.featureStability, r.lastFeatureStability, r.ramHours,
                        r.throughput, r.peakMB,
                        r.driftCount, r.selectionChangeCount,
                        r.detection.tp, r.detection.fp, r.detection.fn,
                        r.detection.precision, r.detection.recall, r.detection.f1,
                        r.detection.falseAlarmRate, r.detection.meanDetectionDelay,
                        r.status, joinIntArr(r.gtPositions));
            }
        }
    }

    public static void writeAggregated(Cfg cfg, List<RunResult> all) throws Exception {
        Path p = Paths.get(cfg.outputDir, "E5_aggregated.csv");
        Map<String, List<RunResult>> grp = new LinkedHashMap<>();
        for (RunResult r : all) {
            if ("FAIL".equals(r.status)) continue;
            String key = r.generator + "|" + r.detector + "|" + r.model + "|" + r.magnitude;
            grp.computeIfAbsent(key, k -> new ArrayList<>()).add(r);
        }
        try (PrintWriter w = new PrintWriter(new FileWriter(p.toFile()))) {
            w.println("generator,detector,model,magnitude,n_seeds,"
                    + "mean_kappa,std_kappa,mean_recovery,std_recovery,mean_drift_count,"
                    + "mean_f1,std_f1,mean_precision,mean_recall,mean_false_alarm_rate,mean_detection_delay,"
                    + "mean_throughput,mean_peak_mb");
            for (Map.Entry<String, List<RunResult>> e : grp.entrySet()) {
                List<RunResult> rs = e.getValue();
                String[] keys = e.getKey().split("\\|");
                double[] kap = rs.stream().mapToDouble(r -> r.kappa).toArray();
                double[] rec = rs.stream().mapToDouble(r -> r.recoveryTime).filter(v -> v >= 0).toArray();
                double[] dc  = rs.stream().mapToDouble(r -> r.driftCount).toArray();
                double[] f1  = rs.stream().mapToDouble(r -> r.detection.f1).toArray();
                double[] pr  = rs.stream().mapToDouble(r -> r.detection.precision).toArray();
                double[] re  = rs.stream().mapToDouble(r -> r.detection.recall).toArray();
                double[] far = rs.stream().mapToDouble(r -> r.detection.falseAlarmRate).toArray();
                double[] del = rs.stream().mapToDouble(r -> r.detection.meanDetectionDelay).filter(v -> v >= 0).toArray();
                double[] thr = rs.stream().mapToDouble(r -> r.throughput).toArray();
                double[] peak= rs.stream().mapToDouble(r -> r.peakMB).toArray();
                w.printf(Locale.ROOT,
                        "%s,%s,%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%.2f,%.4f,%.4f,%.4f,%.4f,%.6f,%.2f,%.2f,%.2f%n",
                        keys[0], keys[1], keys[2], keys[3], rs.size(),
                        mean(kap), std(kap), mean(rec), std(rec), mean(dc),
                        mean(f1), std(f1), mean(pr), mean(re), mean(far), mean(del),
                        mean(thr), mean(peak));
            }
        }
    }

    public static void writeRanking(Cfg cfg, List<RunResult> all) throws Exception {
        Path p = Paths.get(cfg.outputDir, "E5_ranking.csv");
        Map<String, double[]> agg = new LinkedHashMap<>();
        Map<String, Integer> cnt = new LinkedHashMap<>();
        for (RunResult r : all) {
            if ("FAIL".equals(r.status)) continue;
            double[] a = agg.computeIfAbsent(r.detector, k -> new double[5]);
            a[0] += r.kappa;
            a[1] += r.detection.f1;
            a[2] += r.recoveryTime < 0 ? 0 : r.recoveryTime;
            a[3] += r.detection.falseAlarmRate;
            a[4] += r.detection.meanDetectionDelay < 0 ? 0 : r.detection.meanDetectionDelay;
            cnt.merge(r.detector, 1, Integer::sum);
        }
        List<Map.Entry<String, double[]>> list = new ArrayList<>(agg.entrySet());
        list.sort((x, y) -> {
            double sx = x.getValue()[1] / cnt.get(x.getKey());
            double sy = y.getValue()[1] / cnt.get(y.getKey());
            return Double.compare(sy, sx);
        });
        try (PrintWriter w = new PrintWriter(new FileWriter(p.toFile()))) {
            w.println("rank,detector,n_runs,mean_f1,mean_kappa,mean_recovery,mean_false_alarm_rate,mean_detection_delay");
            int rank = 1;
            for (Map.Entry<String, double[]> e : list) {
                int n = cnt.get(e.getKey());
                double[] a = e.getValue();
                w.printf(Locale.ROOT, "%d,%s,%d,%.4f,%.4f,%.4f,%.6f,%.2f%n",
                        rank++, e.getKey(), n,
                        a[1] / n, a[0] / n, a[2] / n, a[3] / n, a[4] / n);
            }
        }
    }

    public static void writeSanityChecks(Cfg cfg, List<RunResult> all) throws Exception {
        Path p = Paths.get(cfg.outputDir, "E5_sanity_checks.txt");
        Map<String, List<RunResult>> byKey = new LinkedHashMap<>();
        for (RunResult r : all) {
            if ("FAIL".equals(r.status)) continue;
            String key = r.generator + "|" + r.magnitude + "|" + r.seed + "|" + r.model;
            byKey.computeIfAbsent(key, k -> new ArrayList<>()).add(r);
        }
        try (PrintWriter w = new PrintWriter(new FileWriter(p.toFile()))) {
            w.println("=== E5 Sanity Checks ===");
            int identicalCount = 0, neverFiredCount = 0, alwaysFiredCount = 0, suspiciousDelayCount = 0;
            for (Map.Entry<String, List<RunResult>> e : byKey.entrySet()) {
                List<RunResult> rs = e.getValue();
                Map<Long, Integer> driftCounts = new LinkedHashMap<>();
                for (RunResult r : rs) driftCounts.merge(r.driftCount, 1, Integer::sum);
                if (driftCounts.size() == 1 && rs.size() > 1) {
                    identicalCount++;
                    w.println("WARN_IDENTICAL " + e.getKey() + " all detectors fired " + rs.get(0).driftCount + " times");
                }
                for (RunResult r : rs) {
                    long mon = r.instances - cfg.warmup;
                    if (r.driftCount == 0) {
                        neverFiredCount++;
                        w.println("WARN_NEVER_FIRED " + r.detector + " on " + e.getKey());
                    }
                    if (mon > 0 && r.driftCount > mon / 100) {
                        alwaysFiredCount++;
                        w.printf("WARN_ALWAYS_FIRED %s on %s (%d alarms in %d inst, >1%%)%n",
                                r.detector, e.getKey(), r.driftCount, mon);
                    }
                    if (r.detection.tp > 0 && r.detection.meanDetectionDelay <= 0.01) {
                        suspiciousDelayCount++;
                        w.printf("WARN_ZERO_DELAY %s on %s TP=%d but delay=%.4f (suspicious)%n",
                                r.detector, e.getKey(), r.detection.tp, r.detection.meanDetectionDelay);
                    }
                }
            }
            w.println();
            w.printf("--- TOTALS: identical=%d never_fired=%d always_fired=%d zero_delay=%d ---%n",
                    identicalCount, neverFiredCount, alwaysFiredCount, suspiciousDelayCount);
            w.println();
            int failCount = 0;
            for (RunResult r : all) if ("FAIL".equals(r.status)) failCount++;
            w.printf("--- FAILED RUNS: %d ---%n", failCount);
            w.println();
            Map<String, long[]> diffMatrix = new LinkedHashMap<>();
            for (RunResult r : all) {
                if ("FAIL".equals(r.status)) continue;
                long[] arr = diffMatrix.computeIfAbsent(r.detector, k -> new long[3]);
                arr[0] += r.driftCount;
                arr[1] += r.detection.tp;
                arr[2] += r.detection.fp;
            }
            w.println("Per-detector totals:");
            for (Map.Entry<String, long[]> e : diffMatrix.entrySet()) {
                w.printf("  %-18s drift_total=%d TP_total=%d FP_total=%d%n",
                        e.getKey(), e.getValue()[0], e.getValue()[1], e.getValue()[2]);
            }
        }
    }

    static double mean(double[] x) {
        if (x == null || x.length == 0) return Double.NaN;
        double s = 0; int c = 0;
        for (double v : x) if (Double.isFinite(v)) { s += v; c++; }
        return c == 0 ? Double.NaN : s / c;
    }
    static double std(double[] x) {
        double m = mean(x); if (Double.isNaN(m)) return Double.NaN;
        double s = 0; int c = 0;
        for (double v : x) if (Double.isFinite(v)) { double d = v - m; s += d * d; c++; }
        return c <= 1 ? 0.0 : Math.sqrt(s / (c - 1));
    }
    static String joinIntArr(int[] a) {
        if (a == null || a.length == 0) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++) { if (i > 0) sb.append('|'); sb.append(a[i]); }
        return sb.toString();
    }
}
