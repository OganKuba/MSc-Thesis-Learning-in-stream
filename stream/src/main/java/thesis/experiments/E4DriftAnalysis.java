package thesis.experiments;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.options.OptionHandler;
import moa.streams.ConceptDriftStream;
import moa.streams.InstanceStream;
import moa.streams.generators.SEAGenerator;
import moa.streams.generators.STAGGERGenerator;
import thesis.detection.TwoLevelDriftDetector;
import thesis.evaluation.MetricsCollector;
import thesis.models.DriftAwareSRP;
import thesis.models.FeatureSpace;
import thesis.models.MajorityClassWrapper;
import thesis.models.ModelWrapper;
import thesis.models.NoChangeWrapper;
import thesis.pipeline.SyntheticStreamFactory;
import thesis.selection.FeatureSelector;
import thesis.selection.StaticFeatureSelector;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

public class E4DriftAnalysis {

    public enum Magnitude { LOW, MEDIUM, HIGH }

    public static final class GeneratorSpec {
        public String name;
        public int n = 50_000;
        public int noiseFeatures = 0;
        public int driftFeatures = 5;
        public boolean abrupt;
    }

    public static final class Cfg {
        public String experimentGroup = "E4";
        public String outputDir = "results/E4";
        public int warmup = 1500;
        public int windowSize = 1000;
        public int logEvery = 1000;
        public int ramSampleEvery = 200;
        public int importanceUpdateEvery = 500;
        public int toleranceWindow = 500;
        public long maxInstances = Long.MAX_VALUE;
        public double detectorDelta = 0.002;
        public double f1Threshold = 0.5;
        public double kappaThreshold = 0.5;
        public int methodPeriodicInterval = 500;
        public int methodWPostDrift = 300;
        public double recoveryCapFraction = 0.5;
        public List<GeneratorSpec> generators = new ArrayList<>();
        public List<String> methods = new ArrayList<>();
        public List<String> magnitudes = new ArrayList<>();
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
            c.importanceUpdateEvery = r.path("importance_update_every").asInt(c.importanceUpdateEvery);
            c.toleranceWindow = r.path("tolerance_window").asInt(c.toleranceWindow);
            c.maxInstances    = r.path("max_instances").asLong(c.maxInstances);
            c.detectorDelta   = r.path("detector_delta").asDouble(c.detectorDelta);
            c.f1Threshold     = r.path("f1_threshold").asDouble(c.f1Threshold);
            c.kappaThreshold  = r.path("kappa_threshold").asDouble(c.kappaThreshold);
            c.methodPeriodicInterval = r.path("method_periodic_interval").asInt(c.methodPeriodicInterval);
            c.methodWPostDrift       = r.path("method_w_post_drift").asInt(c.methodWPostDrift);
            c.recoveryCapFraction    = r.path("recovery_cap_fraction").asDouble(c.recoveryCapFraction);
            r.path("seeds").forEach(n -> c.seeds.add(n.asInt()));
            r.path("methods").forEach(n -> c.methods.add(n.asText()));
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
            if (c.methods.isEmpty())    throw new IllegalArgumentException("methods empty");
            if (c.magnitudes.isEmpty()) throw new IllegalArgumentException("magnitudes empty");
            if (c.generators.isEmpty()) throw new IllegalArgumentException("generators empty");
            return c;
        }
    }

    public static final class AlarmEvent {
        public long instance;
        public Set<Integer> detectedFeatures = new TreeSet<>();
        public boolean isTP;
        public int delay = -1;
        public int matchedGT = -1;
    }

    public static final class DetectionStats {
        public long tp, fp, fn;
        public double precision, recall, f1, falseAlarmRate;
        public double meanDetectionDelay = -1.0;
        public List<Integer> delays = new ArrayList<>();
        public void compute(long totalInstances) {
            double p = (tp + fp) == 0 ? 0.0 : (double) tp / (tp + fp);
            double r = (tp + fn) == 0 ? 0.0 : (double) tp / (tp + fn);
            precision = p; recall = r;
            f1 = (p + r == 0.0) ? 0.0 : 2 * p * r / (p + r);
            falseAlarmRate = totalInstances == 0 ? 0.0 : (double) fp / totalInstances;
            if (!delays.isEmpty()) {
                double s = 0; for (int d : delays) s += d;
                meanDetectionDelay = s / delays.size();
            }
        }
    }

    public static final class FeatureDetStats {
        public long tp, fp, fn;
        public double precision, recall, f1;
        public boolean applicable = false;
        public void compute() {
            double p = (tp + fp) == 0 ? 0.0 : (double) tp / (tp + fp);
            double r = (tp + fn) == 0 ? 0.0 : (double) tp / (tp + fn);
            precision = p; recall = r;
            f1 = (p + r == 0.0) ? 0.0 : 2 * p * r / (p + r);
        }
    }

    public static final class RunResult {
        public String generator, method, magnitude;
        public int seed, d;
        public long instances;
        public double accuracy, kappa, kappaPer, recoveryTime, featureStability, ramHours;
        public long driftCount, selectionChangeCount;
        public long totalKept, totalSurgical, totalFull, weightedPredictions, unweightedFallbacks;
        public int[] gtPositions = new int[0];
        public Set<Integer> gtFeatures = new TreeSet<>();
        public List<AlarmEvent> alarms = new ArrayList<>();
        public DetectionStats detection = new DetectionStats();
        public FeatureDetStats featureDetection = new FeatureDetStats();
        public boolean abrupt;
    }

    public static void main(String[] args) throws Exception {
        String configPath = args.length > 0 ? args[0]
                : "stream/configs/e4_synthetic_drift.json";
        Cfg cfg = Cfg.load(Paths.get(configPath));
        new E4DriftAnalysis().run(cfg);
    }

    public void run(Cfg cfg) throws Exception {
        Files.createDirectories(Paths.get(cfg.outputDir));
        Path winCsv = Paths.get(cfg.outputDir, "E4_window.csv");
        Path alrCsv = Paths.get(cfg.outputDir, "E4_alarms.csv");
        Path sumCsv = Paths.get(cfg.outputDir, "E4_summary.csv");
        List<RunResult> all = new ArrayList<>();

        try (PrintWriter w = new PrintWriter(new FileWriter(winCsv.toFile()));
             PrintWriter a = new PrintWriter(new FileWriter(alrCsv.toFile()))) {
            w.println(windowHeader());
            a.println(alarmHeader());
            for (GeneratorSpec g : cfg.generators) {
                for (String mag : cfg.magnitudes) {
                    Magnitude m = Magnitude.valueOf(mag);
                    for (int seed : cfg.seeds) {
                        for (String method : cfg.methods) {
                            try {
                                RunResult r = runOne(cfg, g, m, method, seed, w);
                                writeAlarmsCsv(a, r);
                                all.add(r);
                            } catch (Exception ex) {
                                System.err.printf("[E4][FAIL] %s|%s|%s|seed=%d -> %s%n",
                                        g.name, method, mag, seed, ex);
                                ex.printStackTrace(System.err);
                            }
                        }
                    }
                }
            }
        }

        writeSummary(all, sumCsv);
        writeAggregated(cfg, all);
        writeMinDetectableMagnitude(cfg, all);
        writeRanking(cfg, all);
        System.out.println("[E4] Done -> " + cfg.outputDir);
    }

    public RunResult runOne(Cfg cfg, GeneratorSpec g, Magnitude mag, String methodName,
                            int seed, PrintWriter wcsv) throws Exception {

        InstanceStream stream = buildStreamWithMagnitude(g, mag, seed);
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

        ModelHolder mh = buildMethodModel(cfg, methodName, header, d, C, seed, window, labels);
        ModelWrapper main = mh.main;
        DriftAwareSRP da = mh.da;

        TwoLevelDriftDetector detector = E2AdaptiveFS.buildDetector(
                methodOptDetector(methodName), d, cfg.detectorDelta);

        ModelWrapper majority = new MajorityClassWrapper(mh.baselineSelector, C);
        ModelWrapper noChange = new NoChangeWrapper(mh.baselineSelector, C);
        for (int i = 0; i < collected; i++) {
            majority.train(null, labels[i]);
            noChange.train(null, labels[i]);
        }

        MetricsCollector mc = new MetricsCollector(C, cfg.windowSize, cfg.logEvery, cfg.ramSampleEvery);
        MetricsCollector mM = new MetricsCollector(C, cfg.windowSize, cfg.logEvery, cfg.ramSampleEvery);
        MetricsCollector mN = new MetricsCollector(C, cfg.windowSize, cfg.logEvery, cfg.ramSampleEvery);

        long n = collected;
        long driftCount = 0, selChgCount = 0;
        int[] lastSel = main.getCurrentSelection().clone();
        List<AlarmEvent> alarms = new ArrayList<>();
        int[] gtPositions = groundTruthPositions(g, cfg.warmup);
        Set<Integer> gtFeatures = groundTruthFeatures(g);
        Set<Integer> nextGtSet = new HashSet<>();
        for (int p : gtPositions) nextGtSet.add(p);

        while (stream.hasMoreInstances() && n < cfg.maxInstances) {
            Instance raw = stream.nextInstance().getData();
            int y = (int) raw.classValue();
            double[] x = space.extractFeatures(raw);

            if (mh.scoreUpdater != null) mh.scoreUpdater.accept(x, y);

            long t0 = System.nanoTime();
            int yhat = main.predict(raw);
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
            if (alarm) {
                mc.onDriftAlarm();
                driftCount++;
                AlarmEvent ev = new AlarmEvent();
                ev.instance = n;
                ev.detectedFeatures.addAll(drifting);
                alarms.add(ev);
            }

            if (mh.selector != null) mh.selector.update(x, y, alarm, drifting);
            main.train(raw, y, alarm, drifting);
            majority.train(raw, y);
            noChange.train(raw, y);

            if (mh.importanceUpdater != null && cfg.importanceUpdateEvery > 0
                    && n % cfg.importanceUpdateEvery == 0) {
                double[] pv = detector.getLastPValues();
                mh.importanceUpdater.accept(pv);
            }

            int[] sel = main.getCurrentSelection();
            mc.onSelectionChanged(sel);
            boolean selChanged = !Arrays.equals(sel, lastSel);
            if (selChanged) { selChgCount++; lastSel = sel.clone(); }

            n++;
            if (cfg.logEvery > 0 && n % cfg.logEvery == 0) {
                MetricsCollector.Snapshot s = mc.snapshot();
                int trueDriftHere = 0;
                for (int p : gtPositions) {
                    if (p > n - cfg.logEvery && p <= n) { trueDriftHere = 1; break; }
                }
                wcsv.printf(Locale.ROOT,
                        "%d,%s,%s,%s,%d,%d,%d,\"%s\",\"%s\",%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%.4f,%.6f%n",
                        n, g.name, methodName, mag.name(), seed,
                        trueDriftHere, alarm ? 1 : 0,
                        joinSet(drifting), joinSet(gtFeatures),
                        s.accuracyWindow, s.kappa, s.kappaPer,
                        mM.snapshot().accuracyWindow, mN.snapshot().accuracyWindow,
                        s.lastRecoveryTime, driftCount, selChgCount,
                        Double.isNaN(s.featureStabilityRatio) ? 1.0 : s.featureStabilityRatio,
                        s.ramHoursGB);
                wcsv.flush();
            }
        }

        MetricsCollector.Snapshot s = mc.snapshot();
        RunResult r = new RunResult();
        r.generator = g.name; r.method = methodName; r.magnitude = mag.name();
        r.seed = seed; r.d = d;
        r.instances = n;
        r.accuracy = s.accuracyWindow; r.kappa = s.kappa; r.kappaPer = s.kappaPer;
        r.recoveryTime = Double.isNaN(s.avgRecoveryTime) ? -1.0 : s.avgRecoveryTime;
        long monitoredInstances = n - cfg.warmup;
        if (!g.abrupt && r.recoveryTime > cfg.recoveryCapFraction * monitoredInstances) {
            r.recoveryTime = -1.0;
        }
        r.featureStability = Double.isNaN(s.featureStabilityRatio) ? 1.0 : s.featureStabilityRatio;
        r.ramHours = s.ramHoursGB;
        r.driftCount = driftCount;
        r.selectionChangeCount = selChgCount;
        r.gtPositions = gtPositions;
        r.gtFeatures = gtFeatures;
        r.alarms = alarms;
        r.abrupt = g.abrupt;
        if (da != null) {
            r.totalKept = da.getTotalKept();
            r.totalSurgical = da.getTotalSurgical();
            r.totalFull = da.getTotalFull();
            r.weightedPredictions = da.getWeightedPredictions();
            r.unweightedFallbacks = da.getUnweightedFallbacks();
        }
        r.detection = computeDetection(gtPositions, alarms, cfg.toleranceWindow, g.abrupt, n);
        r.featureDetection = computeFeatureDetection(alarms, gtFeatures, "CustomFeatureDrift".equalsIgnoreCase(g.name));

        System.out.printf(Locale.ROOT,
                "[E4] %-18s %-12s %-6s seed=%d  acc=%.4f k=%.4f  drift=%d  TP=%d FP=%d FN=%d  P=%.3f R=%.3f F1=%.3f delay=%.1f%n",
                g.name, methodName, mag.name(), seed,
                r.accuracy, r.kappa, r.driftCount,
                r.detection.tp, r.detection.fp, r.detection.fn,
                r.detection.precision, r.detection.recall, r.detection.f1,
                r.detection.meanDetectionDelay);
        return r;
    }

    public static final class ModelHolder {
        public ModelWrapper main;
        public DriftAwareSRP da;
        public FeatureSelector selector;
        public StaticFeatureSelector baselineSelector;
        public java.util.function.BiConsumer<double[], Integer> scoreUpdater;
        public java.util.function.Consumer<double[]> importanceUpdater;
    }

    public static ModelHolder buildMethodModel(Cfg cfg, String method, InstancesHeader header,
                                               int d, int C, int seed,
                                               double[][] warmupWin, int[] warmupLab) {

        ModelHolder mh = new ModelHolder();
        StaticFeatureSelector sBase = new StaticFeatureSelector(d, C);
        sBase.initialize(warmupWin, warmupLab);
        mh.baselineSelector = sBase;

        String mu = method.toUpperCase(Locale.ROOT);
        if (mu.equals("HT+S1") || mu.equals("HT_S1")) {
            mh.main = E1Baselines.buildModel("HT", sBase, header, C);
            mh.selector = sBase;
            return mh;
        }
        if (mu.startsWith("ARF+") || mu.startsWith("SRP+")) {
            String[] parts = method.split("\\+");
            String model = parts[0];
            String selName = parts[1];
            E2AdaptiveFS.Variant v = new E2AdaptiveFS.Variant();
            v.name = method; v.model = model; v.selector = selName;
            v.detector = "ADWIN";
            v.periodicInterval = cfg.methodPeriodicInterval;
            v.wPostDrift       = cfg.methodWPostDrift;
            int K = StaticFeatureSelector.defaultK(d);
            FeatureSelector fs = E2AdaptiveFS.buildSelector(v, d, C, K);
            fs.initialize(warmupWin, warmupLab);
            mh.selector = fs;
            mh.main = E2AdaptiveFS.buildModel(model, fs, header);
            return mh;
        }
        if (mu.startsWith("DA-SRP") || mu.startsWith("DASRP")) {
            E3DASRP.Variant ev = new E3DASRP.Variant();
            ev.name = method;
            if (mu.contains("ABC"))      ev.mode = E3DASRP.Mode.DASRP_ABC;
            else if (mu.contains("AB"))  ev.mode = E3DASRP.Mode.DASRP_AB;
            else                          ev.mode = E3DASRP.Mode.DASRP_A;
            ev.tau = 0.5; ev.w1 = 0.7;
            ev.kswinAlpha = 0.005; ev.kswinWindow = 200;
            ev.detector = "ADWIN"; ev.ensembleSize = 10; ev.lambda = 6.0;
            ev.wPostDrift = 1000;
            E3DASRP.ModelHolder em = E3DASRP.buildModel(ev, header, d, C, seed, warmupWin, warmupLab);
            mh.main = em.main;
            mh.da = em.da;
            mh.selector = null;
            mh.baselineSelector = em.selectorForBaselines;
            if (em.scorePid != null && em.scoreRanker != null) {
                mh.scoreUpdater = (x, y) -> {
                    if (allFinite(x)) {
                        em.scorePid.update(x, y);
                        if (em.scorePid.isReady()) em.scoreRanker.update(em.scorePid.discretizeAll(x), y);
                    }
                };
            }
            if (em.importance != null) {
                mh.importanceUpdater = (pv) -> {
                    if (em.scoreRanker == null || em.scorePid == null || !em.scorePid.isReady()) return;
                    double[] mi = em.scoreRanker.getFeatureScores();
                    double[] ks = new double[d];
                    for (int i = 0; i < d; i++) ks[i] = 1.0 - clamp01(pv[i]);
                    if (mi.length == d) try { em.importance.update(mi, ks); } catch (Exception ignore) {}
                };
            }
            return mh;
        }
        throw new IllegalArgumentException("Unknown method: " + method);
    }

    public static InstanceStream buildStreamWithMagnitude(GeneratorSpec g, Magnitude mag, int seed) {
        InstanceStream base;
        String gn = g.name.toUpperCase(Locale.ROOT);
        switch (gn) {
            case "SEA":
                base = buildAbruptSEA(seed, g.n, abruptWidth(mag));
                break;
            case "STAGGER":
                base = buildAbruptSTAGGER(seed, g.n, abruptWidth(mag));
                break;
            case "HYPERPLANE":
                base = SyntheticStreamFactory.createHyperplane(seed, hyperplaneSigma(mag), g.n);
                break;
            case "RANDOMRBF":
                base = SyntheticStreamFactory.createRandomRBF(seed, rbfSpeed(mag), g.n);
                break;
            case "CUSTOMFEATUREDRIFT":
            case "FEATUREDRIFT":
                base = SyntheticStreamFactory.createCustomFeatureDrift(
                        seed, g.driftFeatures, customFeatureSigma(mag), g.n);
                break;
            default: throw new IllegalArgumentException("Unknown generator: " + g.name);
        }
        return g.noiseFeatures > 0
                ? SyntheticStreamFactory.addNoiseFeatures(base, g.noiseFeatures, seed)
                : base;
    }

    static InstanceStream buildAbruptSEA(int seed, int n, int width) {
        int p1 = Math.max(1, n / 4);
        int p2 = Math.max(p1 + 1, n / 2);
        int p3 = Math.max(p2 + 1, (3 * n) / 4);
        SEAGenerator g1 = newSEA(seed,     1);
        SEAGenerator g2 = newSEA(seed + 1, 2);
        SEAGenerator g3 = newSEA(seed + 2, 3);
        SEAGenerator g4 = newSEA(seed + 3, 4);
        ConceptDriftStream s34   = newDrift(g3, g4, p3, width, seed);
        ConceptDriftStream s234  = newDrift(g2, s34, p2, width, seed + 10);
        ConceptDriftStream s1234 = newDrift(g1, s234, p1, width, seed + 20);
        s1234.prepareForUse();
        return limit(s1234, n);
    }

    static InstanceStream buildAbruptSTAGGER(int seed, int n, int width) {
        int p1 = Math.max(1, n / 5);
        int p2 = Math.max(p1 + 1, (2 * n) / 5);
        int p3 = Math.max(p2 + 1, (3 * n) / 5);
        STAGGERGenerator s1  = newStagger(seed,     1);
        STAGGERGenerator s2  = newStagger(seed + 1, 2);
        STAGGERGenerator s3  = newStagger(seed + 2, 3);
        STAGGERGenerator s1b = newStagger(seed + 3, 1);
        ConceptDriftStream d3 = newDrift(s3, s1b, p3, width, seed + 30);
        ConceptDriftStream d2 = newDrift(s2, d3,  p2, width, seed + 40);
        ConceptDriftStream d1 = newDrift(s1, d2,  p1, width, seed + 50);
        d1.prepareForUse();
        return limit(d1, n);
    }

    private static SEAGenerator newSEA(int seed, int function) {
        SEAGenerator g = new SEAGenerator();
        g.instanceRandomSeedOption.setValue(seed);
        g.functionOption.setValue(function);
        g.balanceClassesOption.setValue(false);
        g.noisePercentageOption.setValue(10);
        g.prepareForUse();
        return g;
    }
    private static STAGGERGenerator newStagger(int seed, int function) {
        STAGGERGenerator s = new STAGGERGenerator();
        s.instanceRandomSeedOption.setValue(seed);
        s.functionOption.setValue(function);
        s.prepareForUse();
        return s;
    }
    private static ConceptDriftStream newDrift(InstanceStream a, InstanceStream b,
                                               int position, int width, int seed) {
        ConceptDriftStream d = new ConceptDriftStream();
        d.streamOption.setCurrentObject(a);
        d.driftstreamOption.setCurrentObject(b);
        d.positionOption.setValue(position);
        d.widthOption.setValue(Math.max(1, width));
        d.randomSeedOption.setValue(seed);
        return d;
    }
    private static InstanceStream limit(InstanceStream s, int n) {
        try {
            java.lang.reflect.Method mm = SyntheticStreamFactory.class
                    .getDeclaredMethod("limit", InstanceStream.class, int.class);
            mm.setAccessible(true);
            return (InstanceStream) mm.invoke(null, s, n);
        } catch (Exception e) {
            return s;
        }
    }

    static int abruptWidth(Magnitude m) {
        switch (m) { case LOW: return 500; case MEDIUM: return 100; default: return 1; }
    }
    static double hyperplaneSigma(Magnitude m) {
        switch (m) { case LOW: return 0.001; case MEDIUM: return 0.01; default: return 0.1; }
    }
    static double rbfSpeed(Magnitude m) {
        switch (m) { case LOW: return 0.0001; case MEDIUM: return 0.001; default: return 0.01; }
    }
    static double customFeatureSigma(Magnitude m) {
        switch (m) { case LOW: return 0.02; case MEDIUM: return 0.05; default: return 0.10; }
    }

    static boolean isAbrupt(String generator) {
        String g = generator.toUpperCase(Locale.ROOT);
        return g.equals("SEA") || g.equals("STAGGER");
    }

    static int[] groundTruthPositions(GeneratorSpec g, int warmup) {
        String gn = g.name.toUpperCase(Locale.ROOT);
        switch (gn) {
            case "SEA":     return new int[] { g.n / 4, g.n / 2, (3 * g.n) / 4 };
            case "STAGGER": return new int[] { g.n / 5, (2 * g.n) / 5, (3 * g.n) / 5 };
            default:        return new int[] { warmup };
        }
    }

    static Set<Integer> groundTruthFeatures(GeneratorSpec g) {
        Set<Integer> s = new TreeSet<>();
        if ("CustomFeatureDrift".equalsIgnoreCase(g.name) || "FeatureDrift".equalsIgnoreCase(g.name)) {
            for (int i = 0; i < g.driftFeatures; i++) s.add(i);
        }
        return s;
    }

    static String methodOptDetector(String method) {
        return "ADWIN";
    }

    static DetectionStats computeDetection(int[] gt, List<AlarmEvent> alarms,
                                           int tolerance, boolean abrupt, long total) {
        DetectionStats st = new DetectionStats();
        if (alarms == null) alarms = new ArrayList<>();
        List<AlarmEvent> sorted = new ArrayList<>(alarms);
        sorted.sort(Comparator.comparingLong(a -> a.instance));
        if (abrupt) {
            boolean[] gtMatched = new boolean[gt.length];
            for (AlarmEvent a : sorted) {
                int bestGt = -1, bestDist = Integer.MAX_VALUE;
                for (int i = 0; i < gt.length; i++) {
                    if (gtMatched[i]) continue;
                    int d = (int) Math.abs(a.instance - gt[i]);
                    if (d <= tolerance && d < bestDist) { bestDist = d; bestGt = i; }
                }
                if (bestGt >= 0) {
                    gtMatched[bestGt] = true;
                    a.isTP = true; a.delay = bestDist; a.matchedGT = bestGt;
                    st.tp++; st.delays.add(bestDist);
                } else {
                    a.isTP = false; a.delay = -1;
                    st.fp++;
                }
            }
            for (boolean b : gtMatched) if (!b) st.fn++;
        } else {
            long onset = gt.length > 0 ? gt[0] : 0;
            AlarmEvent first = null;
            for (AlarmEvent a : sorted) {
                if (a.instance >= onset) { first = a; break; }
            }
            for (AlarmEvent a : sorted) {
                if (a == first) {
                    a.isTP = true;
                    a.delay = (int) (a.instance - onset);
                    a.matchedGT = 0;
                    st.tp++; st.delays.add(a.delay);
                } else {
                    a.isTP = false; a.delay = -1;
                    st.fp++;
                }
            }
            if (first == null) st.fn++;
        }
        st.compute(total);
        return st;
    }

    static FeatureDetStats computeFeatureDetection(List<AlarmEvent> alarms,
                                                   Set<Integer> gtFeatures,
                                                   boolean isFeatureDrift) {
        FeatureDetStats st = new FeatureDetStats();
        st.applicable = isFeatureDrift && !gtFeatures.isEmpty();
        if (!st.applicable) return st;
        for (AlarmEvent a : alarms) {
            if (!a.isTP) continue;
            for (int f : a.detectedFeatures) {
                if (gtFeatures.contains(f)) st.tp++; else st.fp++;
            }
            for (int f : gtFeatures) {
                if (!a.detectedFeatures.contains(f)) st.fn++;
            }
        }
        st.compute();
        return st;
    }

    static String windowHeader() {
        return "instance_num,generator,method,magnitude,seed,"
                + "true_drift,detected_drift,drifting_features_detected,drifting_features_true,"
                + "accuracy_window,kappa_window,kappa_per_window,"
                + "majority_baseline_window,nochange_baseline_window,"
                + "recovery_time,drift_count,selection_change_count,"
                + "feature_stability_ratio,ram_hours";
    }

    static String alarmHeader() {
        return "generator,method,magnitude,seed,instance_num,is_tp,delay,matched_gt,"
                + "drifting_features_detected";
    }

    static void writeAlarmsCsv(PrintWriter w, RunResult r) {
        for (AlarmEvent a : r.alarms) {
            w.printf(Locale.ROOT, "%s,%s,%s,%d,%d,%s,%d,%d,\"%s\"%n",
                    r.generator, r.method, r.magnitude, r.seed,
                    a.instance, a.isTP ? "TP" : "FP", a.delay, a.matchedGT,
                    joinSet(a.detectedFeatures));
        }
        w.flush();
    }

    static void writeSummary(List<RunResult> all, Path path) throws Exception {
        try (PrintWriter w = new PrintWriter(new FileWriter(path.toFile()))) {
            w.println("generator,method,magnitude,seed,instances,d,"
                    + "accuracy,kappa,kappa_per,recovery_time,feature_stability,ram_hours,"
                    + "drift_count,selection_change_count,"
                    + "tp,fp,fn,precision,recall,f1,false_alarm_rate,mean_detection_delay,"
                    + "feat_tp,feat_fp,feat_fn,feat_precision,feat_recall,feat_f1,"
                    + "total_kept,total_surgical,total_full,weighted_predictions,unweighted_fallbacks,"
                    + "gt_positions");
            for (RunResult r : all) {
                w.printf(Locale.ROOT,
                        "%s,%s,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%d,%d,"
                                + "%d,%d,%d,%.4f,%.4f,%.4f,%.6f,%.2f,"
                                + "%d,%d,%d,%.4f,%.4f,%.4f,"
                                + "%d,%d,%d,%d,%d,\"%s\"%n",
                        r.generator, r.method, r.magnitude, r.seed, r.instances, r.d,
                        r.accuracy, r.kappa, r.kappaPer, r.recoveryTime,
                        r.featureStability, r.ramHours, r.driftCount, r.selectionChangeCount,
                        r.detection.tp, r.detection.fp, r.detection.fn,
                        r.detection.precision, r.detection.recall, r.detection.f1,
                        r.detection.falseAlarmRate, r.detection.meanDetectionDelay,
                        r.featureDetection.tp, r.featureDetection.fp, r.featureDetection.fn,
                        r.featureDetection.precision, r.featureDetection.recall, r.featureDetection.f1,
                        r.totalKept, r.totalSurgical, r.totalFull,
                        r.weightedPredictions, r.unweightedFallbacks,
                        joinIntArr(r.gtPositions));
            }
        }
    }

    static void writeAggregated(Cfg cfg, List<RunResult> all) throws Exception {
        Path p = Paths.get(cfg.outputDir, "E4_aggregated.csv");
        Map<String, List<RunResult>> grp = new LinkedHashMap<>();
        for (RunResult r : all) {
            String key = r.generator + "|" + r.method + "|" + r.magnitude;
            grp.computeIfAbsent(key, k -> new ArrayList<>()).add(r);
        }
        try (PrintWriter w = new PrintWriter(new FileWriter(p.toFile()))) {
            w.println("generator,method,magnitude,n_seeds,"
                    + "mean_accuracy,std_accuracy,mean_kappa,std_kappa,"
                    + "mean_recovery,std_recovery,mean_f1,std_f1,"
                    + "mean_precision,mean_recall,mean_false_alarm_rate,mean_detection_delay,"
                    + "mean_feat_f1,mean_feat_precision,mean_feat_recall");
            for (Map.Entry<String, List<RunResult>> e : grp.entrySet()) {
                List<RunResult> rs = e.getValue();
                String[] keys = e.getKey().split("\\|");
                double[] acc = rs.stream().mapToDouble(r -> r.accuracy).toArray();
                double[] kap = rs.stream().mapToDouble(r -> r.kappa).toArray();
                double[] rec = rs.stream().mapToDouble(r -> r.recoveryTime).filter(v -> v >= 0).toArray();
                double[] f1  = rs.stream().mapToDouble(r -> r.detection.f1).toArray();
                double[] pr  = rs.stream().mapToDouble(r -> r.detection.precision).toArray();
                double[] re  = rs.stream().mapToDouble(r -> r.detection.recall).toArray();
                double[] far = rs.stream().mapToDouble(r -> r.detection.falseAlarmRate).toArray();
                double[] del = rs.stream().mapToDouble(r -> r.detection.meanDetectionDelay).toArray();
                double[] ff1 = rs.stream().mapToDouble(r -> r.featureDetection.f1).toArray();
                double[] fpr = rs.stream().mapToDouble(r -> r.featureDetection.precision).toArray();
                double[] fre = rs.stream().mapToDouble(r -> r.featureDetection.recall).toArray();
                w.printf(Locale.ROOT,
                        "%s,%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%.2f,%.4f,%.4f,%.4f%n",
                        keys[0], keys[1], keys[2], rs.size(),
                        mean(acc), std(acc), mean(kap), std(kap),
                        mean(rec), std(rec), mean(f1), std(f1),
                        mean(pr), mean(re), mean(far), mean(del),
                        mean(ff1), mean(fpr), mean(fre));
            }
        }
    }

    static void writeMinDetectableMagnitude(Cfg cfg, List<RunResult> all) throws Exception {
        Path p = Paths.get(cfg.outputDir, "E4_min_detectable_magnitude.csv");
        Map<String, Map<String, double[]>> agg = new LinkedHashMap<>();
        Map<String, Map<String, Double>> recAgg = new LinkedHashMap<>();
        for (RunResult r : all) {
            String key = r.method + "|" + r.generator;
            agg.computeIfAbsent(key, k -> new LinkedHashMap<>())
                    .computeIfAbsent(r.magnitude, k -> new double[2]);
            double[] acc = agg.get(key).get(r.magnitude);
            acc[0] += r.detection.f1;
            acc[1] += r.kappa;
            recAgg.computeIfAbsent(key, k -> new LinkedHashMap<>())
                    .merge(r.magnitude, r.recoveryTime, Double::sum);
        }
        Map<String, Map<String, Integer>> counts = new LinkedHashMap<>();
        for (RunResult r : all) {
            String key = r.method + "|" + r.generator;
            counts.computeIfAbsent(key, k -> new LinkedHashMap<>()).merge(r.magnitude, 1, Integer::sum);
        }
        try (PrintWriter w = new PrintWriter(new FileWriter(p.toFile()))) {
            w.println("method,generator,minimum_detectable_magnitude,mean_F1_at_min,mean_kappa_at_min,mean_recovery_at_min");
            String[] order = { "LOW", "MEDIUM", "HIGH" };
            for (Map.Entry<String, Map<String, double[]>> e : agg.entrySet()) {
                String[] keys = e.getKey().split("\\|");
                String foundMag = "none";
                double foundF1 = 0, foundKap = 0, foundRec = -1;
                for (String mag : order) {
                    double[] sums = e.getValue().get(mag);
                    if (sums == null) continue;
                    int n = counts.get(e.getKey()).getOrDefault(mag, 0);
                    if (n == 0) continue;
                    double mf1 = sums[0] / n;
                    double mka = sums[1] / n;
                    if (mf1 >= cfg.f1Threshold && mka >= cfg.kappaThreshold) {
                        foundMag = mag; foundF1 = mf1; foundKap = mka;
                        foundRec = recAgg.get(e.getKey()).getOrDefault(mag, 0.0) / n;
                        break;
                    }
                }
                w.printf(Locale.ROOT, "%s,%s,%s,%.4f,%.4f,%.4f%n",
                        keys[0], keys[1], foundMag, foundF1, foundKap, foundRec);
            }
        }
    }

    static void writeRanking(Cfg cfg, List<RunResult> all) throws Exception {
        Path p = Paths.get(cfg.outputDir, "E4_ranking.csv");
        Map<String, double[]> mAgg = new LinkedHashMap<>();
        Map<String, Integer> nAgg = new LinkedHashMap<>();
        for (RunResult r : all) {
            double[] a = mAgg.computeIfAbsent(r.method, k -> new double[4]);
            a[0] += r.kappa;
            a[1] += r.detection.f1;
            a[2] += r.recoveryTime < 0 ? 0 : r.recoveryTime;
            a[3] += r.featureStability;
            nAgg.merge(r.method, 1, Integer::sum);
        }
        List<Map.Entry<String, double[]>> list = new ArrayList<>(mAgg.entrySet());
        list.sort((x, y) -> Double.compare(y.getValue()[0] / nAgg.get(y.getKey()),
                x.getValue()[0] / nAgg.get(x.getKey())));
        try (PrintWriter w = new PrintWriter(new FileWriter(p.toFile()))) {
            w.println("rank,method,n_runs,mean_kappa,mean_f1,mean_recovery,mean_feature_stability");
            int rank = 1;
            for (Map.Entry<String, double[]> e : list) {
                int n = nAgg.get(e.getKey());
                double[] a = e.getValue();
                w.printf(Locale.ROOT, "%d,%s,%d,%.4f,%.4f,%.4f,%.4f%n",
                        rank++, e.getKey(), n, a[0]/n, a[1]/n, a[2]/n, a[3]/n);
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
        for (int i = 0; i < a.length; i++) { if (i>0) sb.append('|'); sb.append(a[i]); }
        return sb.toString();
    }
    static String joinSet(Set<Integer> s) {
        if (s == null || s.isEmpty()) return "";
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for (int v : s) { if (!first) sb.append('|'); sb.append(v); first = false; }
        return sb.toString();
    }
    static boolean allFinite(double[] r) {
        for (double v : r) if (!Double.isFinite(v)) return false;
        return true;
    }
    static double clamp01(double v) {
        if (Double.isNaN(v) || v < 0) return 0; if (v > 1) return 1; return v;
    }
}
