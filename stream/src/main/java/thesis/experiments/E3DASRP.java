package thesis.experiments;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.options.OptionHandler;
import moa.streams.InstanceStream;
import thesis.detection.TwoLevelDriftDetector;
import thesis.discretization.PiDDiscretizer;
import thesis.evaluation.MetricsCollector;
import thesis.evaluation.StatisticalTests;
import thesis.experiments.E2AdaptiveFS.DatasetSpec;
import thesis.models.ARFWrapper;
import thesis.models.DriftActionSummary;
import thesis.models.DriftAwareSRP;
import thesis.models.FeatureImportance;
import thesis.models.FeatureSpace;
import thesis.models.MajorityClassWrapper;
import thesis.models.ModelWrapper;
import thesis.models.NoChangeWrapper;
import thesis.models.SRPWrapper;
import thesis.selection.AlarmTriggeredSelector;
import thesis.selection.FilterRanker;
import thesis.selection.InformationGainRanker;
import thesis.selection.StaticFeatureSelector;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public class E3DASRP {

    public enum Mode { BASELINE_SRP_S1, BASELINE_ARF_S2, DASRP_A, DASRP_AB, DASRP_ABC }

    public static final class Variant {
        public String name;
        public Mode mode;
        public double tau = 0.5;
        public double w1 = 0.7;
        public double kswinAlpha = 0.005;
        public int kswinWindow = 200;
        public String detector = "ADWIN";
        public int ensembleSize = 10;
        public double lambda = 6.0;
        public int wPostDrift = 1000;
    }

    public static final class Cfg {
        public String experimentGroup = "E3";
        public String outputDir = "results/E3";
        public int warmup = 1500;
        public int windowSize = 1000;
        public int logEvery = 1000;
        public int ramSampleEvery = 200;
        public int importanceUpdateEvery = 1000;
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
            c.importanceUpdateEvery = r.path("importance_update_every").asInt(c.importanceUpdateEvery);
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
            for (JsonNode v : r.path("variants")) c.variants.add(parseVariant(v));
            JsonNode grid = r.path("sensitivity_grid");
            if (!grid.isMissingNode() && !grid.isNull()) {
                c.variants.addAll(expandGrid(grid));
            }
            if (c.seeds.isEmpty())    throw new IllegalArgumentException("seeds empty");
            if (c.variants.isEmpty()) throw new IllegalArgumentException("variants empty");
            if (c.datasets.isEmpty()) throw new IllegalArgumentException("datasets empty");
            return c;
        }

        static Variant parseVariant(JsonNode v) {
            Variant vv = new Variant();
            vv.name         = v.path("name").asText();
            vv.mode         = Mode.valueOf(v.path("mode").asText());
            vv.tau          = v.path("tau").asDouble(vv.tau);
            vv.w1           = v.path("w1").asDouble(vv.w1);
            vv.kswinAlpha   = v.path("kswin_alpha").asDouble(vv.kswinAlpha);
            vv.kswinWindow  = v.path("kswin_window").asInt(vv.kswinWindow);
            vv.detector     = v.path("detector").asText(vv.detector);
            vv.ensembleSize = v.path("ensemble_size").asInt(vv.ensembleSize);
            vv.lambda       = v.path("lambda").asDouble(vv.lambda);
            vv.wPostDrift   = v.path("w_post_drift").asInt(vv.wPostDrift);
            if (vv.name == null || vv.name.isEmpty()) throw new IllegalArgumentException("variant.name empty");
            return vv;
        }

        static List<Variant> expandGrid(JsonNode g) {
            String baseMode = g.path("base_mode").asText("DASRP_ABC");
            String baseName = g.path("base_name").asText("DA-SRP-ABC");
            List<Double> w1s = new ArrayList<>();
            List<Double> alphas = new ArrayList<>();
            List<Integer> wins = new ArrayList<>();
            g.path("w1").forEach(x -> w1s.add(x.asDouble()));
            g.path("kswin_alpha").forEach(x -> alphas.add(x.asDouble()));
            g.path("kswin_window").forEach(x -> wins.add(x.asInt()));
            List<Variant> out = new ArrayList<>();
            for (double w : w1s) for (double a : alphas) for (int wn : wins) {
                Variant v = new Variant();
                v.name = String.format(Locale.ROOT, "%s_w1=%.2f_a=%.3f_W=%d", baseName, w, a, wn);
                v.mode = Mode.valueOf(baseMode);
                v.w1 = w; v.kswinAlpha = a; v.kswinWindow = wn;
                out.add(v);
            }
            return out;
        }
    }

    public static final class RunResult {
        public String dataset, variant;
        public int seed, d;
        public long instances;
        public double accuracy, kappa, kappaPer, recoveryTime, featureStability, ramHours;
        public long driftCount;
        public long totalKept, totalSurgical, totalFull, totalNoReplacement, refreshCalls;
        public long weightedPredictions, unweightedFallbacks;
        public double[] finalImportanceTopK;
        public double[] finalLearnerWeights;
    }

    public static void main(String[] args) throws Exception {
        String configPath = args.length > 0 ? args[0] : "stream/configs/e3_da_srp.json";
        Cfg cfg = Cfg.load(Paths.get(configPath));
        new E3DASRP().run(cfg);
    }

    public void run(Cfg cfg) throws Exception {
        Files.createDirectories(Paths.get(cfg.outputDir));
        Path winCsv = Paths.get(cfg.outputDir, "E3_window.csv");
        Path summaryCsv = Paths.get(cfg.outputDir, "E3_summary.csv");
        List<RunResult> all = new ArrayList<>();

        try (PrintWriter csv = new PrintWriter(new FileWriter(winCsv.toFile()))) {
            csv.println(windowHeader());
            for (DatasetSpec ds : cfg.datasets) {
                if ("arff".equalsIgnoreCase(ds.type)
                        && (ds.path == null || !Files.exists(Paths.get(ds.path)))) {
                    System.err.printf("[E3] skip arff %s%n", ds.name);
                    if (cfg.skipMissingArff) continue;
                    throw new java.io.FileNotFoundException(String.valueOf(ds.path));
                }
                for (int seed : cfg.seeds) {
                    for (Variant v : cfg.variants) {
                        try {
                            all.add(runOne(cfg, ds, v, seed, csv));
                        } catch (Exception e) {
                            System.err.printf("[E3][FAIL] %s|%s|seed=%d -> %s%n",
                                    ds.name, v.name, seed, e);
                            e.printStackTrace(System.err);
                        }
                    }
                }
            }
        }

        writeSummary(all, summaryCsv);
        runStatistics(cfg, all);
        System.out.println("[E3] Done -> " + cfg.outputDir);
    }

    public static String windowHeader() {
        return "instance_num,dataset,variant,seed,selected_count,selected_features,drifting_features,"
                + "overlap_per_learner,num_surgical_updates,num_full_replacements,num_kept,num_no_replacement,"
                + "importance_top5,learner_weights,accuracy_window,kappa_window,kappa_per_window,"
                + "majority_baseline_window,nochange_baseline_window,recovery_time,drift_count,"
                + "feature_stability_ratio,ram_hours";
    }

    public RunResult runOne(Cfg cfg, DatasetSpec ds, Variant v, int seed, PrintWriter csv) throws Exception {
        InstanceStream stream = E2AdaptiveFS.buildStream(ds, seed);
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

        ModelHolder mh = buildModel(v, header, d, C, seed, window, labels);
        ModelWrapper mainModel = mh.main;
        DriftAwareSRP da = mh.da;

        TwoLevelDriftDetector detector = buildDetectorWithKswin(v, d);

        ModelWrapper majority = new MajorityClassWrapper(mh.selectorForBaselines, C);
        ModelWrapper noChange = new NoChangeWrapper(mh.selectorForBaselines, C);
        for (int i = 0; i < collected; i++) {
            majority.train(null, labels[i]);
            noChange.train(null, labels[i]);
        }

        MetricsCollector mc = new MetricsCollector(C, cfg.windowSize, cfg.logEvery, cfg.ramSampleEvery);
        MetricsCollector mM = new MetricsCollector(C, cfg.windowSize, cfg.logEvery, cfg.ramSampleEvery);
        MetricsCollector mN = new MetricsCollector(C, cfg.windowSize, cfg.logEvery, cfg.ramSampleEvery);

        long n = collected;
        long driftCount = 0;
        long lastSurgical = 0, lastFull = 0;
        Set<Integer> lastDrifting = Set.of();
        int[] lastSel = mainModel.getCurrentSelection().clone();

        while (stream.hasMoreInstances() && n < cfg.maxInstances) {
            Instance raw = stream.nextInstance().getData();
            int y = (int) raw.classValue();
            double[] x = space.extractFeatures(raw);

            if (mh.scorePid != null && allFinite(x)) {
                mh.scorePid.update(x, y);
                if (mh.scorePid.isReady() && mh.scoreRanker != null) {
                    mh.scoreRanker.update(mh.scorePid.discretizeAll(x), y);
                }
            }

            long t0 = System.nanoTime();
            int yhat = mainModel.predict(raw);
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
                lastDrifting = drifting;
            }

            mainModel.train(raw, y, alarm, drifting);
            majority.train(raw, y);
            noChange.train(raw, y);

            if (mh.importance != null && cfg.importanceUpdateEvery > 0
                    && n % cfg.importanceUpdateEvery == 0
                    && mh.scoreRanker != null && mh.scorePid != null && mh.scorePid.isReady()) {
                double[] mi = mh.scoreRanker.getFeatureScores();
                double[] pv = detector.getLastPValues();
                double[] ks = new double[d];
                for (int i = 0; i < d; i++) ks[i] = 1.0 - clamp01(pv[i]);
                if (mi.length == d && ks.length == d) {
                    try { mh.importance.update(mi, ks); } catch (Exception ignore) {}
                }
            }

            int[] sel = mainModel.getCurrentSelection();
            mc.onSelectionChanged(sel);
            if (!Arrays.equals(sel, lastSel)) lastSel = sel.clone();

            n++;
            if (cfg.logEvery > 0 && n % cfg.logEvery == 0) {
                MetricsCollector.Snapshot s = mc.snapshot();
                long surgNow = (da == null) ? 0 : da.getTotalSurgical();
                long fullNow = (da == null) ? 0 : da.getTotalFull();
                long surgDelta = surgNow - lastSurgical;
                long fullDelta = fullNow - lastFull;
                lastSurgical = surgNow; lastFull = fullNow;

                String overlapStr = "";
                if (da != null && da.getLastSummary() != null) {
                    overlapStr = joinIntArr(da.getLastSummary().getOverlapCounts());
                }
                String impStr = "";
                if (da != null && da.getImportance() != null) {
                    impStr = topKImportance(da.getImportance().getImportance(), 5);
                }
                String lwStr = "";
                if (da != null) lwStr = joinDoubleArr(da.getLastLearnerWeights());

                csv.printf(Locale.ROOT,
                        "%d,%s,%s,%d,%d,\"%s\",\"%s\",\"%s\",%d,%d,%d,%d,\"%s\",\"%s\",%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%.4f,%.6f%n",
                        n, ds.name, v.name, seed,
                        sel.length, joinIntArr(sel),
                        joinSet(lastDrifting),
                        overlapStr,
                        surgDelta, fullDelta,
                        da == null ? 0L : da.getTotalKept(),
                        da == null ? 0L : da.getTotalNoReplacement(),
                        impStr, lwStr,
                        s.accuracyWindow, s.kappa, s.kappaPer,
                        mM.snapshot().accuracyWindow, mN.snapshot().accuracyWindow,
                        s.lastRecoveryTime, driftCount,
                        Double.isNaN(s.featureStabilityRatio) ? 1.0 : s.featureStabilityRatio,
                        s.ramHoursGB);
                csv.flush();
            }
        }

        MetricsCollector.Snapshot s = mc.snapshot();
        RunResult r = new RunResult();
        r.dataset = ds.name; r.variant = v.name; r.seed = seed; r.d = d;
        r.instances = n;
        r.accuracy = s.accuracyWindow; r.kappa = s.kappa; r.kappaPer = s.kappaPer;
        r.recoveryTime = s.avgRecoveryTime;
        r.featureStability = Double.isNaN(s.featureStabilityRatio) ? 1.0 : s.featureStabilityRatio;
        r.ramHours = s.ramHoursGB;
        r.driftCount = driftCount;
        if (da != null) {
            r.totalKept = da.getTotalKept();
            r.totalSurgical = da.getTotalSurgical();
            r.totalFull = da.getTotalFull();
            r.totalNoReplacement = da.getTotalNoReplacement();
            r.refreshCalls = da.getRefreshCalls();
            r.weightedPredictions = da.getWeightedPredictions();
            r.unweightedFallbacks = da.getUnweightedFallbacks();
            if (da.getImportance() != null)
                r.finalImportanceTopK = topKDoubles(da.getImportance().getImportance(), 5);
            r.finalLearnerWeights = da.getLastLearnerWeights();
        }

        System.out.printf(Locale.ROOT,
                "[E3] %-14s %-22s seed=%d  acc=%.4f k=%.4f kPer=%.4f  drift=%d kept=%d surg=%d full=%d noR=%d wPred=%d%n",
                ds.name, v.name, seed,
                r.accuracy, r.kappa, r.kappaPer,
                r.driftCount, r.totalKept, r.totalSurgical, r.totalFull, r.totalNoReplacement,
                r.weightedPredictions);
        return r;
    }

    public static final class ModelHolder {
        public ModelWrapper main;
        public DriftAwareSRP da;
        public FeatureImportance importance;
        public PiDDiscretizer scorePid;
        public FilterRanker scoreRanker;
        public StaticFeatureSelector selectorForBaselines;
    }

    public static ModelHolder buildModel(Variant v, InstancesHeader header, int d, int C,
                                         int seed, double[][] warmupWindow, int[] warmupLabels) {
        ModelHolder mh = new ModelHolder();
        StaticFeatureSelector sBase = new StaticFeatureSelector(d, C);
        sBase.initialize(warmupWindow, warmupLabels);
        mh.selectorForBaselines = sBase;

        switch (v.mode) {
            case BASELINE_SRP_S1: {
                mh.main = new SRPWrapper(sBase, header, v.ensembleSize, v.lambda, false, true);
                return mh;
            }
            case BASELINE_ARF_S2: {
                AlarmTriggeredSelector s2 = new AlarmTriggeredSelector(
                        d, C,
                        StaticFeatureSelector.defaultK(d),
                        Math.max(50, v.wPostDrift),
                        new PiDDiscretizer(d, C),
                        (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc));
                s2.initialize(warmupWindow, warmupLabels);
                mh.main = new ARFWrapper(s2, header, v.ensembleSize, v.lambda, false, true);
                return mh;
            }
            default:
                break;
        }

        StaticFeatureSelector wideSel = new StaticFeatureSelector(d, C, d,
                new PiDDiscretizer(d, C),
                (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc));
        wideSel.initializeIdentity();
        SRPWrapper srp = new SRPWrapper(wideSel, header, v.ensembleSize, v.lambda, false, false);

        mh.scorePid = new PiDDiscretizer(d, C);
        for (int i = 0; i < warmupWindow.length; i++) {
            if (allFinite(warmupWindow[i])) mh.scorePid.update(warmupWindow[i], warmupLabels[i]);
        }
        if (mh.scorePid.isReady()) {
            mh.scoreRanker = new InformationGainRanker(d, mh.scorePid.getB2(), C);
            for (int i = 0; i < warmupWindow.length; i++) {
                if (allFinite(warmupWindow[i])) {
                    mh.scoreRanker.update(mh.scorePid.discretizeAll(warmupWindow[i]), warmupLabels[i]);
                }
            }
        }

        FeatureImportance imp = null;
        if (v.mode == Mode.DASRP_AB || v.mode == Mode.DASRP_ABC) {
            double w1 = v.w1; double w2 = Math.max(1e-6, 1.0 - v.w1);
            imp = new FeatureImportance(d, w1, w2, 1e-6, true);
            if (mh.scoreRanker != null) {
                double[] mi = mh.scoreRanker.getFeatureScores();
                double[] ks = new double[d];
                try { imp.update(mi, ks); } catch (Exception ignore) {}
            }
        }
        mh.importance = imp;

        DriftAwareSRP da;
        if (v.mode == Mode.DASRP_AB) {
            da = new DASRPNoWeighting(srp, v.tau, seed, imp);
        } else if (v.mode == Mode.DASRP_ABC) {
            da = new DriftAwareSRP(srp, v.tau, seed, imp);
        } else {
            da = new DriftAwareSRP(srp, v.tau, seed, null);
        }
        final FilterRanker rankerRef = mh.scoreRanker;
        da.setScoreProvider(() -> {
            if (rankerRef == null) return null;
            double[] sc = rankerRef.getFeatureScores();
            return (sc != null && sc.length == d) ? sc : null;
        });

        mh.main = da;
        mh.da = da;
        return mh;
    }

    public static TwoLevelDriftDetector buildDetectorWithKswin(Variant v, int d) {
        TwoLevelDriftDetector.Config c = new TwoLevelDriftDetector.Config(d);
        c.level1Delta = 0.002;
        switch (v.detector.toUpperCase(Locale.ROOT)) {
            case "ADWIN":  c.level1Type = TwoLevelDriftDetector.Level1Type.ADWIN;  break;
            case "HDDM_A": c.level1Type = TwoLevelDriftDetector.Level1Type.HDDM_A; break;
            case "HDDM_W": c.level1Type = TwoLevelDriftDetector.Level1Type.HDDM_W; break;
            default: throw new IllegalArgumentException("Unknown detector: " + v.detector);
        }
        c.kswinAlpha = v.kswinAlpha;
        c.kswinWindowSize = Math.max(10, v.kswinWindow);
        return new TwoLevelDriftDetector(c);
    }

    static boolean allFinite(double[] r) {
        for (double v : r) if (!Double.isFinite(v)) return false;
        return true;
    }
    static double clamp01(double v) {
        if (Double.isNaN(v) || v < 0) return 0; if (v > 1) return 1; return v;
    }
    static String joinIntArr(int[] a) {
        if (a == null || a.length == 0) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++) { if (i>0) sb.append('|'); sb.append(a[i]); }
        return sb.toString();
    }
    static String joinDoubleArr(double[] a) {
        if (a == null || a.length == 0) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++) {
            if (i>0) sb.append('|');
            sb.append(String.format(Locale.ROOT, "%.4f", a[i]));
        }
        return sb.toString();
    }
    static String joinSet(Set<Integer> s) {
        if (s == null || s.isEmpty()) return "";
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for (int v : s) { if (!first) sb.append('|'); sb.append(v); first = false; }
        return sb.toString();
    }
    static String topKImportance(double[] imp, int k) {
        if (imp == null) return "";
        Integer[] idx = new Integer[imp.length];
        for (int i = 0; i < imp.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Double.compare(imp[b], imp[a]));
        int kk = Math.min(k, imp.length);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < kk; i++) {
            if (i>0) sb.append('|');
            sb.append(idx[i]).append(":").append(String.format(Locale.ROOT, "%.3f", imp[idx[i]]));
        }
        return sb.toString();
    }
    static double[] topKDoubles(double[] imp, int k) {
        if (imp == null) return new double[0];
        Integer[] idx = new Integer[imp.length];
        for (int i = 0; i < imp.length; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Double.compare(imp[b], imp[a]));
        int kk = Math.min(k, imp.length);
        double[] out = new double[kk];
        for (int i = 0; i < kk; i++) out[i] = imp[idx[i]];
        return out;
    }

    public static void writeSummary(List<RunResult> all, Path path) throws Exception {
        try (PrintWriter w = new PrintWriter(new FileWriter(path.toFile()))) {
            w.println("dataset,variant,seed,instances,accuracy,kappa,kappa_per,recovery_time,"
                    + "drift_count,feature_stability,total_kept,total_surgical,total_full,"
                    + "total_no_replacement,refresh_calls,weighted_predictions,unweighted_fallbacks,ram_hours");
            for (RunResult r : all) {
                w.printf(Locale.ROOT,
                        "%s,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%d,%.6f,%d,%d,%d,%d,%d,%d,%d,%.6f%n",
                        r.dataset, r.variant, r.seed, r.instances,
                        r.accuracy, r.kappa, r.kappaPer,
                        Double.isNaN(r.recoveryTime) ? -1.0 : r.recoveryTime,
                        r.driftCount, r.featureStability,
                        r.totalKept, r.totalSurgical, r.totalFull, r.totalNoReplacement,
                        r.refreshCalls, r.weightedPredictions, r.unweightedFallbacks,
                        r.ramHours);
            }
        }
    }

    public static void runStatistics(Cfg cfg, List<RunResult> all) throws Exception {
        Map<String, Map<String, List<Double>>> byDsByVar = new LinkedHashMap<>();
        for (RunResult r : all) {
            byDsByVar.computeIfAbsent(r.dataset, k -> new LinkedHashMap<>())
                    .computeIfAbsent(r.variant, k -> new ArrayList<>())
                    .add(r.kappa);
        }
        List<String> datasets = new ArrayList<>(byDsByVar.keySet());
        List<String> methods = new ArrayList<>();
        for (Map<String, List<Double>> m : byDsByVar.values()) {
            for (String var : m.keySet()) if (!methods.contains(var)) methods.add(var);
        }
        if (methods.size() < 2 || datasets.size() < 2) {
            System.err.println("[E3] not enough methods/datasets for statistical tests");
            return;
        }
        double[][] matrix = new double[datasets.size()][methods.size()];
        Path mPath = Paths.get(cfg.outputDir, "E3_kappa_matrix.csv");
        try (PrintWriter w = new PrintWriter(new FileWriter(mPath.toFile()))) {
            w.print("dataset");
            for (String m : methods) w.print("," + m);
            w.println();
            for (int i = 0; i < datasets.size(); i++) {
                String ds = datasets.get(i);
                w.print(ds);
                for (int j = 0; j < methods.size(); j++) {
                    List<Double> ks = byDsByVar.get(ds).getOrDefault(methods.get(j), new ArrayList<>());
                    double mean = ks.isEmpty() ? Double.NaN
                            : ks.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
                    matrix[i][j] = Double.isNaN(mean) ? 0.0 : mean;
                    w.printf(Locale.ROOT, ",%.6f", mean);
                }
                w.println();
            }
        }

        StatisticalTests st = new StatisticalTests(0.05, true);
        StatisticalTests.Report rep = st.runFull(matrix, datasets, methods);
        Path reportPath = Paths.get(cfg.outputDir, "E3_stats_report.txt");
        try (PrintWriter w = new PrintWriter(new FileWriter(reportPath.toFile()))) {
            w.println(rep.summary());
            w.println();
            w.println("Pairwise Wilcoxon p-values (kappa per dataset, mean over seeds):");
            w.print("from\\to");
            for (String m : methods) w.print("," + m);
            w.println();
            for (int a = 0; a < methods.size(); a++) {
                w.print(methods.get(a));
                for (int b = 0; b < methods.size(); b++) {
                    double p = rep.pairwiseWilcoxon[a][b];
                    w.print("," + (Double.isNaN(p) ? "" : String.format(Locale.ROOT, "%.4g", p)));
                }
                w.println();
            }
            w.println();
            w.println("Targeted comparisons vs baselines (alpha=0.05):");
            String[] baselines = {"SRP+S1", "ARF+S2"};
            for (String base : baselines) {
                int bi = methods.indexOf(base);
                if (bi < 0) continue;
                double[] vb = new double[datasets.size()];
                for (int i = 0; i < datasets.size(); i++) vb[i] = matrix[i][bi];
                for (int j = 0; j < methods.size(); j++) {
                    String mn = methods.get(j);
                    if (!mn.startsWith("DA-SRP")) continue;
                    double[] va = new double[datasets.size()];
                    for (int i = 0; i < datasets.size(); i++) va[i] = matrix[i][j];
                    try {
                        var res = st.wilcoxon(va, vb);
                        double meanA = Arrays.stream(va).average().orElse(0);
                        double meanB = Arrays.stream(vb).average().orElse(0);
                        String tag = res.degenerate ? "DEGEN"
                                : (res.pValue < 0.05 ? (meanA > meanB ? "WIN" : "LOSE") : "ns");
                        w.printf(Locale.ROOT,
                                "  %-30s vs %-10s  meanA=%.4f meanB=%.4f  p=%.4g  W/L/T=%d/%d/%d  %s%n",
                                mn, base, meanA, meanB, res.pValue, res.wins, res.losses, res.ties, tag);
                    } catch (Exception ex) {
                        w.println("  " + mn + " vs " + base + " -> error " + ex);
                    }
                }
            }
        }
        rep.exportCD(Paths.get(cfg.outputDir));
        System.out.println("[E3] stats -> " + reportPath);
    }

    public static class DASRPNoWeighting extends DriftAwareSRP {
        public DASRPNoWeighting(SRPWrapper s, double tau, long seed, FeatureImportance imp) {
            super(s, tau, seed, imp);
        }
        @Override
        public double[] predictProba(Instance full) {
            return getSRPWrapper().predictProba(full);
        }
        @Override
        public int predict(Instance full) {
            double[] v = predictProba(full);
            if (v == null || v.length == 0) return 0;
            int b = 0; for (int i = 1; i < v.length; i++) if (v[i] > v[b]) b = i;
            return b;
        }
        @Override
        public String name() { return "DASRP_AB(no-weighting) over " + getSRPWrapper().name(); }
    }
}
