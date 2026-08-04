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
import thesis.models.DAARFWrapper;
import thesis.models.DriftAwareSRP;
import thesis.models.FeatureImportance;
import thesis.models.FeatureSpace;
import thesis.models.HoeffdingTreeWrapper;
import thesis.models.MajorityClassWrapper;
import thesis.models.ModelWrapper;
import thesis.models.NoChangeWrapper;
import thesis.models.SRPWrapper;
import thesis.pipeline.SyntheticStreamFactory;
import thesis.selection.AlarmTriggeredSelector;
import thesis.selection.DriftAwareSelector;
import thesis.selection.FeatureSelector;
import thesis.selection.FilterRanker;
import thesis.selection.InformationGainRanker;
import thesis.selection.NoFeatureSelection;
import thesis.selection.PeriodicSelector;
import thesis.selection.StaticFeatureSelector;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Unified, multi-threaded experiment runner for the entire E1–E5 thesis matrix.
 *
 * <p>The runner reads {@code master_experiments.json}, expands every
 * {@code (block, dataset, variant, seed)} tuple into a work item, runs them on a
 * fixed-size thread pool, then collects results sequentially and writes:
 * <ul>
 *   <li>{@code results/runs_raw.csv} and {@code results/master_summary.csv} (legacy)</li>
 *   <li>{@code results/<block>/<output_file>.csv} — per-block summary</li>
 *   <li>{@code results/<block>/{windows,drift_alarms,feature_selections,feature_importance,
 *       recovery_time,adaptation_events}.csv} — detailed per-run artefacts</li>
 *   <li>{@code results/<block>/stat_tests/} — Friedman + Nemenyi + Wilcoxon + CD diagrams
 *       (one set per metric)</li>
 * </ul>
 *
 * <p>Each worker thread owns its stream, selector, detector, model, metrics collector
 * and a {@link RunDetailedRecorder}; results are bundled into a {@link RunArtifacts}
 * record that is pushed to a {@link ConcurrentLinkedQueue}. All CSV writers and the
 * statistical analysis run single-threaded on the main thread after the pool drains —
 * there is no shared mutable state during the hot path.
 *
 * <p>Supported models:    {@code HT}, {@code ARF}, {@code SRP}, {@code DA-SRP-A},
 * {@code DA-SRP-AB}, {@code DA-SRP-ABC}, {@code DA-ARF}, {@code MAJORITY},
 * {@code NOCHANGE}. Selectors: {@code NONE}, {@code S1..S4}. Detectors: {@code ADWIN},
 * {@code HDDM_A}, {@code HDDM_W}, {@code KSWIN}.
 */
public final class UnifiedStreamExperimentRunner {

    // ========================================================================
    // Configuration
    // ========================================================================

    public static final class DatasetSpec {
        public String name;
        public String type = "synthetic";
        public String generator;
        public String path;
        public int n = 100_000;
        public int noiseFeatures = 0;
        public int numDrifts = 0;
        public double sigma = 0.01;
        public double speed = 0.001;
        public int driftFeatures = 5;
        public long maxInstances = -1;
    }

    public static final class VariantSpec {
        public String name;
        public String model;
        public String selector = "S1";
        public String detector = "ADWIN";
        public int periodicInterval = 1000;
        public int wPostDrift = 1000;
        public double detectorDelta = 0.002;
        public double kswinAlpha = 0.005;
        public int kswinWindow = 200;
        public int ensembleSize = 10;
        public double lambda = 6.0;
        public double tau = 0.5;
        public double w1 = 0.7;
        public double importancePower = 2.0;
        public double samplingBeta = 0.7;
        public double topKFraction = 0.5;
        public double correctionAlpha = 0.15;
        public double maxBlendAlpha = 0.5;
        public double unlocalizedFallbackFraction = 0.20;
        public double unstableImportanceQuantile = 0.50;
        public double surgicalReplacementTolerance = 0.95;
        public double daArfExternalResetFraction = 0.20;
        public int daArfSubspaceSize = -1;
        public boolean daArfUseBackground = true;
        public double daArfWarningDelta = 1e-4;
        public double daArfDriftDelta = 1e-5;
        public String daArfExternalMode = "RESET";   // A2: RESET | SURGICAL
        public boolean daArfIntrinsicDrift = true;    // A4: disable intrinsic ADWIN when false
        public boolean daArfGateExternal = false;     // A3: skip trees with pending background
        public double daArfSubspaceFraction = 0.5;    // A6b: subspace = ceil(frac*d); <=0 -> ceil(sqrt(d)).
                                                       // Default 0.5: tuned trees need a wider subspace than
                                                       // ceil(sqrt(d)) or they overfit under continuous drift.
        public int daArfTreeGracePeriod = 50;         // A7: base-tree grace period (MOA ARF = 50)
        public double daArfTreeSplitConfidence = 0.01; // A7: base-tree split confidence (MOA ARF = 0.01)
    }

    public static final class Block {
        public String id;
        public String description = "";
        public String outputFile;
        public List<DatasetSpec> datasets = new ArrayList<>();
        public List<VariantSpec> variants = new ArrayList<>();
    }

    public static final class Cfg {
        public String outputDir = "results";
        public int warmup = 1500;
        public int windowSize = 1000;
        public int ramSampleEvery = 200;
        public int numThreads = 12;
        public long defaultMaxInstances = Long.MAX_VALUE;
        public boolean skipMissingArff = true;
        public boolean realDatasetsReadAll = true;
        public List<Integer> seeds = new ArrayList<>();
        public List<Block> blocks = new ArrayList<>();

        // Per-artifact toggles — default on so legacy behaviour is preserved.
        public boolean writeWindowMetrics = true;
        public boolean writeDriftAlarms = true;
        public boolean writeFeatureSelections = true;
        public boolean writeFeatureImportance = true;
        public boolean writeRecoveryTime = true;
        public boolean writeAdaptationEvents = true;
        public boolean writeStatisticalTests = true;

        public double statisticalAlpha = 0.05;

        public static Cfg load(Path path) throws Exception {
            ObjectMapper m = new ObjectMapper();
            JsonNode r = m.readTree(path.toFile());
            Cfg c = new Cfg();
            c.outputDir          = r.path("output_dir").asText(c.outputDir);
            c.warmup             = r.path("warmup").asInt(c.warmup);
            c.windowSize         = r.path("window_size").asInt(c.windowSize);
            c.ramSampleEvery     = r.path("ram_sample_every").asInt(c.ramSampleEvery);
            c.numThreads         = r.path("num_threads").asInt(c.numThreads);
            c.defaultMaxInstances = r.path("default_max_instances").asLong(c.defaultMaxInstances);
            c.skipMissingArff    = r.path("skip_missing_arff").asBoolean(c.skipMissingArff);
            c.realDatasetsReadAll = r.path("real_datasets_read_all").asBoolean(c.realDatasetsReadAll);
            c.writeWindowMetrics      = r.path("write_window_metrics").asBoolean(c.writeWindowMetrics);
            c.writeDriftAlarms        = r.path("write_drift_alarms").asBoolean(c.writeDriftAlarms);
            c.writeFeatureSelections  = r.path("write_feature_selections").asBoolean(c.writeFeatureSelections);
            c.writeFeatureImportance  = r.path("write_feature_importance").asBoolean(c.writeFeatureImportance);
            c.writeRecoveryTime       = r.path("write_recovery_time").asBoolean(c.writeRecoveryTime);
            c.writeAdaptationEvents   = r.path("write_adaptation_events").asBoolean(c.writeAdaptationEvents);
            c.writeStatisticalTests   = r.path("write_statistical_tests").asBoolean(c.writeStatisticalTests);
            c.statisticalAlpha        = r.path("statistical_alpha").asDouble(c.statisticalAlpha);
            r.path("seeds").forEach(n -> c.seeds.add(n.asInt()));
            if (c.seeds.isEmpty()) c.seeds.addAll(List.of(1, 2, 3, 4, 5));
            if (c.seeds.size() < 5) {
                System.err.printf("[Unified][WARN] only %d seed(s) configured; thesis spec requires 5.%n",
                        c.seeds.size());
            }
            if (c.numThreads < 1 || c.numThreads > 64)
                throw new IllegalArgumentException("num_threads must be in [1,64]");
            for (JsonNode bj : r.path("blocks")) {
                Block b = new Block();
                b.id          = req(bj, "id");
                b.description = bj.path("description").asText(b.description);
                b.outputFile  = bj.path("output_file").asText(null);
                for (JsonNode dj : bj.path("datasets")) b.datasets.add(parseDataset(dj));
                for (JsonNode vj : bj.path("variants")) b.variants.add(parseVariant(vj));
                if (b.datasets.isEmpty()) throw new IllegalArgumentException("block " + b.id + ": datasets empty");
                if (b.variants.isEmpty()) throw new IllegalArgumentException("block " + b.id + ": variants empty");
                c.blocks.add(b);
            }
            if (c.blocks.isEmpty()) throw new IllegalArgumentException("blocks empty");
            return c;
        }

        private static String req(JsonNode n, String key) {
            String v = n.path(key).asText("");
            if (v.isEmpty()) throw new IllegalArgumentException("missing required field: " + key);
            return v;
        }

        private static DatasetSpec parseDataset(JsonNode d) {
            DatasetSpec ds = new DatasetSpec();
            ds.name          = req(d, "name");
            ds.type          = d.path("type").asText(ds.type);
            ds.generator     = d.path("generator").asText(null);
            ds.path          = d.path("path").asText(null);
            ds.n             = d.path("n").asInt(ds.n);
            ds.noiseFeatures = d.path("noise_features").asInt(ds.noiseFeatures);
            ds.numDrifts     = d.path("num_drifts").asInt(ds.numDrifts);
            ds.sigma         = d.path("sigma").asDouble(ds.sigma);
            ds.speed         = d.path("speed").asDouble(ds.speed);
            ds.driftFeatures = d.path("drift_features").asInt(ds.driftFeatures);
            ds.maxInstances  = d.path("max_instances").asLong(ds.maxInstances);
            return ds;
        }

        private static VariantSpec parseVariant(JsonNode v) {
            VariantSpec vs = new VariantSpec();
            vs.name             = req(v, "name");
            vs.model            = req(v, "model");
            vs.selector         = v.path("selector").asText(vs.selector);
            vs.detector         = v.path("detector").asText(vs.detector);
            vs.periodicInterval = v.path("periodic_interval").asInt(vs.periodicInterval);
            vs.wPostDrift       = v.path("w_post_drift").asInt(vs.wPostDrift);
            vs.detectorDelta    = v.path("detector_delta").asDouble(vs.detectorDelta);
            vs.kswinAlpha       = v.path("kswin_alpha").asDouble(vs.kswinAlpha);
            vs.kswinWindow      = v.path("kswin_window").asInt(vs.kswinWindow);
            vs.ensembleSize     = v.path("ensemble_size").asInt(vs.ensembleSize);
            vs.lambda           = v.path("lambda").asDouble(vs.lambda);
            vs.tau              = v.path("tau").asDouble(vs.tau);
            vs.w1               = v.path("w1").asDouble(vs.w1);
            vs.importancePower  = v.path("importance_power").asDouble(vs.importancePower);
            vs.samplingBeta     = v.path("sampling_beta").asDouble(vs.samplingBeta);
            vs.topKFraction     = v.path("topk_fraction").asDouble(vs.topKFraction);
            vs.correctionAlpha  = v.path("correction_alpha").asDouble(vs.correctionAlpha);
            vs.maxBlendAlpha    = v.path("max_blend_alpha").asDouble(vs.maxBlendAlpha);
            vs.unlocalizedFallbackFraction =
                    v.path("unlocalized_fallback_fraction").asDouble(vs.unlocalizedFallbackFraction);
            vs.unstableImportanceQuantile =
                    v.path("unstable_importance_quantile").asDouble(vs.unstableImportanceQuantile);
            vs.surgicalReplacementTolerance =
                    v.path("surgical_replacement_tolerance").asDouble(vs.surgicalReplacementTolerance);
            vs.daArfExternalResetFraction =
                    v.path("daarf_external_reset_fraction").asDouble(vs.daArfExternalResetFraction);
            vs.daArfSubspaceSize    = v.path("daarf_subspace_size").asInt(vs.daArfSubspaceSize);
            vs.daArfUseBackground   = v.path("daarf_use_background").asBoolean(vs.daArfUseBackground);
            vs.daArfWarningDelta    = v.path("daarf_warning_delta").asDouble(vs.daArfWarningDelta);
            vs.daArfDriftDelta      = v.path("daarf_drift_delta").asDouble(vs.daArfDriftDelta);
            vs.daArfExternalMode    = v.path("daarf_external_mode").asText(vs.daArfExternalMode);
            vs.daArfIntrinsicDrift  = v.path("daarf_intrinsic_drift").asBoolean(vs.daArfIntrinsicDrift);
            vs.daArfGateExternal    = v.path("daarf_gate_external").asBoolean(vs.daArfGateExternal);
            vs.daArfSubspaceFraction = v.path("daarf_subspace_fraction").asDouble(vs.daArfSubspaceFraction);
            vs.daArfTreeGracePeriod = v.path("daarf_tree_grace_period").asInt(vs.daArfTreeGracePeriod);
            vs.daArfTreeSplitConfidence = v.path("daarf_tree_split_confidence").asDouble(vs.daArfTreeSplitConfidence);
            return vs;
        }
    }

    // ========================================================================
    // Result types
    // ========================================================================

    /** Final per-run metrics — immutable, safe to enqueue from any thread. */
    public static final class RunResult {
        public final String blockId, dataset, variant, model, selector, detector;
        public final int seed;
        public final long instances;
        public final double accuracy, kappa, kappaPer;
        public final long driftCount;
        public final double ramHoursGB, peakMB, throughput;
        public final long wallMillis;
        public final long extKeepCount, extFullCount;
        public final long daSrpKept, daSrpSurgical, daSrpFull, daSrpNoReplacement;
        public final String status;
        public final String error;

        public RunResult(String blockId, String dataset, String variant, String model,
                         String selector, String detector, int seed, long instances,
                         double accuracy, double kappa, double kappaPer, long driftCount,
                         double ramHoursGB, double peakMB, double throughput, long wallMillis,
                         long extKeepCount, long extFullCount,
                         long daSrpKept, long daSrpSurgical, long daSrpFull, long daSrpNoReplacement,
                         String status, String error) {
            this.blockId = blockId; this.dataset = dataset; this.variant = variant;
            this.model = model; this.selector = selector; this.detector = detector;
            this.seed = seed; this.instances = instances;
            this.accuracy = accuracy; this.kappa = kappa; this.kappaPer = kappaPer;
            this.driftCount = driftCount;
            this.ramHoursGB = ramHoursGB; this.peakMB = peakMB; this.throughput = throughput;
            this.wallMillis = wallMillis;
            this.extKeepCount = extKeepCount; this.extFullCount = extFullCount;
            this.daSrpKept = daSrpKept; this.daSrpSurgical = daSrpSurgical;
            this.daSrpFull = daSrpFull; this.daSrpNoReplacement = daSrpNoReplacement;
            this.status = status; this.error = error;
        }
    }

    /**
     * Complete bundle of artefacts produced by a single worker. The {@code recorder}
     * is {@code null} for {@code FAIL}ed runs — downstream writers skip those.
     */
    public static final class RunArtifacts {
        public final RunResult result;
        public final RunDetailedRecorder recorder;
        public RunArtifacts(RunResult result, RunDetailedRecorder recorder) {
            this.result = result;
            this.recorder = recorder;
        }
        public boolean ok() { return "OK".equals(result.status); }
    }

    public static final class WorkItem {
        public final Block block;
        public final DatasetSpec ds;
        public final VariantSpec v;
        public final int seed;
        public WorkItem(Block b, DatasetSpec d, VariantSpec vv, int s) {
            this.block = b; this.ds = d; this.v = vv; this.seed = s;
        }
    }

    // ========================================================================
    // Entry point + orchestration
    // ========================================================================

    public static void main(String[] args) throws Exception {
        Path configPath = args.length > 0
                ? Paths.get(args[0])
                : Paths.get("src/main/java/thesis/experiments/master_experiments.json");
        if (!Files.exists(configPath)) {
            throw new RuntimeException("Config not found: " + configPath.toAbsolutePath());
        }
        Cfg cfg = Cfg.load(configPath);
        new UnifiedStreamExperimentRunner().run(cfg);
    }

    public void run(Cfg cfg) throws Exception {
        Files.createDirectories(Paths.get(cfg.outputDir));

        List<WorkItem> work = expandWork(cfg);
        if (work.isEmpty()) {
            System.err.println("[Unified] no work after filtering — exiting.");
            return;
        }

        ConcurrentLinkedQueue<RunArtifacts> sink = executeAll(cfg, work);
        List<RunArtifacts> sorted = sortDeterministically(sink);

        // Sequential writers. Order: runs_raw → summaries → detailed CSVs → stat tests.
        writeRunsRaw(cfg, sorted);
        writeMasterAndBlockSummaries(cfg, sorted);
        writeDetailedPerBlockCsvs(cfg, sorted);
        if (cfg.writeStatisticalTests) {
            writeStatisticalTests(cfg, sorted);
        }
    }

    private static List<WorkItem> expandWork(Cfg cfg) {
        List<WorkItem> work = new ArrayList<>();
        for (Block b : cfg.blocks) {
            for (DatasetSpec ds : b.datasets) {
                if ("arff".equalsIgnoreCase(ds.type)
                        && (ds.path == null || !Files.exists(Paths.get(ds.path)))) {
                    if (cfg.skipMissingArff) {
                        System.err.printf("[Unified] missing ARFF, skip: %s [%s] path=%s%n",
                                ds.name, b.id, ds.path);
                        continue;
                    }
                    throw new RuntimeException("ARFF not found: " + ds.path);
                }
                for (VariantSpec v : b.variants) {
                    for (int seed : cfg.seeds) work.add(new WorkItem(b, ds, v, seed));
                }
            }
        }
        return work;
    }

    private ConcurrentLinkedQueue<RunArtifacts> executeAll(Cfg cfg, List<WorkItem> work)
            throws InterruptedException {
        AtomicInteger done = new AtomicInteger();
        ConcurrentLinkedQueue<RunArtifacts> sink = new ConcurrentLinkedQueue<>();
        long t0 = System.currentTimeMillis();
        System.out.printf(Locale.ROOT, "[Unified] launching %d runs on %d threads%n",
                work.size(), cfg.numThreads);

        ExecutorService pool = Executors.newFixedThreadPool(cfg.numThreads, r -> {
            Thread t = new Thread(r, "exp-worker");
            t.setDaemon(false);
            return t;
        });
        List<Future<RunArtifacts>> futures = new ArrayList<>(work.size());
        try {
            for (WorkItem wi : work) {
                futures.add(pool.submit(() -> {
                    RunArtifacts ra = new RunWorker(cfg, wi).call();
                    int k = done.incrementAndGet();
                    System.out.printf(Locale.ROOT,
                            "[Unified] (%d/%d) %s | %s | %s | seed=%d → n=%d k=%.4f acc=%.4f thr=%.0f/s [%s]%n",
                            k, work.size(), wi.block.id, wi.ds.name, wi.v.name, wi.seed,
                            ra.result.instances, ra.result.kappa, ra.result.accuracy,
                            ra.result.throughput, ra.result.status);
                    sink.add(ra);
                    return ra;
                }));
            }
            for (Future<RunArtifacts> f : futures) {
                try { f.get(); }
                catch (ExecutionException ex) { ex.printStackTrace(); }
            }
        } finally {
            pool.shutdown();
            if (!pool.awaitTermination(120, TimeUnit.SECONDS)) pool.shutdownNow();
        }
        long t1 = System.currentTimeMillis();
        System.out.printf(Locale.ROOT, "[Unified] all done in %.1fs%n", (t1 - t0) / 1000.0);
        return sink;
    }

    private static List<RunArtifacts> sortDeterministically(ConcurrentLinkedQueue<RunArtifacts> sink) {
        List<RunArtifacts> sorted = new ArrayList<>(sink);
        sorted.sort(ARTIFACT_ORDER);
        return sorted;
    }

    private static final Comparator<RunArtifacts> ARTIFACT_ORDER = (a, b) -> {
        int c = a.result.blockId.compareTo(b.result.blockId);
        if (c != 0) return c;
        c = a.result.dataset.compareTo(b.result.dataset);
        if (c != 0) return c;
        c = a.result.variant.compareTo(b.result.variant);
        if (c != 0) return c;
        return Integer.compare(a.result.seed, b.result.seed);
    };

    /** Map a block id like {@code E1_baselines} or {@code E1} to {@code results/E1/}. */
    static Path blockFolder(Cfg cfg, String blockId) {
        String prefix = blockId;
        int u = blockId.indexOf('_');
        if (u > 0) prefix = blockId.substring(0, u);
        return Paths.get(cfg.outputDir, prefix);
    }

    // ========================================================================
    // RunWorker — owns a single run end-to-end.
    // ========================================================================

    private static final class RunWorker {
        private final Cfg cfg;
        private final WorkItem wi;
        private final long t0;

        // Late-bound state, owned by this worker thread only.
        private InstanceStream stream;
        private InstancesHeader header;
        private FeatureSpace space;
        private int numFeatures;
        private int numClasses;
        private double[][] warmupWindow;
        private int[] warmupLabels;
        private int warmupCollected;

        private FeatureSelector selector;
        private FeatureImportance importance;
        private PiDDiscretizer fullRankingPid;
        private FilterRanker fullRanker;
        private ModelWrapper model;
        private TwoLevelDriftDetector detector;
        private MetricsCollector metrics;
        private RunDetailedRecorder recorder;

        RunWorker(Cfg cfg, WorkItem wi) {
            this.cfg = cfg;
            this.wi = wi;
            this.t0 = System.currentTimeMillis();
        }

        RunArtifacts call() {
            try {
                openStream();
                collectWarmup();
                buildComponents();
                attachRecorder();
                long n = runPrequentialLoop();
                recorder.finalizeAtEnd(n, metrics);
                return new RunArtifacts(buildSuccessResult(n), recorder);
            } catch (Throwable t) {
                long wall = System.currentTimeMillis() - t0;
                return new RunArtifacts(buildFailResult(t, wall), null);
            }
        }

        private void openStream() {
            stream = buildStream(wi.ds, wi.seed);
            if (stream instanceof OptionHandler) ((OptionHandler) stream).prepareForUse();
            header = stream.getHeader();
            numFeatures = header.numAttributes() - 1;
            numClasses  = header.numClasses();
            if (numFeatures < 1) throw new IllegalStateException("numFeatures < 1");
            if (numClasses  < 2) throw new IllegalStateException("numClasses < 2");
            space = new FeatureSpace(header);
        }

        private void collectWarmup() {
            int collected = 0;
            warmupWindow = new double[cfg.warmup][];
            warmupLabels = new int[cfg.warmup];
            while (collected < cfg.warmup && stream.hasMoreInstances()) {
                Instance x = stream.nextInstance().getData();
                warmupWindow[collected] = space.extractFeatures(x);
                warmupLabels[collected] = (int) x.classValue();
                collected++;
            }
            if (collected == 0) throw new IllegalStateException("no instances during warmup");
            if (collected < cfg.warmup) {
                warmupWindow = Arrays.copyOf(warmupWindow, collected);
                warmupLabels = Arrays.copyOf(warmupLabels, collected);
            }
            this.warmupCollected = collected;
        }

        private void buildComponents() {
            selector = buildSelector(wi.v, numFeatures, numClasses);
            selector.initialize(warmupWindow, warmupLabels);
            importance = new FeatureImportance(numFeatures);
            buildFullFeatureRankerFromWarmup();
            model = buildModel(wi.v, selector, header, numClasses, wi.seed, importance);
            detector = buildDetector(wi.v, numFeatures);
            metrics = new MetricsCollector(numClasses, cfg.windowSize, /*logEvery=*/0, cfg.ramSampleEvery);
        }

        private void buildFullFeatureRankerFromWarmup() {
            fullRankingPid = new PiDDiscretizer(numFeatures, numClasses);
            fullRanker = new InformationGainRanker(numFeatures, fullRankingPid.getB2(), numClasses);
            for (int i = 0; i < warmupWindow.length; i++) {
                if (!rowAllFinite(warmupWindow[i])) continue;
                fullRankingPid.update(warmupWindow[i], warmupLabels[i]);
            }
            if (!fullRankingPid.isReady()) return;
            for (int i = 0; i < warmupWindow.length; i++) {
                if (!rowAllFinite(warmupWindow[i])) continue;
                fullRanker.update(fullRankingPid.discretizeAll(warmupWindow[i]), warmupLabels[i]);
            }
            if (fullRanker.isReady()) {
                importance.update(fullRanker.getFeatureScores(), new double[numFeatures]);
            }
        }

        private void attachRecorder() {
            recorder = new RunDetailedRecorder(
                    wi.block.id, wi.ds.name, wi.v.name, wi.v.model, wi.v.selector, wi.v.detector,
                    wi.seed, numFeatures, cfg.windowSize);
            recorder.onInitialSelection(warmupCollected, selector.getCurrentSelection());
            recorder.onImportanceSnapshot(warmupCollected, importance.getImportance(),
                    selector.getCurrentSelection(), Set.of());

            if (model instanceof DriftAwareSRP) {
                ((DriftAwareSRP) model).setDriftListener(ev ->
                        recorder.onDASRPEvent(ev.instanceIdx, ev.summary));
            }
        }

        private long runPrequentialLoop() {
            long effMax = effectiveMaxInstances(cfg, wi.ds);
            long prevExtKeep = 0, prevExtFull = 0, prevExtSurg = 0, prevIntrReset = 0, prevPromo = 0;
            DAARFWrapper daArf = (model instanceof DAARFWrapper) ? (DAARFWrapper) model : null;
            if (daArf != null) {
                prevExtKeep = daArf.getExtKeepCount();
                prevExtFull = daArf.getExtFullCount();
                prevExtSurg = daArf.getExtSurgicalCount();
                prevIntrReset = daArf.getIntrinsicFullResetCount();
                prevPromo = daArf.getBkgPromotions();
            }

            long n = warmupCollected;
            while (stream.hasMoreInstances() && n < effMax) {
                Instance x = stream.nextInstance().getData();
                int yTrue = (int) x.classValue();
                double[] feats = space.extractFeatures(x);
                updateFullFeatureRanker(feats, yTrue);
                long s0 = System.nanoTime();
                int yHat = model.predict(x);
                long elapsed = System.nanoTime() - s0;
                double err = (yHat == yTrue) ? 0.0 : 1.0;

                detector.update(err, feats);
                boolean alarm = detector.isGlobalDriftDetected();
                Set<Integer> drifting = alarm ? detector.getDriftingFeatureIndices() : Set.of();
                if (alarm) updateFeatureImportanceFromDetector();
                metrics.update(yTrue, yHat, elapsed);
                if (alarm) metrics.onDriftAlarm();
                selector.update(feats, yTrue, alarm, drifting);
                model.train(x, yTrue, alarm, drifting);
                n++;

                recorder.onInstance(n, yTrue, yHat,
                        selector.getCurrentSelection(), alarm, drifting, metrics);
                if (alarm) {
                    recorder.onImportanceSnapshot(n, importance.getImportance(),
                            selector.getCurrentSelection(), drifting);
                }
                if (daArf != null) {
                    long curK = daArf.getExtKeepCount();
                    long curF = daArf.getExtFullCount();
                    long curS = daArf.getExtSurgicalCount();
                    long curI = daArf.getIntrinsicFullResetCount();
                    long curP = daArf.getBkgPromotions();
                    if (curK != prevExtKeep || curF != prevExtFull || curS != prevExtSurg
                            || curI != prevIntrReset || curP != prevPromo) {
                        recorder.onDAARFEvent(n, curK - prevExtKeep, curF - prevExtFull,
                                curS - prevExtSurg, curI - prevIntrReset, curP - prevPromo);
                        prevExtKeep = curK;
                        prevExtFull = curF;
                        prevExtSurg = curS;
                        prevIntrReset = curI;
                        prevPromo = curP;
                    }
                }
            }
            return n;
        }

        private void updateFullFeatureRanker(double[] feats, int classLabel) {
            if (fullRankingPid == null || fullRanker == null) return;
            if (feats == null || feats.length != numFeatures || !rowAllFinite(feats)) return;
            fullRankingPid.update(feats, classLabel);
            if (fullRankingPid.isReady()) {
                fullRanker.update(fullRankingPid.discretizeAll(feats), classLabel);
            }
        }

        private void updateFeatureImportanceFromDetector() {
            if (importance == null || fullRanker == null || !fullRanker.isReady()) return;
            double[] scores = fullRanker.getFeatureScores();
            if (scores == null || scores.length != numFeatures) return;
            double[] p = detector == null ? null : detector.getLastPValues();
            double[] ksProxy = invertPValues(p, numFeatures);
            importance.update(scores, ksProxy);
        }

        private static boolean rowAllFinite(double[] row) {
            if (row == null) return false;
            for (double v : row) if (!Double.isFinite(v)) return false;
            return true;
        }

        private RunResult buildSuccessResult(long n) {
            MetricsCollector.Snapshot snap = metrics.snapshot();
            long wall = System.currentTimeMillis() - t0;
            double secs = wall / 1000.0;
            double thr = secs > 0.0 ? (double) n / secs : 0.0;
            long extKeep = 0, extFull = 0;
            long daKept = 0, daSurg = 0, daFull = 0, daNoRep = 0;
            if (model instanceof DAARFWrapper) {
                DAARFWrapper d = (DAARFWrapper) model;
                extKeep = d.getExtKeepCount();
                extFull = d.getExtFullCount();
            } else if (model instanceof DriftAwareSRP) {
                DriftAwareSRP d = (DriftAwareSRP) model;
                daKept = d.getTotalKept();
                daSurg = d.getTotalSurgical();
                daFull = d.getTotalFull();
                daNoRep = d.getTotalNoReplacement();
            }
            return new RunResult(wi.block.id, wi.ds.name, wi.v.name, wi.v.model,
                    wi.v.selector, wi.v.detector, wi.seed, n,
                    snap.accuracyWindow, snap.kappa, snap.kappaPer, snap.driftCount,
                    snap.ramHoursGB, snap.peakMB, thr, wall,
                    extKeep, extFull, daKept, daSurg, daFull, daNoRep,
                    "OK", null);
        }

        private RunResult buildFailResult(Throwable t, long wall) {
            return new RunResult(wi.block.id, wi.ds.name, wi.v.name, wi.v.model,
                    wi.v.selector, wi.v.detector, wi.seed, 0,
                    Double.NaN, Double.NaN, Double.NaN, 0,
                    0.0, 0.0, 0.0, wall,
                    0, 0, 0, 0, 0, 0,
                    "FAIL", t.toString());
        }
    }

    private static long effectiveMaxInstances(Cfg cfg, DatasetSpec ds) {
        if (ds.maxInstances > 0) return ds.maxInstances;
        if ("arff".equalsIgnoreCase(ds.type) && cfg.realDatasetsReadAll) return Long.MAX_VALUE;
        return cfg.defaultMaxInstances;
    }

    // ========================================================================
    // Factories
    // ========================================================================

    static InstanceStream buildStream(DatasetSpec ds, int seed) {
        if ("arff".equalsIgnoreCase(ds.type)) {
            ArffFileStream s = new ArffFileStream(ds.path, -1);
            s.prepareForUse();
            return s;
        }
        String g = ds.generator == null ? ds.name : ds.generator;
        InstanceStream base;
        switch (g.toUpperCase(Locale.ROOT)) {
            case "SEA":
                base = ds.numDrifts > 0
                        ? SyntheticStreamFactory.createMultiDriftSEA(seed, ds.n, ds.numDrifts)
                        : SyntheticStreamFactory.createSEA(seed, ds.n);
                break;
            case "STAGGER":
                base = ds.numDrifts > 0
                        ? SyntheticStreamFactory.createMultiDriftSTAGGER(seed, ds.n, ds.numDrifts)
                        : SyntheticStreamFactory.createSTAGGER(seed, ds.n);
                break;
            case "HYPERPLANE":
                base = SyntheticStreamFactory.createHyperplane(seed, ds.sigma, ds.n);
                break;
            case "RANDOMRBF":
                base = SyntheticStreamFactory.createRandomRBF(seed, ds.speed, ds.n);
                break;
            case "CUSTOMFEATUREDRIFT":
            case "FEATUREDRIFT":
                base = SyntheticStreamFactory.createCustomFeatureDrift(seed, ds.driftFeatures, ds.sigma, ds.n);
                break;
            case "LED":
            case "LEDDRIFT":
                base = SyntheticStreamFactory.createLEDDrift(seed, ds.driftFeatures, ds.n);
                break;
            default: throw new IllegalArgumentException("Unknown generator: " + g);
        }
        return ds.noiseFeatures > 0
                ? SyntheticStreamFactory.addNoiseFeatures(base, ds.noiseFeatures, seed)
                : base;
    }

    static FeatureSelector buildSelector(VariantSpec v, int d, int numClasses) {
        int K = StaticFeatureSelector.defaultK(d);
        switch (v.selector.toUpperCase(Locale.ROOT)) {
            case "NONE":
            case "NO_FS":
            case "ALL":
                return new NoFeatureSelection(d);
            case "S1":
                return new StaticFeatureSelector(d, numClasses);
            case "S2":
                return new AlarmTriggeredSelector(
                        d, numClasses, K, Math.max(50, v.wPostDrift),
                        new PiDDiscretizer(d, numClasses),
                        (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc));
            case "S3":
                return new PeriodicSelector(
                        d, numClasses, K, Math.max(100, v.periodicInterval), 100,
                        new PiDDiscretizer(d, numClasses),
                        (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc));
            case "S4":
                return new DriftAwareSelector(
                        d, numClasses, K, Math.max(100, v.periodicInterval), 100,
                        Math.max(50, v.wPostDrift),
                        new PiDDiscretizer(d, numClasses),
                        (nf, nb, nc) -> new InformationGainRanker(nf, nb, nc));
            default: throw new IllegalArgumentException("Unknown selector: " + v.selector);
        }
    }

    static TwoLevelDriftDetector buildDetector(VariantSpec v, int numFeatures) {
        TwoLevelDriftDetector.Config c = new TwoLevelDriftDetector.Config(numFeatures);
        c.level1Delta = v.detectorDelta;
        c.kswinAlpha = v.kswinAlpha;
        c.kswinWindowSize = Math.max(10, v.kswinWindow);
        switch (v.detector.toUpperCase(Locale.ROOT)) {
            case "ADWIN":  c.level1Type = TwoLevelDriftDetector.Level1Type.ADWIN;  break;
            case "HDDM_A": c.level1Type = TwoLevelDriftDetector.Level1Type.HDDM_A; break;
            case "HDDM_W": c.level1Type = TwoLevelDriftDetector.Level1Type.HDDM_W; break;
            case "KSWIN":
                // Tight ADWIN at level-1; per-feature KSWIN at level-2 drives localization.
                c.level1Type = TwoLevelDriftDetector.Level1Type.ADWIN;
                c.level1Delta = Math.min(c.level1Delta, 1e-4);
                break;
            default: throw new IllegalArgumentException("Unknown detector: " + v.detector);
        }
        return new TwoLevelDriftDetector(c);
    }

    static ModelWrapper buildModel(VariantSpec v, FeatureSelector selector,
                                   InstancesHeader header, int numClasses,
                                   int seed, FeatureImportance importance) {
        String key = v.model.toUpperCase(Locale.ROOT).replace('_', '-');
        boolean noExternalFeatureSelection = isNoFeatureSelection(v.selector);
        switch (key) {
            case "HT":            return new HoeffdingTreeWrapper(selector, header);
            case "ARF":           return new ARFWrapper(selector, header,
                                          v.ensembleSize, v.lambda, false, !noExternalFeatureSelection);
            case "SRP":           return new SRPWrapper(selector, header,
                                          v.ensembleSize, v.lambda, false, !noExternalFeatureSelection);
            case "MAJORITY":
            case "MAJORITYCLASS": return new MajorityClassWrapper(selector, numClasses);
            case "NOCHANGE":      return new NoChangeWrapper(selector, numClasses);
            case "DA-SRP-A":      return newDASRP(v, selector, header, seed, /*imp=*/null, /*topK=*/false);
            case "DA-SRP-AB":     return newDASRP(v, selector, header, seed, importance, /*topK=*/false);
            case "DA-SRP-ABC":    return newDASRP(v, selector, header, seed, importance, /*topK=*/true);
            case "DA-ARF":        return newDAARF(v, selector, header, numClasses, seed, importance);
            default: throw new IllegalArgumentException("Unknown model: " + v.model);
        }
    }

    private static boolean isNoFeatureSelection(String selector) {
        String key = selector == null ? "" : selector.toUpperCase(Locale.ROOT).replace('-', '_');
        return "NONE".equals(key) || "NO_FS".equals(key) || "ALL".equals(key);
    }

    private static DriftAwareSRP newDASRP(VariantSpec v, FeatureSelector selector,
                                          InstancesHeader header, int seed,
                                          FeatureImportance importance, boolean useTopK) {
        SRPWrapper srp = new SRPWrapper(selector, header,
                v.ensembleSize, v.lambda, /*resetOnSelectionChange=*/false, /*useHardFilter=*/false);
        DriftAwareSRP da = new DriftAwareSRP(srp, v.tau, seed, importance);
        da.setImportancePower(v.importancePower);
        da.setSamplingBeta(v.samplingBeta);
        da.setUnlocalizedFallbackFraction(v.unlocalizedFallbackFraction);
        da.setUnstableImportanceQuantile(v.unstableImportanceQuantile);
        da.setSurgicalReplacementTolerance(v.surgicalReplacementTolerance);
        if (useTopK) {
            da.setTopKFraction(v.topKFraction);
            da.setCorrectionAlpha(v.correctionAlpha);
            da.setMaxBlendAlpha(v.maxBlendAlpha);
        } else {
            da.setCorrectionAlpha(0.0);
        }
        return da;
    }

    private static DAARFWrapper newDAARF(VariantSpec v, FeatureSelector selector,
                                         InstancesHeader header, int numClasses,
                                         int seed, FeatureImportance importance) {
        int origDim = header.numAttributes() - 1;
        int sub;
        if (v.daArfSubspaceSize > 0) {
            sub = v.daArfSubspaceSize;
        } else if (v.daArfSubspaceFraction > 0.0) {
            sub = Math.max(2, (int) Math.ceil(v.daArfSubspaceFraction * origDim));
        } else {
            sub = Math.max(2, (int) Math.ceil(Math.sqrt(origDim)));
        }
        sub = Math.min(sub, origDim);
        DAARFWrapper da = new DAARFWrapper(selector, header, numClasses,
                v.ensembleSize, sub, v.lambda,
                /*accWindow=*/1000, v.topKFraction,
                v.importancePower, v.samplingBeta,
                v.daArfUseBackground, v.daArfWarningDelta, v.daArfDriftDelta,
                seed, importance);
        da.setTreeParams(v.daArfTreeGracePeriod, v.daArfTreeSplitConfidence);
        da.setUnstableImportanceQuantile(v.unstableImportanceQuantile);
        da.setExternalResetFraction(v.daArfExternalResetFraction);
        da.setSurgicalReplacementTolerance(v.surgicalReplacementTolerance);
        da.setExternalActionMode("SURGICAL".equalsIgnoreCase(v.daArfExternalMode)
                ? DAARFWrapper.ExternalActionMode.SURGICAL
                : DAARFWrapper.ExternalActionMode.RESET);
        da.setIntrinsicDriftEnabled(v.daArfIntrinsicDrift);
        da.setGateExternalOnPendingBackground(v.daArfGateExternal);
        return da;
    }

    // ========================================================================
    // Output: runs_raw.csv
    // ========================================================================

    private static void writeRunsRaw(Cfg cfg, List<RunArtifacts> sorted) throws Exception {
        Path out = Paths.get(cfg.outputDir, "runs_raw.csv");
        try (PrintWriter w = new PrintWriter(new FileWriter(out.toFile()))) {
            w.println("block,dataset,variant,model,selector,detector,seed,instances,"
                    + "accuracy,kappa,kappa_per,drift_count,ram_hours_gb,peak_mb,throughput,wall_ms,"
                    + "ext_keep_count,ext_full_count,da_kept,da_surgical,da_full,da_no_replacement,"
                    + "status,error");
            for (RunArtifacts ra : sorted) {
                RunResult r = ra.result;
                w.printf(Locale.ROOT,
                        "%s,%s,%s,%s,%s,%s,%d,%d,%.6f,%.6f,%.6f,%d,%.6f,%.1f,%.2f,%d,%d,%d,%d,%d,%d,%d,%s,%s%n",
                        r.blockId, r.dataset, r.variant, r.model, r.selector, r.detector,
                        r.seed, r.instances,
                        r.accuracy, r.kappa, r.kappaPer, r.driftCount,
                        r.ramHoursGB, r.peakMB, r.throughput, r.wallMillis,
                        r.extKeepCount, r.extFullCount,
                        r.daSrpKept, r.daSrpSurgical, r.daSrpFull, r.daSrpNoReplacement,
                        r.status, r.error == null ? "" : r.error.replace(',', ';').replace('\n', ' '));
            }
        }
        System.out.println("[Unified] per-run   -> " + out);
    }

    // ========================================================================
    // Output: master_summary.csv + per-block summary CSV
    // ========================================================================

    private static void writeMasterAndBlockSummaries(Cfg cfg, List<RunArtifacts> sorted) throws Exception {
        Map<String, List<RunArtifacts>> agg = new LinkedHashMap<>();
        for (RunArtifacts ra : sorted) {
            if (!ra.ok()) continue;
            String key = ra.result.blockId + "|" + ra.result.dataset + "|" + ra.result.variant;
            agg.computeIfAbsent(key, k -> new ArrayList<>()).add(ra);
        }
        List<String> keys = new ArrayList<>(agg.keySet());
        keys.sort(String::compareTo);

        Path master = Paths.get(cfg.outputDir, "master_summary.csv");
        try (PrintWriter w = new PrintWriter(new FileWriter(master.toFile()))) {
            w.println(summaryHeader());
            for (String k : keys) w.println(summaryRow(agg.get(k)));
        }
        System.out.println("[Unified] master    -> " + master);

        Map<String, List<String>> rowsByBlock = new LinkedHashMap<>();
        Map<String, String> outFileByBlock = new HashMap<>();
        for (Block b : cfg.blocks) {
            outFileByBlock.put(b.id, b.outputFile != null ? b.outputFile : (b.id + ".csv"));
        }
        for (String k : keys) {
            String blockId = k.substring(0, k.indexOf('|'));
            rowsByBlock.computeIfAbsent(blockId, kk -> new ArrayList<>()).add(summaryRow(agg.get(k)));
        }
        for (Map.Entry<String, List<String>> e : rowsByBlock.entrySet()) {
            String file = outFileByBlock.getOrDefault(e.getKey(), e.getKey() + ".csv");
            Path folder = blockFolder(cfg, e.getKey());
            Files.createDirectories(folder);
            Path p = folder.resolve(file);
            try (PrintWriter w = new PrintWriter(new FileWriter(p.toFile()))) {
                w.println(summaryHeader());
                for (String row : e.getValue()) w.println(row);
            }
            System.out.println("[Unified] block " + e.getKey() + " -> " + p);
        }
    }

    private static String summaryHeader() {
        return "block,dataset,variant,model,selector,detector,num_seeds,instances_mean,"
                + "accuracy_mean,accuracy_std,kappa_mean,kappa_std,kappa_per_mean,kappa_per_std,"
                + "drift_count_mean,drift_count_std,ram_hours_gb_mean,ram_hours_gb_std,"
                + "peak_mb_mean,throughput_mean,wall_ms_mean,"
                + "ext_keep_count_mean,ext_full_count_mean,"
                + "da_kept_mean,da_surgical_mean,da_full_mean,da_no_replacement_mean,"
                + "temporal_kappa_mean,temporal_kappa_std,"
                + "recovery_time_mean,recovery_time_std,"
                + "feature_stability_mean,feature_stability_std,"
                + "selection_changes_mean,selection_changes_std,"
                + "drift_alarms_mean,drift_alarms_std,"
                + "mean_selected_feature_count,mean_selected_feature_count_std";
    }

    private static String summaryRow(List<RunArtifacts> rs) {
        RunResult any = rs.get(0).result;
        double[] inst = doubles(rs, r -> (double) r.result.instances);
        double[] acc  = doubles(rs, r -> r.result.accuracy);
        double[] kap  = doubles(rs, r -> r.result.kappa);
        double[] kper = doubles(rs, r -> r.result.kappaPer);
        double[] dc   = doubles(rs, r -> (double) r.result.driftCount);
        double[] rh   = doubles(rs, r -> r.result.ramHoursGB);
        double[] peak = doubles(rs, r -> r.result.peakMB);
        double[] thr  = doubles(rs, r -> r.result.throughput);
        double[] wall = doubles(rs, r -> (double) r.result.wallMillis);
        double[] xk   = doubles(rs, r -> (double) r.result.extKeepCount);
        double[] xf   = doubles(rs, r -> (double) r.result.extFullCount);
        double[] dak  = doubles(rs, r -> (double) r.result.daSrpKept);
        double[] das  = doubles(rs, r -> (double) r.result.daSrpSurgical);
        double[] daf  = doubles(rs, r -> (double) r.result.daSrpFull);
        double[] dan  = doubles(rs, r -> (double) r.result.daSrpNoReplacement);

        double[] tkap = doubles(rs, r -> r.recorder == null ? Double.NaN : r.recorder.meanTemporalKappa());
        double[] recT = doubles(rs, r -> r.recorder == null ? Double.NaN : r.recorder.meanRecoveryLength());
        double[] stab = doubles(rs, r -> r.recorder == null ? Double.NaN : r.recorder.averageStabilityRatio());
        double[] sch  = doubles(rs, r -> r.recorder == null ? Double.NaN : (double) r.recorder.selectionChangeCount());
        double[] dal  = doubles(rs, r -> r.recorder == null ? Double.NaN : (double) r.recorder.driftAlarms.size());
        double[] msel = doubles(rs, r -> r.recorder == null ? Double.NaN : r.recorder.meanSelectedFeatureCount());

        return String.format(Locale.ROOT,
                "%s,%s,%s,%s,%s,%s,%d,%.0f,"
                        + "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                        + "%.2f,%.2f,%.6f,%.6f,"
                        + "%.1f,%.2f,%.0f,"
                        + "%.2f,%.2f,"
                        + "%.2f,%.2f,%.2f,%.2f,"
                        + "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s",
                any.blockId, any.dataset, any.variant, any.model, any.selector, any.detector,
                rs.size(), mean(inst),
                mean(acc), std(acc), mean(kap), std(kap), mean(kper), std(kper),
                mean(dc), std(dc), mean(rh), std(rh),
                mean(peak), mean(thr), mean(wall),
                mean(xk), mean(xf),
                mean(dak), mean(das), mean(daf), mean(dan),
                fmt(meanNan(tkap)), fmt(stdNan(tkap)),
                fmt(meanNan(recT)), fmt(stdNan(recT)),
                fmt(meanNan(stab)), fmt(stdNan(stab)),
                fmt(meanNan(sch)),  fmt(stdNan(sch)),
                fmt(meanNan(dal)),  fmt(stdNan(dal)),
                fmt(meanNan(msel)), fmt(stdNan(msel)));
    }

    // ========================================================================
    // Output: detailed per-block CSVs (windows, drift_alarms, …)
    // ========================================================================

    private static void writeDetailedPerBlockCsvs(Cfg cfg, List<RunArtifacts> sorted) throws Exception {
        Map<String, List<RunArtifacts>> byBlock = new LinkedHashMap<>();
        for (RunArtifacts ra : sorted) {
            if (!ra.ok() || ra.recorder == null) continue;
            byBlock.computeIfAbsent(ra.result.blockId, k -> new ArrayList<>()).add(ra);
        }
        for (Map.Entry<String, List<RunArtifacts>> e : byBlock.entrySet()) {
            Path folder = blockFolder(cfg, e.getKey());
            Files.createDirectories(folder);
            List<RunArtifacts> runs = e.getValue();
            if (cfg.writeWindowMetrics)      writeWindows(folder.resolve("windows.csv"), runs);
            if (cfg.writeDriftAlarms)        writeDriftAlarms(folder.resolve("drift_alarms.csv"), runs);
            if (cfg.writeFeatureSelections)  writeFeatureSelections(folder.resolve("feature_selections.csv"), runs);
            if (cfg.writeFeatureImportance)  writeFeatureImportance(folder.resolve("feature_importance.csv"), runs);
            if (cfg.writeRecoveryTime)       writeRecoveryTime(folder.resolve("recovery_time.csv"), runs);
            if (cfg.writeAdaptationEvents)   writeAdaptationEvents(folder.resolve("adaptation_events.csv"), runs);
            System.out.println("[Unified] detailed " + e.getKey() + " -> " + folder);
        }
    }

    private static void writeWindows(Path out, List<RunArtifacts> runs) throws Exception {
        try (PrintWriter w = new PrintWriter(new FileWriter(out.toFile()))) {
            w.println("block,dataset,variant,model,selector,detector,seed,"
                    + "window_id,start_instance,end_instance,"
                    + "accuracy,kappa,kappa_per,temporal_kappa,"
                    + "ram_hours_gb,peak_mb,throughput,"
                    + "drift_count_in_window,total_drift_count");
            for (RunArtifacts ra : runs) {
                RunResult r = ra.result;
                for (RunDetailedRecorder.WindowRow row : ra.recorder.windows) {
                    w.printf(Locale.ROOT,
                            "%s,%s,%s,%s,%s,%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%s,%.6f,%.2f,%.2f,%d,%d%n",
                            r.blockId, r.dataset, r.variant, r.model, r.selector, r.detector,
                            r.seed, row.windowId, row.startInstance, row.endInstance,
                            row.accuracy, row.kappa, row.kappaPer,
                            fmt(row.temporalKappa),
                            row.ramHoursGB, row.peakMB, row.throughput,
                            row.driftCountInWindow, row.totalDriftCount);
                }
            }
        }
    }

    private static void writeDriftAlarms(Path out, List<RunArtifacts> runs) throws Exception {
        try (PrintWriter w = new PrintWriter(new FileWriter(out.toFile()))) {
            w.println("block,dataset,variant,model,selector,detector,seed,"
                    + "instance_index,global_alarm,drifting_features,num_drifting_features,"
                    + "error_at_alarm,window_accuracy_before,window_accuracy_after");
            for (RunArtifacts ra : runs) {
                RunResult r = ra.result;
                List<RunDetailedRecorder.DriftAlarmRow> rows = new ArrayList<>(ra.recorder.driftAlarms);
                rows.sort(Comparator.comparingLong(a -> a.instanceIndex));
                for (RunDetailedRecorder.DriftAlarmRow row : rows) {
                    w.printf(Locale.ROOT,
                            "%s,%s,%s,%s,%s,%s,%d,%d,%d,%s,%d,%s,%s,%s%n",
                            r.blockId, r.dataset, r.variant, r.model, r.selector, r.detector,
                            r.seed, row.instanceIndex,
                            row.globalAlarm ? 1 : 0,
                            joinPipe(row.driftingFeatures), row.numDriftingFeatures,
                            fmt(row.errorAtAlarm),
                            fmt(row.windowAccuracyBefore), fmt(row.windowAccuracyAfter));
                }
            }
        }
    }

    private static void writeFeatureSelections(Path out, List<RunArtifacts> runs) throws Exception {
        try (PrintWriter w = new PrintWriter(new FileWriter(out.toFile()))) {
            w.println("block,dataset,variant,model,selector,detector,seed,"
                    + "instance_index,trigger_type,selected_features,selected_feature_count,"
                    + "changed_features,jaccard_to_previous,stability_ratio");
            for (RunArtifacts ra : runs) {
                RunResult r = ra.result;
                List<RunDetailedRecorder.SelectionRow> rows = new ArrayList<>(ra.recorder.selections);
                rows.sort(Comparator.comparingLong(a -> a.instanceIndex));
                for (RunDetailedRecorder.SelectionRow row : rows) {
                    w.printf(Locale.ROOT,
                            "%s,%s,%s,%s,%s,%s,%d,%d,%s,%s,%d,%s,%s,%s%n",
                            r.blockId, r.dataset, r.variant, r.model, r.selector, r.detector,
                            r.seed, row.instanceIndex,
                            row.triggerType,
                            joinPipe(row.selectedFeatures), row.selectedFeatureCount,
                            joinPipe(row.changedFeatures),
                            fmt(row.jaccardToPrevious),
                            fmt(row.stabilityRatio));
                }
            }
        }
    }

    private static void writeFeatureImportance(Path out, List<RunArtifacts> runs) throws Exception {
        try (PrintWriter w = new PrintWriter(new FileWriter(out.toFile()))) {
            w.println("block,dataset,variant,model,selector,detector,seed,"
                    + "instance_index,feature_index,importance,rank,is_selected,is_drifting");
            for (RunArtifacts ra : runs) {
                RunResult r = ra.result;
                List<RunDetailedRecorder.ImportanceRow> rows = new ArrayList<>(ra.recorder.importanceSnapshots);
                rows.sort((a, b) -> {
                    int c = Long.compare(a.instanceIndex, b.instanceIndex);
                    return c != 0 ? c : Integer.compare(a.featureIndex, b.featureIndex);
                });
                for (RunDetailedRecorder.ImportanceRow row : rows) {
                    w.printf(Locale.ROOT,
                            "%s,%s,%s,%s,%s,%s,%d,%d,%d,%s,%d,%d,%d%n",
                            r.blockId, r.dataset, r.variant, r.model, r.selector, r.detector,
                            r.seed, row.instanceIndex, row.featureIndex,
                            fmt(row.importance), row.rank,
                            row.isSelected ? 1 : 0, row.isDrifting ? 1 : 0);
                }
            }
        }
    }

    private static void writeRecoveryTime(Path out, List<RunArtifacts> runs) throws Exception {
        try (PrintWriter w = new PrintWriter(new FileWriter(out.toFile()))) {
            w.println("block,dataset,variant,model,selector,detector,seed,"
                    + "drift_id,drift_instance,recovered_instance,recovery_length,"
                    + "baseline_accuracy_before_drift,threshold_accuracy,"
                    + "max_drop,area_under_recovery_curve");
            for (RunArtifacts ra : runs) {
                RunResult r = ra.result;
                List<RunDetailedRecorder.RecoveryRow> rows = new ArrayList<>(ra.recorder.recoveries);
                rows.sort(Comparator.comparingLong(a -> a.driftInstance));
                for (RunDetailedRecorder.RecoveryRow row : rows) {
                    w.printf(Locale.ROOT,
                            "%s,%s,%s,%s,%s,%s,%d,%d,%d,%d,%d,%s,%s,%s,%s%n",
                            r.blockId, r.dataset, r.variant, r.model, r.selector, r.detector,
                            r.seed, row.driftId, row.driftInstance,
                            row.recoveredInstance, row.recoveryLength,
                            fmt(row.baselineAccuracyBeforeDrift),
                            fmt(row.thresholdAccuracy),
                            fmt(row.maxDrop),
                            fmt(row.areaUnderRecoveryCurve));
                }
            }
        }
    }

    private static void writeAdaptationEvents(Path out, List<RunArtifacts> runs) throws Exception {
        try (PrintWriter w = new PrintWriter(new FileWriter(out.toFile()))) {
            w.println("block,dataset,variant,model,selector,detector,seed,"
                    + "instance_index,event_type,"
                    + "kept_count,surgical_count,full_replacement_count,no_replacement_count,"
                    + "ext_keep_count,ext_full_count");
            for (RunArtifacts ra : runs) {
                RunResult r = ra.result;
                List<RunDetailedRecorder.AdaptationRow> rows = new ArrayList<>(ra.recorder.adaptations);
                rows.sort(Comparator.comparingLong(a -> a.instanceIndex));
                for (RunDetailedRecorder.AdaptationRow row : rows) {
                    w.printf(Locale.ROOT,
                            "%s,%s,%s,%s,%s,%s,%d,%d,%s,%d,%d,%d,%d,%d,%d%n",
                            r.blockId, r.dataset, r.variant, r.model, r.selector, r.detector,
                            r.seed, row.instanceIndex, row.eventType,
                            row.keptCount, row.surgicalCount,
                            row.fullReplacementCount, row.noReplacementCount,
                            row.extKeepCount, row.extFullCount);
                }
            }
        }
    }

    // ========================================================================
    // Output: per-block statistical tests
    // ========================================================================

    private static void writeStatisticalTests(Cfg cfg, List<RunArtifacts> sorted) throws Exception {
        Map<String, List<RunArtifacts>> byBlock = new LinkedHashMap<>();
        for (RunArtifacts ra : sorted) {
            if (!ra.ok()) continue;
            byBlock.computeIfAbsent(ra.result.blockId, k -> new ArrayList<>()).add(ra);
        }
        if (byBlock.isEmpty()) {
            System.err.println("[Unified] no OK runs — skipping statistical analysis.");
            return;
        }
        BlockStatisticalAnalysis analyser = new BlockStatisticalAnalysis(cfg.statisticalAlpha);
        for (Map.Entry<String, List<RunArtifacts>> e : byBlock.entrySet()) {
            Path folder = blockFolder(cfg, e.getKey()).resolve("stat_tests");
            try {
                analyser.analyseBlock(e.getKey(), e.getValue(), folder);
                System.out.println("[Unified] stat tests " + e.getKey() + " -> " + folder);
            } catch (Exception ex) {
                System.err.println("[Unified] stat tests for " + e.getKey() + " failed: " + ex);
            }
        }
    }

    // ========================================================================
    // Numeric / formatting helpers
    // ========================================================================

    private static String joinPipe(int[] vs) {
        if (vs == null || vs.length == 0) return "";
        StringJoiner sj = new StringJoiner("|");
        for (int v : vs) sj.add(Integer.toString(v));
        return sj.toString();
    }

    @FunctionalInterface
    private interface ToDouble { double apply(RunArtifacts r); }

    private static double[] doubles(List<RunArtifacts> rs, ToDouble f) {
        double[] out = new double[rs.size()];
        for (int i = 0; i < rs.size(); i++) out[i] = f.apply(rs.get(i));
        return out;
    }

    private static double mean(double[] x) {
        if (x.length == 0) return 0.0;
        double s = 0.0;
        for (double v : x) s += v;
        return s / x.length;
    }

    private static double std(double[] x) {
        if (x.length < 2) return 0.0;
        double m = mean(x), s = 0.0;
        for (double v : x) s += (v - m) * (v - m);
        return Math.sqrt(s / (x.length - 1));
    }

    private static double meanNan(double[] x) {
        double s = 0.0; int n = 0;
        for (double v : x) if (Double.isFinite(v)) { s += v; n++; }
        return n == 0 ? Double.NaN : s / n;
    }

    private static double stdNan(double[] x) {
        double m = meanNan(x);
        if (Double.isNaN(m)) return Double.NaN;
        double s = 0.0; int n = 0;
        for (double v : x) if (Double.isFinite(v)) { s += (v - m) * (v - m); n++; }
        return n < 2 ? 0.0 : Math.sqrt(s / (n - 1));
    }

    private static double[] invertPValues(double[] pValues, int expectedLength) {
        double[] out = new double[expectedLength];
        if (pValues == null || pValues.length != expectedLength) return out;
        for (int i = 0; i < expectedLength; i++) {
            double p = pValues[i];
            if (!Double.isFinite(p)) p = 1.0;
            if (p < 0.0) p = 0.0;
            if (p > 1.0) p = 1.0;
            out[i] = 1.0 - p;
        }
        return out;
    }

    private static String fmt(double v) {
        return Double.isNaN(v) ? "NaN" : String.format(Locale.ROOT, "%.6f", v);
    }
}
