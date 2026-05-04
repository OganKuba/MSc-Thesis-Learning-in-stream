package thesis.pipeline;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.streams.InstanceStream;
import thesis.detection.TwoLevelDriftDetector;
import thesis.discretization.PiDDiscretizer;
import thesis.models.FeatureImportance;
import thesis.models.ModelWrapper;
import thesis.selection.FeatureSelector;
import thesis.selection.FilterRanker;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class ExperimentRunner {

    public static final class Variant {
        public String name;
        public String model;
        public String selector;
        public String detector = "TWO_LEVEL";
    }

    public static final class Config {
        public String experimentGroup;
        public List<String> datasets = new ArrayList<>();
        public List<Variant> variants = new ArrayList<>();
        public List<Integer> seeds = new ArrayList<>();
        public String outputDir = "results/";
        public int warmup = 1500;
        public int logEvery = 1000;
        public int refreshEvery = 0;
        public long maxInstances = Long.MAX_VALUE;
        public boolean verbose = false;
        public boolean continueOnError = true;
        public boolean strict = false;
    }

    public interface StreamFactory   { InstanceStream create(String dataset, int seed); }
    public interface SelectorFactory { FeatureSelector create(String name, int numFeatures, int numClasses, int seed); }
    public interface ModelFactory    { ModelWrapper create(String name, FeatureSelector sel,
                                                           InstancesHeader header,
                                                           int numClasses, int seed); }
    public interface DetectorFactory { TwoLevelDriftDetector create(String name, int numFeatures); }
    public interface RankerFactory   { FilterRanker create(int numFeatures, int numClasses, int seed); }

    private final StreamFactory streamFactory;
    private final SelectorFactory selectorFactory;
    private final ModelFactory modelFactory;
    private final DetectorFactory detectorFactory;
    private final RankerFactory rankerFactory;

    private final Map<String, String> seenSelectorClasses = new LinkedHashMap<>();
    private final Map<String, String> seenModelClasses = new LinkedHashMap<>();
    private final Map<String, String> seenDetectorClasses = new LinkedHashMap<>();

    private long completed;
    private long failed;

    public ExperimentRunner(StreamFactory sf, SelectorFactory selF,
                            ModelFactory mf, DetectorFactory df) {
        this(sf, selF, mf, df, null);
    }

    public ExperimentRunner(StreamFactory sf, SelectorFactory selF,
                            ModelFactory mf, DetectorFactory df,
                            RankerFactory rf) {
        if (sf == null || selF == null || mf == null || df == null) {
            throw new IllegalArgumentException("factories must not be null");
        }
        this.streamFactory = sf;
        this.selectorFactory = selF;
        this.modelFactory = mf;
        this.detectorFactory = df;
        this.rankerFactory = rf;
    }

    public void runAll(Config cfg) throws IOException {
        validateConfig(cfg);
        for (String dataset : cfg.datasets) {
            for (Variant v : cfg.variants) {
                for (int seed : cfg.seeds) {
                    try {
                        runOne(cfg, dataset, v, seed);
                        completed++;
                    } catch (Throwable t) {
                        failed++;
                        System.err.printf("[FAIL] %s | %s | %s | seed=%d → %s%n",
                                cfg.experimentGroup, dataset, v.name, seed, t);
                        if (!cfg.continueOnError) throw new RuntimeException(t);
                    }
                }
            }
        }
        System.out.printf("[runAll] completed=%d  failed=%d%n", completed, failed);
        verifyVariantDiversity(cfg);
    }

    private static void validateConfig(Config cfg) {
        if (cfg == null) throw new IllegalArgumentException("cfg == null");
        if (cfg.datasets == null || cfg.datasets.isEmpty())
            throw new IllegalArgumentException("datasets empty");
        if (cfg.variants == null || cfg.variants.isEmpty())
            throw new IllegalArgumentException("variants empty");
        if (cfg.seeds == null || cfg.seeds.isEmpty())
            throw new IllegalArgumentException("seeds empty");
        if (cfg.warmup < 1) throw new IllegalArgumentException("warmup must be >= 1");
        if (cfg.logEvery < 1) throw new IllegalArgumentException("logEvery must be >= 1");
        java.util.Set<String> names = new java.util.HashSet<>();
        for (Variant v : cfg.variants) {
            if (v.name == null || v.name.isBlank())
                throw new IllegalArgumentException("variant.name empty");
            if (v.model == null || v.model.isBlank())
                throw new IllegalArgumentException("variant.model empty for " + v.name);
            if (v.selector == null || v.selector.isBlank())
                throw new IllegalArgumentException("variant.selector empty for " + v.name);
            if (!names.add(v.name))
                throw new IllegalArgumentException("duplicate variant name: " + v.name);
        }
    }

    private void runOne(Config cfg, String dataset, Variant v, int seed) throws IOException {
        Path outFile = Paths.get(cfg.outputDir,
                cfg.experimentGroup, dataset, v.name, "seed_" + seed + ".csv");
        Files.createDirectories(outFile.getParent());

        InstanceStream source = streamFactory.create(dataset, seed);
        if (source == null) throw new IllegalStateException("streamFactory returned null");
        if (source instanceof moa.options.OptionHandler) {
            ((moa.options.OptionHandler) source).prepareForUse();
        }
        InstancesHeader header = source.getHeader();
        if (header == null) throw new IllegalStateException("source header is null");
        int numFeatures = header.numAttributes() - 1;
        int numClasses  = header.numClasses();
        if (numFeatures < 1) throw new IllegalStateException("numFeatures < 1");
        if (numClasses < 2)  throw new IllegalStateException("numClasses < 2");

        FeatureSelector selector = selectorFactory.create(v.selector, numFeatures, numClasses, seed);
        if (selector == null) throw new IllegalStateException("selectorFactory returned null for " + v.selector);

        ModelWrapper model = modelFactory.create(v.model, selector, header, numClasses, seed);
        if (model == null) throw new IllegalStateException("modelFactory returned null for " + v.model);

        TwoLevelDriftDetector detector = detectorFactory.create(v.detector, numFeatures);
        if (detector == null) throw new IllegalStateException("detectorFactory returned null for " + v.detector);

        recordClass(seenSelectorClasses, v.selector, selector.getClass().getName());
        recordClass(seenModelClasses, v.model, model.getClass().getName());
        recordClass(seenDetectorClasses, v.detector, detector.getClass().getName());

        PiDDiscretizer pid = new PiDDiscretizer(numFeatures, numClasses);
        FilterRanker fullRanker = rankerFactory == null
                ? null : rankerFactory.create(numFeatures, numClasses, seed);
        FeatureImportance importance = new FeatureImportance(numFeatures);

        System.out.printf("▶ %s | ds=%s var=%s seed=%d → %s%n",
                cfg.experimentGroup, dataset, v.name, seed, outFile);
        System.out.printf("    selector=%s  model=%s  detector=%s  F=%d C=%d K=%d%n",
                selector.getClass().getSimpleName(),
                model.getClass().getSimpleName(),
                detector.getClass().getSimpleName(),
                numFeatures, numClasses, selector.getK());

        try (BufferedWriter w = Files.newBufferedWriter(outFile)) {
            RecordingMetrics metrics = new RecordingMetrics(
                    w, cfg.logEvery, selector, numClasses,
                    dataset, v.name, model.name());

            StreamPipeline pipeline = StreamPipeline.builder()
                    .source(source)
                    .selector(selector)
                    .model(model)
                    .detector(detector)
                    .metrics(metrics)
                    .rankingPid(pid)
                    .fullRanker(fullRanker)
                    .importance(importance)
                    .warmup(cfg.warmup)
                    .logEvery(cfg.logEvery)
                    .refreshEvery(cfg.refreshEvery)
                    .maxInstances(cfg.maxInstances)
                    .verbose(cfg.verbose)
                    .strict(cfg.strict)
                    .build();

            pipeline.run();

            System.out.printf("    done: processed=%d  selection=%s%n",
                    pipeline.getProcessed(),
                    java.util.Arrays.toString(selector.getCurrentSelection()));
        }
    }

    private static void recordClass(Map<String, String> map, String key, String cls) {
        String prev = map.get(key);
        if (prev != null && !prev.equals(cls)) {
            System.err.printf("[WARN] %s mapped to multiple classes: %s vs %s%n", key, prev, cls);
        }
        map.put(key, cls);
    }

    private void verifyVariantDiversity(Config cfg) {
        java.util.Set<String> distinct = new java.util.HashSet<>(seenSelectorClasses.values());
        if (cfg.variants.size() > 1 && distinct.size() == 1) {
            System.err.printf("[WARN] all selectors mapped to same class: %s — S1=S2=S3=S4 likely bug%n",
                    distinct.iterator().next());
        }
        if (cfg.verbose || cfg.strict) {
            System.out.println("[diversity] selector mapping:");
            seenSelectorClasses.forEach((k, v) -> System.out.printf("    %s → %s%n", k, v));
            System.out.println("[diversity] model mapping:");
            seenModelClasses.forEach((k, v) -> System.out.printf("    %s → %s%n", k, v));
            System.out.println("[diversity] detector mapping:");
            seenDetectorClasses.forEach((k, v) -> System.out.printf("    %s → %s%n", k, v));
        }
        if (cfg.strict && cfg.variants.size() > 1 && distinct.size() == 1) {
            throw new IllegalStateException("strict: all selectors collapsed to same class");
        }
    }

    public static Config loadConfig(Path path) throws IOException {
        ObjectMapper m = new ObjectMapper();
        JsonNode root = m.readTree(path.toFile());

        Config c = new Config();
        c.experimentGroup = root.path("experiment_group").asText("E?");
        root.path("datasets").forEach(n -> c.datasets.add(n.asText()));
        root.path("seeds").forEach(n -> c.seeds.add(n.asInt()));
        c.outputDir       = root.path("output_dir").asText(c.outputDir);
        c.warmup          = root.path("warmup").asInt(c.warmup);
        c.logEvery        = root.path("log_every").asInt(c.logEvery);
        c.refreshEvery    = root.path("refresh_every").asInt(c.refreshEvery);
        c.maxInstances    = root.path("max_instances").asLong(c.maxInstances);
        c.verbose         = root.path("verbose").asBoolean(c.verbose);
        c.continueOnError = root.path("continue_on_error").asBoolean(c.continueOnError);
        c.strict          = root.path("strict").asBoolean(c.strict);

        for (Iterator<JsonNode> it = root.path("model_variants").elements(); it.hasNext(); ) {
            JsonNode n = it.next();
            Variant v = new Variant();
            v.name     = n.path("name").asText();
            v.model    = n.path("model").asText();
            v.selector = n.path("selector").asText();
            v.detector = n.path("detector").asText(v.detector);
            c.variants.add(v);
        }
        return c;
    }

    public long getCompleted() { return completed; }
    public long getFailed() { return failed; }
    public Map<String, String> getSeenSelectorClasses() { return new LinkedHashMap<>(seenSelectorClasses); }
    public Map<String, String> getSeenModelClasses() { return new LinkedHashMap<>(seenModelClasses); }
    public Map<String, String> getSeenDetectorClasses() { return new LinkedHashMap<>(seenDetectorClasses); }
}