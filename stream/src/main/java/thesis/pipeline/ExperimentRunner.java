package thesis.pipeline;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import moa.streams.InstanceStream;
import thesis.detection.TwoLevelDriftDetector;
import thesis.models.ModelWrapper;
import thesis.selection.FeatureSelector;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Loads a JSON experiment config and runs (dataset × variant × seed) sweeps,
 * writing one CSV per run:
 *   {output_dir}/{group}/{dataset}/{variant}/seed_{s}.csv
 *
 * CSV columns:
 *   instance_num, kappa, kappa_per, accuracy,
 *   ram_hours, feature_stability_ratio, drift_count, recovery_time
 */
public final class ExperimentRunner {

    /* ---------- config DTOs ---------- */

    public static final class Variant {
        public String name;
        public String model;
        public String selector;
        public String detector = "TWO_LEVEL"; // default
    }

    public static final class Config {
        public String experimentGroup;
        public List<String> datasets = new ArrayList<>();
        public List<Variant> variants = new ArrayList<>();
        public List<Integer> seeds = new ArrayList<>();
        public String outputDir = "results/";
        public int warmup = 1500;
        public int logEvery = 1000;       // also = CSV sampling interval
        public int refreshEvery = 0;
        public long maxInstances = Long.MAX_VALUE;
        public boolean verbose = false;
    }

    /* ---------- entrypoint ---------- */

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.err.println("usage: ExperimentRunner <config.json>");
            System.exit(2);
        }
        Config cfg = loadConfig(Paths.get(args[0]));
        new ExperimentRunner().runAll(cfg);
    }

    public void runAll(Config cfg) throws IOException {
        for (String dataset : cfg.datasets) {
            for (Variant v : cfg.variants) {
                for (int seed : cfg.seeds) {
                    runOne(cfg, dataset, v, seed);
                }
            }
        }
    }

    /* ---------- single run ---------- */

    private void runOne(Config cfg, String dataset, Variant v, int seed) throws IOException {
        Path outFile = Paths.get(cfg.outputDir,
                cfg.experimentGroup, dataset, v.name, "seed_" + seed + ".csv");
        Files.createDirectories(outFile.getParent());

        System.out.printf("▶ %s | %s | %s | seed=%d → %s%n",
                cfg.experimentGroup, dataset, v.name, seed, outFile);

        try (BufferedWriter w = Files.newBufferedWriter(outFile)) {
            w.write("instance_num,kappa,kappa_per,accuracy,ram_hours,"
                    + "feature_stability_ratio,drift_count,recovery_time");
            w.newLine();

            // Build all components for this run (factories you wire to real classes):
            InstanceStream         source   = DatasetFactory.create(dataset, seed);
            FeatureSelector        selector = SelectorFactory.create(v.selector, seed);
            ModelWrapper           model    = ModelFactory.create(v.model, seed);
            TwoLevelDriftDetector  detector = DetectorFactory.create(v.detector, seed);

            RecordingMetrics metrics = new RecordingMetrics(w, cfg.logEvery, selector);

            StreamPipeline pipeline = StreamPipeline.builder()
                    .source(source)
                    .selector(selector)
                    .model(model)
                    .detector(detector)
                    .metrics(metrics)
                    // Optional add-ons — uncomment once you have factories for them:
                    // .rankingPid(new PiDDiscretizer(...))
                    // .fullRanker(new FilterRanker(...))
                    // .importance(new FeatureImportance(...))
                    .warmup(cfg.warmup)
                    .logEvery(cfg.logEvery)
                    .refreshEvery(cfg.refreshEvery)
                    .maxInstances(cfg.maxInstances)
                    .verbose(cfg.verbose)
                    .build();

            // Tell the recorder where to pull "drift_count" snapshots from:
            metrics.bindPipeline(pipeline, detector);

            pipeline.run();

            // Always emit a final row, even if maxInstances isn't a multiple of logEvery:
            metrics.flushFinalRow();
        }
    }

    /* ---------- JSON loading ---------- */

    private static Config loadConfig(Path path) throws IOException {
        ObjectMapper m = new ObjectMapper();
        JsonNode root = m.readTree(path.toFile());

        Config c = new Config();
        c.experimentGroup = root.path("experiment_group").asText("E?");
        root.path("datasets").forEach(n -> c.datasets.add(n.asText()));
        root.path("seeds").forEach(n -> c.seeds.add(n.asInt()));
        c.outputDir    = root.path("output_dir").asText(c.outputDir);
        c.warmup       = root.path("warmup").asInt(c.warmup);
        c.logEvery     = root.path("log_every").asInt(c.logEvery);
        c.refreshEvery = root.path("refresh_every").asInt(c.refreshEvery);
        c.maxInstances = root.path("max_instances").asLong(c.maxInstances);
        c.verbose      = root.path("verbose").asBoolean(c.verbose);

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
}