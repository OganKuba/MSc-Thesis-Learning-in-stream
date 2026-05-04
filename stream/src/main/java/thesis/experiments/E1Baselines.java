package thesis.experiments;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.options.OptionHandler;
import moa.streams.ArffFileStream;
import moa.streams.InstanceStream;
import thesis.evaluation.MetricsCollector;
import thesis.models.ARFWrapper;
import thesis.models.FeatureSpace;
import thesis.models.HoeffdingTreeWrapper;
import thesis.models.MajorityClassWrapper;
import thesis.models.ModelWrapper;
import thesis.models.NoChangeWrapper;
import thesis.models.SRPWrapper;
import thesis.pipeline.SyntheticStreamFactory;
import thesis.selection.StaticFeatureSelector;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public class E1Baselines {

    static final class DatasetSpec {
        String name;
        String type = "synthetic";
        String generator;
        String path;
        int n = 100_000;
        int noiseFeatures = 0;
        double sigma = 0.01;
        double speed = 0.001;
        int driftFeatures = 5;
        long maxInstances = -1;
    }

    static final class E1Config {
        String experimentGroup = "E1_baselines";
        String outputDir = "results/E1_baselines";
        int warmup = 1500;
        int windowSize = 1000;
        int logEvery = 1000;
        int ramSampleEvery = 200;
        long maxInstances = Long.MAX_VALUE;
        boolean skipMissingArff = true;
        boolean realDatasetsReadAll = true;
        List<DatasetSpec> datasets = new ArrayList<>();
        List<String> models = new ArrayList<>();
        List<Integer> seeds = new ArrayList<>();

        static E1Config load(Path path) throws Exception {
            ObjectMapper m = new ObjectMapper();
            JsonNode r = m.readTree(path.toFile());
            E1Config c = new E1Config();
            c.experimentGroup = r.path("experiment_group").asText(c.experimentGroup);
            c.outputDir       = r.path("output_dir").asText(c.outputDir);
            c.warmup          = r.path("warmup").asInt(c.warmup);
            c.windowSize      = r.path("window_size").asInt(c.windowSize);
            c.logEvery        = r.path("log_every").asInt(c.logEvery);
            c.ramSampleEvery  = r.path("ram_sample_every").asInt(c.ramSampleEvery);
            c.maxInstances    = r.path("max_instances").asLong(c.maxInstances);
            c.skipMissingArff = r.path("skip_missing_arff").asBoolean(c.skipMissingArff);
            c.realDatasetsReadAll = r.path("real_datasets_read_all").asBoolean(c.realDatasetsReadAll);
            r.path("seeds").forEach(n -> c.seeds.add(n.asInt()));
            r.path("models").forEach(n -> c.models.add(n.asText()));
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
                ds.maxInstances   = d.path("max_instances").asLong(ds.maxInstances);
                c.datasets.add(ds);
            }
            if (c.seeds.isEmpty())    throw new IllegalArgumentException("seeds empty");
            if (c.models.isEmpty())   throw new IllegalArgumentException("models empty");
            if (c.datasets.isEmpty()) throw new IllegalArgumentException("datasets empty");
            return c;
        }
    }

    public static void main(String[] args) throws Exception {

        Path configPath = args.length > 0
                ? Paths.get(args[0])
                : Paths.get("src/main/java/thesis/experiments/E1_baselines.json");

        if (!Files.exists(configPath)) {
            throw new RuntimeException(
                    "Config not found: " + configPath.toAbsolutePath() +
                            "\nWorking dir: " + Paths.get(".").toAbsolutePath()
            );
        }

        E1Config cfg = E1Config.load(configPath);
        new E1Baselines().run(cfg);
    }

    static long effectiveMaxInstances(E1Config cfg, DatasetSpec ds) {
        if (ds.maxInstances > 0) return ds.maxInstances;
        if ("arff".equalsIgnoreCase(ds.type) && cfg.realDatasetsReadAll) return Long.MAX_VALUE;
        return cfg.maxInstances;
    }

    public void run(E1Config cfg) throws Exception {
        Files.createDirectories(Paths.get(cfg.outputDir));

        int validDatasets = 0;
        for (DatasetSpec ds : cfg.datasets) {
            if ("arff".equalsIgnoreCase(ds.type)
                    && (ds.path == null || !Files.exists(Paths.get(ds.path)))) {
                if (cfg.skipMissingArff) continue;
            }
            validDatasets++;
        }
        int runTotal = validDatasets * cfg.seeds.size() * cfg.models.size();
        int runIdx = 0;

        try (PrintWriter csv = new PrintWriter(new FileWriter(
                Paths.get(cfg.outputDir, "E1_results.csv").toFile()));
             PrintWriter sum = new PrintWriter(new FileWriter(
                     Paths.get(cfg.outputDir, "E1_summary.csv").toFile()));
             PrintWriter drifts = new PrintWriter(new FileWriter(
                     Paths.get(cfg.outputDir, cfg.experimentGroup + "_drifts.csv").toFile()))) {

            csv.println("dataset,model,selector,seed,instance_num,"
                    + "selected_count,model_num_attributes,"
                    + "accuracy_window,kappa_window,kappa_per_window,"
                    + "majority_baseline_window,nochange_baseline_window,ram_hours,"
                    + "drift_count,feature_stability,throughput_inst_per_sec,peak_ram_mb");

            sum.println("dataset,model,selector,seed,instance_num,"
                    + "accuracy_window,kappa_window,kappa_per_window,"
                    + "majority_baseline_window,nochange_baseline_window,ram_hours,"
                    + "drift_count,feature_stability,throughput_inst_per_sec,peak_ram_mb,status");

            drifts.println("dataset,variant,seed,alarm_at,kappa_before_500,"
                    + "kappa_after_500,recovery_instances,drift_type");
            drifts.flush();

            for (DatasetSpec ds : cfg.datasets) {
                if ("arff".equalsIgnoreCase(ds.type)
                        && (ds.path == null || !Files.exists(Paths.get(ds.path)))) {
                    System.err.printf("[E1] missing ARFF, skip: %s path=%s%n", ds.name, ds.path);
                    if (cfg.skipMissingArff) continue;
                    throw new java.io.FileNotFoundException(String.valueOf(ds.path));
                }
                for (int seed : cfg.seeds) {
                    for (String modelName : cfg.models) {
                        runIdx++;
                        try {
                            MetricsCollector.Snapshot[] majOut = new MetricsCollector.Snapshot[1];
                            MetricsCollector.Snapshot[] ncOut  = new MetricsCollector.Snapshot[1];
                            long[] nOut = new long[1];
                            MetricsCollector.Snapshot s = runOne(cfg, ds, modelName, seed, csv,
                                    runIdx, runTotal, majOut, ncOut, nOut);

                            double elapsedSec = s.elapsedHours * 3600.0;
                            double thr = elapsedSec > 0.0 ? (double) nOut[0] / elapsedSec : 0.0;
                            double stab = Double.isNaN(s.lastFeatureStabilityRatio)
                                    ? 0.0 : s.lastFeatureStabilityRatio;

                            sum.printf(Locale.ROOT,
                                    "%s,%s,S1,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%d,%.4f,%.2f,%.1f,OK%n",
                                    ds.name, modelName, seed, nOut[0],
                                    s.accuracyWindow, s.kappa, s.kappaPer,
                                    majOut[0].accuracyWindow,
                                    ncOut[0].accuracyWindow,
                                    s.ramHoursGB,
                                    s.driftCount, stab, thr, s.peakMB);
                            sum.flush();
                        } catch (Exception e) {
                            System.err.printf("[E1][FAIL] (%d/%d) %s|%s|seed=%d -> %s%n",
                                    runIdx, runTotal, ds.name, modelName, seed, e);
                            sum.printf(Locale.ROOT,
                                    "%s,%s,S1,%d,0,NaN,NaN,NaN,NaN,NaN,NaN,0,0.0000,0.00,0.0,FAIL%n",
                                    ds.name, modelName, seed);
                            sum.flush();
                        }
                    }
                }
            }
        }
        System.out.println("[E1] Done. Results -> " + cfg.outputDir + "/E1_results.csv");
        System.out.println("[E1] Summary  -> " + cfg.outputDir + "/E1_summary.csv");
        System.out.println("[E1] Drifts   -> " + cfg.outputDir + "/" + cfg.experimentGroup + "_drifts.csv");
    }


    private MetricsCollector.Snapshot runOne(E1Config cfg, DatasetSpec ds, String modelName,
                                             int seed, PrintWriter csv,
                                             int runIdx, int runTotal,
                                             MetricsCollector.Snapshot[] majOut,
                                             MetricsCollector.Snapshot[] ncOut,
                                             long[] nOut) throws Exception {

        long effMax = effectiveMaxInstances(cfg, ds);
        String capStr = (effMax == Long.MAX_VALUE) ? "ALL" : Long.toString(effMax);
        System.out.printf(Locale.ROOT, "[E1] (%d/%d) %s | %s | seed=%d | cap=%s ...%n",
                runIdx, runTotal, ds.name, modelName, seed, capStr);

        InstanceStream stream = buildStream(ds, seed);
        if (stream instanceof OptionHandler) ((OptionHandler) stream).prepareForUse();
        InstancesHeader header = stream.getHeader();
        int numFeatures = header.numAttributes() - 1;
        int numClasses  = header.numClasses();
        if (numFeatures < 1) throw new IllegalStateException("numFeatures < 1");
        if (numClasses  < 2) throw new IllegalStateException("numClasses < 2");

        StaticFeatureSelector selector = new StaticFeatureSelector(numFeatures, numClasses);
        FeatureSpace space = new FeatureSpace(header);

        int collected = 0;
        double[][] window = new double[cfg.warmup][];
        int[] labels = new int[cfg.warmup];
        while (collected < cfg.warmup && stream.hasMoreInstances()) {
            Instance x = stream.nextInstance().getData();
            window[collected] = space.extractFeatures(x);
            labels[collected] = (int) x.classValue();
            collected++;
        }
        if (collected == 0) throw new IllegalStateException("no instances during warmup");
        if (collected < cfg.warmup) {
            window = Arrays.copyOf(window, collected);
            labels = Arrays.copyOf(labels, collected);
        }
        selector.initialize(window, labels);

        ModelWrapper model    = buildModel(modelName, selector, header, numClasses);
        ModelWrapper majority = new MajorityClassWrapper(selector, numClasses);
        ModelWrapper noChange = new NoChangeWrapper(selector, numClasses);

        for (int i = 0; i < collected; i++) {
            majority.train(null, labels[i]);
            noChange.train(null, labels[i]);
        }

        MetricsCollector metrics = new MetricsCollector(numClasses, cfg.windowSize, cfg.logEvery, cfg.ramSampleEvery);
        MetricsCollector mMaj    = new MetricsCollector(numClasses, cfg.windowSize, cfg.logEvery, cfg.ramSampleEvery);
        MetricsCollector mNC     = new MetricsCollector(numClasses, cfg.windowSize, cfg.logEvery, cfg.ramSampleEvery);

        long n = collected;
        long lastLogTime = System.nanoTime();
        long lastLogN = n;

        while (stream.hasMoreInstances() && n < effMax) {
            Instance x = stream.nextInstance().getData();
            int yTrue = (int) x.classValue();

            long t0 = System.nanoTime();
            int yHat = model.predict(x);
            long elapsed = System.nanoTime() - t0;
            int yMaj = majority.predict(x);
            int yNC  = noChange.predict(x);

            metrics.update(yTrue, yHat, elapsed);
            mMaj.update(yTrue, yMaj, 0);
            mNC.update(yTrue, yNC, 0);

            model.train(x, yTrue);
            majority.train(x, yTrue);
            noChange.train(x, yTrue);

            n++;
            if (cfg.logEvery > 0 && n % cfg.logEvery == 0) {
                MetricsCollector.Snapshot s = metrics.snapshot();
                long now = System.nanoTime();
                double dtSec = (now - lastLogTime) / 1e9;
                double thr = dtSec > 0.0 ? (double) (n - lastLogN) / dtSec : 0.0;
                lastLogTime = now;
                lastLogN = n;
                double stab = Double.isNaN(s.lastFeatureStabilityRatio)
                        ? 0.0 : s.lastFeatureStabilityRatio;

                csv.printf(Locale.ROOT,
                        "%s,%s,S1,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f,%d,%.4f,%.2f,%.1f%n",
                        ds.name, modelName, seed, n,
                        selector.getCurrentSelection().length,
                        model.getCurrentSelection().length,
                        s.accuracyWindow, s.kappa, s.kappaPer,
                        mMaj.snapshot().accuracyWindow,
                        mNC.snapshot().accuracyWindow,
                        s.ramHoursGB,
                        s.driftCount, stab, thr, s.peakMB);
                csv.flush();

                if (effMax == Long.MAX_VALUE && n % (cfg.logEvery * 10L) == 0) {
                    System.out.printf(Locale.ROOT,
                            "[E1] (%d/%d) %s | %s | seed=%d | n=%d acc=%.4f k=%.4f thr=%.0f/s peak=%.0fMB%n",
                            runIdx, runTotal, ds.name, modelName, seed, n,
                            s.accuracyWindow, s.kappa, thr, s.peakMB);
                }
            }
        }

        MetricsCollector.Snapshot s = metrics.snapshot();
        MetricsCollector.Snapshot sMaj = mMaj.snapshot();
        MetricsCollector.Snapshot sNC  = mNC.snapshot();

        System.out.printf(Locale.ROOT,
                "[E1] (%d/%d) DONE %s | %s | seed=%d | n=%d | acc=%.4f kappa=%.4f (maj=%.4f nc=%.4f) peak=%.0fMB%n",
                runIdx, runTotal, ds.name, modelName, seed, n,
                s.accuracyWindow, s.kappa, sMaj.accuracyWindow, sNC.accuracyWindow, s.peakMB);

        majOut[0] = sMaj;
        ncOut[0]  = sNC;
        nOut[0]   = n;
        return s;
    }


    static InstanceStream buildStream(DatasetSpec ds, int seed) {
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

    static ModelWrapper buildModel(String name, StaticFeatureSelector sel,
                                   InstancesHeader header, int numClasses) {
        switch (name.toUpperCase(Locale.ROOT)) {
            case "HT":            return new HoeffdingTreeWrapper(sel, header);
            case "ARF":           return new ARFWrapper(sel, header);
            case "SRP":           return new SRPWrapper(sel, header);
            case "MAJORITY":
            case "MAJORITYCLASS": return new MajorityClassWrapper(sel, numClasses);
            case "NOCHANGE":      return new NoChangeWrapper(sel, numClasses);
            default: throw new IllegalArgumentException("Unknown model: " + name);
        }
    }
}
