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
import moa.streams.ArffFileStream;
import thesis.models.ARFWrapper;
import thesis.models.HoeffdingTreeWrapper;
import thesis.models.SRPWrapper;


import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public class E1PrototypeRunner {

    static final class DatasetSpec {
        String name;
        String type = "synthetic";
        String generator;
        String path;
        int n = 6000;
        int noiseFeatures = 0;
        double sigma = 0.01;
        double speed = 0.001;
        int driftFeatures = 5;
    }

    static final class Cfg {
        String outputDir = "results/E1_prototype";
        int warmup = 500;
        int windowSize = 500;
        int logEvery = 500;
        int ramSampleEvery = 200;
        long maxInstances = 5000;
        boolean skipMissingArff = true;
        List<DatasetSpec> datasets = new ArrayList<>();
        List<String> models = new ArrayList<>();
        List<Integer> seeds = new ArrayList<>();

        static Cfg load(Path path) throws Exception {
            ObjectMapper m = new ObjectMapper();
            JsonNode r = m.readTree(path.toFile());
            Cfg c = new Cfg();
            c.outputDir       = r.path("output_dir").asText(c.outputDir);
            c.warmup          = r.path("warmup").asInt(c.warmup);
            c.windowSize      = r.path("window_size").asInt(c.windowSize);
            c.logEvery        = r.path("log_every").asInt(c.logEvery);
            c.ramSampleEvery  = r.path("ram_sample_every").asInt(c.ramSampleEvery);
            c.maxInstances    = r.path("max_instances").asLong(c.maxInstances);
            c.skipMissingArff = r.path("skip_missing_arff").asBoolean(c.skipMissingArff);
            r.path("seeds").forEach(n -> c.seeds.add(n.asInt()));
            r.path("models").forEach(n -> c.models.add(n.asText()));
            for (JsonNode d : r.path("datasets")) {
                DatasetSpec ds = new DatasetSpec();
                ds.name          = d.path("name").asText();
                ds.type          = d.path("type").asText(ds.type);
                ds.generator     = d.path("generator").asText(null);
                ds.path          = d.path("path").asText(null);
                ds.n             = d.path("n").asInt(ds.n);
                ds.noiseFeatures = d.path("noise_features").asInt(ds.noiseFeatures);
                ds.sigma         = d.path("sigma").asDouble(ds.sigma);
                ds.speed         = d.path("speed").asDouble(ds.speed);
                ds.driftFeatures = d.path("drift_features").asInt(ds.driftFeatures);
                c.datasets.add(ds);
            }
            return c;
        }
    }

    static final class RunResult {
        String dataset, model;
        int seed, k, d, selectedCount, modelAttrs;
        double accModel, kappaModel, kappaPerModel;
        double accMajority, accNoChange;
        double ramHours;
        long instances;
        boolean kCorrect, filterReduces, modelSeesK, baselinesRan;
        boolean beatsMajority, beatsNoChange;
        boolean pass;
        String reason = "";
    }

    public static void main(String[] args) throws Exception {
        Path cfgPath = args.length > 0
                ? Paths.get(args[0])
                : Paths.get("src/main/java/thesis/experiments/e1_prototype.json");

        if (!Files.exists(cfgPath)) {
            throw new RuntimeException("Config not found: " + cfgPath.toAbsolutePath());
        }

        Cfg cfg = Cfg.load(cfgPath);
        new E1PrototypeRunner().run(cfg);
    }

    public void run(Cfg cfg) throws Exception {
        Files.createDirectories(Paths.get(cfg.outputDir));
        List<RunResult> all = new ArrayList<>();
        try (PrintWriter csv = new PrintWriter(new FileWriter(
                Paths.get(cfg.outputDir, "E1_prototype.csv").toFile()))) {

            csv.println("instance_num,dataset,model,selector,seed,"
                    + "selected_features,selected_count,model_num_attributes,"
                    + "accuracy_window,kappa_window,kappa_per_window,"
                    + "majority_baseline_window,nochange_baseline_window,ram_hours");

            for (DatasetSpec ds : cfg.datasets) {
                if ("arff".equalsIgnoreCase(ds.type)
                        && (ds.path == null || !Files.exists(Paths.get(ds.path)))) {
                    System.err.printf("[proto] missing ARFF, skip: %s path=%s%n", ds.name, ds.path);
                    if (cfg.skipMissingArff) continue;
                    throw new java.io.FileNotFoundException(String.valueOf(ds.path));
                }
                for (int seed : cfg.seeds) {
                    for (String modelName : cfg.models) {
                        try {
                            all.add(runOne(cfg, ds, modelName, seed, csv));
                        } catch (Exception e) {
                            System.err.printf("[proto][FAIL] %s|%s|seed=%d -> %s%n",
                                    ds.name, modelName, seed, e);
                        }
                    }
                }
            }
        }
        printSummary(all, cfg);
    }

    private RunResult runOne(Cfg cfg, DatasetSpec ds, String modelName,
                             int seed, PrintWriter csv) throws Exception {
        RunResult rr = new RunResult();
        rr.dataset = ds.name;
        rr.model = modelName;
        rr.seed = seed;

        InstanceStream stream = buildStream(ds, seed);
        if (stream instanceof OptionHandler) ((OptionHandler) stream).prepareForUse();
        InstancesHeader header = stream.getHeader();
        int numFeatures = header.numAttributes() - 1;
        int numClasses  = header.numClasses();
        rr.d = numFeatures;
        int K = StaticFeatureSelector.defaultK(numFeatures);
        rr.k = K;

        StaticFeatureSelector selector = new StaticFeatureSelector(numFeatures, numClasses);
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
        selector.initialize(window, labels);

        int[] sel = selector.getCurrentSelection();
        rr.selectedCount = sel.length;
        rr.kCorrect = (sel.length == K);

        double[] sample = window[0];
        double[] filtered = selector.filterInstance(sample);
        rr.filterReduces = (filtered.length == K) && (filtered.length < sample.length || numFeatures == K);

        ModelWrapper model = buildModel(modelName, selector, header, numClasses);
        ModelWrapper majority = new MajorityClassWrapper(selector, numClasses);
        ModelWrapper noChange = new NoChangeWrapper(selector, numClasses);

        rr.modelAttrs = model.getCurrentSelection().length;
        rr.modelSeesK = (rr.modelAttrs == K);

        for (int i = 0; i < collected; i++) {
            majority.train(null, labels[i]);
            noChange.train(null, labels[i]);
        }

        MetricsCollector m  = new MetricsCollector(numClasses, cfg.windowSize, cfg.logEvery, cfg.ramSampleEvery);
        MetricsCollector mM = new MetricsCollector(numClasses, cfg.windowSize, cfg.logEvery, cfg.ramSampleEvery);
        MetricsCollector mN = new MetricsCollector(numClasses, cfg.windowSize, cfg.logEvery, cfg.ramSampleEvery);

        long n = collected;
        while (stream.hasMoreInstances() && n < cfg.maxInstances) {
            Instance x = stream.nextInstance().getData();
            int yTrue = (int) x.classValue();

            long t0 = System.nanoTime();
            int yHat = model.predict(x);
            long elapsed = System.nanoTime() - t0;
            int yMaj = majority.predict(x);
            int yNC  = noChange.predict(x);

            m.update(yTrue, yHat, elapsed);
            mM.update(yTrue, yMaj, 0);
            mN.update(yTrue, yNC, 0);

            model.train(x, yTrue);
            majority.train(x, yTrue);
            noChange.train(x, yTrue);

            n++;
            if (cfg.logEvery > 0 && n % cfg.logEvery == 0) {
                MetricsCollector.Snapshot s = m.snapshot();
                csv.printf(Locale.ROOT,
                        "%d,%s,%s,S1,%d,\"%s\",%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.6f%n",
                        n, ds.name, modelName, seed,
                        Arrays.toString(sel),
                        sel.length,
                        model.getCurrentSelection().length,
                        s.accuracyWindow, s.kappa, s.kappaPer,
                        mM.snapshot().accuracyWindow,
                        mN.snapshot().accuracyWindow,
                        s.ramHoursGB);
                csv.flush();
            }
        }

        MetricsCollector.Snapshot s = m.snapshot();
        rr.instances    = n;
        rr.accModel     = s.accuracyWindow;
        rr.kappaModel   = s.kappa;
        rr.kappaPerModel= s.kappaPer;
        rr.accMajority  = mM.snapshot().accuracyWindow;
        rr.accNoChange  = mN.snapshot().accuracyWindow;
        rr.ramHours     = s.ramHoursGB;
        rr.baselinesRan = (mM.getInstances() > 0 && mN.getInstances() > 0);
        rr.beatsMajority= rr.accModel >= rr.accMajority;
        rr.beatsNoChange= rr.accModel >= rr.accNoChange;

        StringBuilder why = new StringBuilder();
        if (!rr.kCorrect)        why.append("K!=ceil(sqrt(d)); ");
        if (!rr.filterReduces)   why.append("filterInstance not reducing; ");
        if (!rr.modelSeesK)      why.append("model attrs != K; ");
        if (!rr.baselinesRan)    why.append("baselines not running; ");
        if (!rr.beatsMajority)   why.append("acc<majority; ");
        if (!rr.beatsNoChange)   why.append("acc<nochange; ");
        rr.reason = why.toString();
        rr.pass = rr.kCorrect && rr.filterReduces && rr.modelSeesK && rr.baselinesRan;

        System.out.printf(Locale.ROOT,
                "[proto] %-14s %-4s seed=%d  d=%d K=%d sel=%s | acc=%.3f k=%.3f kPer=%.3f | maj=%.3f nc=%.3f | %s%n",
                ds.name, modelName, seed, rr.d, rr.k, Arrays.toString(sel),
                rr.accModel, rr.kappaModel, rr.kappaPerModel,
                rr.accMajority, rr.accNoChange,
                rr.pass ? "PASS" : ("FAIL: " + rr.reason));
        return rr;
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


    private void printSummary(List<RunResult> all, Cfg cfg) throws Exception {
        try (PrintWriter w = new PrintWriter(new FileWriter(
                Paths.get(cfg.outputDir, "E1_prototype_summary.txt").toFile()))) {
            int pass = 0, fail = 0;
            w.println("=== E1 Prototype Summary ===");
            for (RunResult r : all) {
                w.printf(Locale.ROOT,
                        "%-14s %-4s seed=%d  d=%d K=%d selCount=%d modelAttrs=%d  "
                                + "acc=%.4f k=%.4f kPer=%.4f  maj=%.4f nc=%.4f  ram=%.6f  %s %s%n",
                        r.dataset, r.model, r.seed, r.d, r.k, r.selectedCount, r.modelAttrs,
                        r.accModel, r.kappaModel, r.kappaPerModel,
                        r.accMajority, r.accNoChange, r.ramHours,
                        r.pass ? "PASS" : "FAIL", r.reason);
                if (r.pass) pass++; else fail++;
            }
            w.printf("--- TOTAL: %d PASS / %d FAIL ---%n", pass, fail);
            System.out.printf("[proto] TOTAL: %d PASS / %d FAIL  -> %s/E1_prototype_summary.txt%n",
                    pass, fail, cfg.outputDir);
        }
    }
}
