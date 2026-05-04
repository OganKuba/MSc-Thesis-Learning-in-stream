package thesis.pipeline;

import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.streams.InstanceStream;
import moa.streams.generators.SEAGenerator;
import thesis.detection.TwoLevelDriftDetector;
import thesis.models.ModelWrapper;
import thesis.models.SRPWrapper;
import thesis.selection.FeatureSelector;
import thesis.selection.FilterRanker;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class ExperimentRunnerSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) throws Exception {
        System.out.println("=".repeat(70));
        System.out.println("EXPERIMENT RUNNER SMOKE TESTS");
        System.out.println("=".repeat(70));

        testValidateConfigRejectsEmpty();
        testValidateConfigRejectsDuplicateVariant();
        testRunsAllCombinationsCreatesFiles();
        testCsvHasSingleHeaderAndMetadataColumns();
        testFreshObjectsPerSeed();
        testFactoryNullIsCaughtPerExperiment();
        testWarnWhenAllSelectorsCollapseToSameClass();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    private static SEAGenerator makeStream(int seed) {
        SEAGenerator g = new SEAGenerator();
        g.instanceRandomSeedOption.setValue(seed);
        g.functionOption.setValue(1);
        g.balanceClassesOption.setValue(false);
        g.noisePercentageOption.setValue(10);
        g.prepareForUse();
        return g;
    }

    private static class StubSelectorS1 implements FeatureSelector {
        private final int F, K;
        private final int[] sel;
        StubSelectorS1(int F, int K) {
            this.F = F; this.K = K;
            this.sel = new int[K];
            for (int i = 0; i < K; i++) sel[i] = i;
        }
        @Override public boolean isInitialized() { return true; }
        @Override public void initialize(double[][] w, int[] y) { }
        @Override public void update(double[] x, int y, boolean a, Set<Integer> df) { }
        @Override public int[] getSelectedFeatures() { return sel.clone(); }
        @Override public int[] getCurrentSelection() { return sel.clone(); }
        @Override public double[] filterInstance(double[] x) {
            double[] o = new double[K];
            for (int i = 0; i < K; i++) o[i] = x[sel[i]];
            return o;
        }
        @Override public int getNumFeatures() { return F; }
        @Override public int getK() { return K; }
        @Override public String name() { return "S1Stub"; }
    }

    private static final class StubSelectorS2 extends StubSelectorS1 {
        StubSelectorS2(int F, int K) { super(F, K); }
        @Override public String name() { return "S2Stub"; }
    }

    private static final class StubDetector extends TwoLevelDriftDetector {
        StubDetector(int F) { super(new TwoLevelDriftDetector.Config(F)); }
    }

    private static ExperimentRunner.Config baseConfig(Path tmpRoot) {
        ExperimentRunner.Config cfg = new ExperimentRunner.Config();
        cfg.experimentGroup = "E_TEST";
        cfg.outputDir = tmpRoot.toString();
        cfg.datasets = List.of("DS1");
        cfg.seeds = List.of(11, 22);
        cfg.warmup = 1;
        cfg.logEvery = 50;
        cfg.maxInstances = 200;
        cfg.verbose = false;
        return cfg;
    }

    private static ExperimentRunner.Variant variant(String name, String model, String sel) {
        ExperimentRunner.Variant v = new ExperimentRunner.Variant();
        v.name = name; v.model = model; v.selector = sel; v.detector = "TWO_LEVEL";
        return v;
    }

    private static ExperimentRunner buildRunner(boolean s1Eq) {
        ExperimentRunner.StreamFactory sf = (ds, s) -> makeStream(s);
        ExperimentRunner.SelectorFactory selF = (n, F, C, s) -> {
            if (s1Eq) return new StubSelectorS1(F, 2);
            switch (n) {
                case "S2": return new StubSelectorS2(F, 2);
                default:   return new StubSelectorS1(F, 2);
            }
        };
        ExperimentRunner.ModelFactory mf = (n, sel, h, c, s) ->
                new SRPWrapper(sel, h, 3, 6.0, false, true);
        ExperimentRunner.DetectorFactory df = (n, F) -> new StubDetector(F);
        return new ExperimentRunner(sf, selF, mf, df, null);
    }

    private static void testValidateConfigRejectsEmpty() {
        ExperimentRunner r = buildRunner(false);
        ExperimentRunner.Config cfg = new ExperimentRunner.Config();
        boolean threw = false;
        try { r.runAll(cfg); } catch (Exception e) { threw = true; }
        report("validateConfig rejects empty datasets/variants/seeds", threw);
    }

    private static void testValidateConfigRejectsDuplicateVariant() throws IOException {
        Path tmp = Files.createTempDirectory("er_smoke_dup");
        ExperimentRunner.Config cfg = baseConfig(tmp);
        cfg.variants = List.of(variant("S1", "SRP", "S1"), variant("S1", "SRP", "S1"));
        ExperimentRunner r = buildRunner(false);
        boolean threw = false;
        try { r.runAll(cfg); } catch (Exception e) { threw = true; }
        report("validateConfig rejects duplicate variant name", threw);
    }

    private static void testRunsAllCombinationsCreatesFiles() throws IOException {
        Path tmp = Files.createTempDirectory("er_smoke_combo");
        ExperimentRunner.Config cfg = baseConfig(tmp);
        cfg.variants = List.of(variant("S1", "SRP", "S1"), variant("S2", "SRP", "S2"));
        ExperimentRunner r = buildRunner(false);
        r.runAll(cfg);
        long n = countCsv(tmp);
        report("runs all combos produces N csv files (got " + n + " expected 4)", n == 4);
    }

    private static void testCsvHasSingleHeaderAndMetadataColumns() throws IOException {
        Path tmp = Files.createTempDirectory("er_smoke_csv");
        ExperimentRunner.Config cfg = baseConfig(tmp);
        cfg.variants = List.of(variant("S1", "SRP", "S1"));
        cfg.seeds = List.of(7);
        ExperimentRunner r = buildRunner(false);
        r.runAll(cfg);
        Path file = anyCsv(tmp);
        List<String> lines = Files.readAllLines(file);
        long headerOccurrences = lines.stream()
                .filter(l -> l.startsWith("instance_num,"))
                .count();
        boolean hasMetaCols = lines.get(0).contains("dataset")
                && lines.get(0).contains("variant")
                && lines.get(0).contains("model")
                && lines.get(0).contains("trigger_type");
        report("CSV has exactly one header (got " + headerOccurrences + ")", headerOccurrences == 1);
        report("CSV header has metadata columns", hasMetaCols);
        boolean hasDsCell = lines.stream().skip(1).anyMatch(l -> l.contains(",DS1,"));
        report("CSV rows include dataset value 'DS1'", hasDsCell);
    }

    private static void testFreshObjectsPerSeed() throws IOException {
        Path tmp = Files.createTempDirectory("er_smoke_fresh");
        Set<Integer> selectorIds = new HashSet<>();
        ExperimentRunner.SelectorFactory selF = (n, F, C, s) -> {
            FeatureSelector sel = new StubSelectorS1(F, 2);
            selectorIds.add(System.identityHashCode(sel));
            return sel;
        };
        ExperimentRunner.StreamFactory sf = (ds, s) -> makeStream(s);
        ExperimentRunner.ModelFactory mf = (n, sel, h, c, s) ->
                new SRPWrapper(sel, h, 3, 6.0, false, true);
        ExperimentRunner.DetectorFactory df = (n, F) -> new StubDetector(F);
        ExperimentRunner r = new ExperimentRunner(sf, selF, mf, df);
        ExperimentRunner.Config cfg = baseConfig(tmp);
        cfg.variants = List.of(variant("S1", "SRP", "S1"));
        cfg.seeds = List.of(1, 2, 3);
        r.runAll(cfg);
        report("each seed gets fresh selector (uniq=" + selectorIds.size() + " expected 3)",
                selectorIds.size() == 3);
    }

    private static void testFactoryNullIsCaughtPerExperiment() throws IOException {
        Path tmp = Files.createTempDirectory("er_smoke_null");
        ExperimentRunner.SelectorFactory selF = (n, F, C, s) -> null;
        ExperimentRunner.StreamFactory sf = (ds, s) -> makeStream(s);
        ExperimentRunner.ModelFactory mf = (n, sel, h, c, s) ->
                new SRPWrapper(sel == null ? new StubSelectorS1(3, 2) : sel, h, 3, 6.0, false, true);
        ExperimentRunner.DetectorFactory df = (n, F) -> new StubDetector(F);
        ExperimentRunner r = new ExperimentRunner(sf, selF, mf, df);
        ExperimentRunner.Config cfg = baseConfig(tmp);
        cfg.variants = List.of(variant("S1", "SRP", "S1"));
        cfg.seeds = List.of(1);
        cfg.continueOnError = true;
        r.runAll(cfg);
        report("factory returning null is caught per experiment (failed=" + r.getFailed() + ")",
                r.getFailed() == 1 && r.getCompleted() == 0);
    }

    private static void testWarnWhenAllSelectorsCollapseToSameClass() throws IOException {
        Path tmp = Files.createTempDirectory("er_smoke_collapse");
        ExperimentRunner.Config cfg = baseConfig(tmp);
        cfg.variants = List.of(
                variant("S1", "SRP", "S1"),
                variant("S2", "SRP", "S2"),
                variant("S3", "SRP", "S3"),
                variant("S4", "SRP", "S4"));
        cfg.seeds = List.of(1);
        ExperimentRunner r = buildRunner(true);
        r.runAll(cfg);
        long distinct = r.getSeenSelectorClasses().values().stream().distinct().count();
        report("collapse detected: 4 variants → 1 selector class (distinct=" + distinct + ")",
                distinct == 1);
    }

    private static long countCsv(Path root) throws IOException {
        try (Stream<Path> s = Files.walk(root)) {
            return s.filter(p -> p.toString().endsWith(".csv")).count();
        }
    }

    private static Path anyCsv(Path root) throws IOException {
        try (Stream<Path> s = Files.walk(root)) {
            return s.filter(p -> p.toString().endsWith(".csv")).findFirst().orElseThrow();
        }
    }

    private static void report(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  [PASSED] " + name); }
        else    { failed++; System.out.println("  [FAILED] " + name); }
    }
}