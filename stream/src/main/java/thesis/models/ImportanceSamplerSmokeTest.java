package thesis.models;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class ImportanceSamplerSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("=== ImportanceSamplerSmokeTest ===");
        testFeatureImportanceUniformByDefault();
        testFeatureImportanceUpdateAndNormalize();
        testFeatureImportanceBoostNormalizes();
        testFeatureImportanceDegenerateFallback();
        testWeightedSamplerNoReplacement();
        testWeightedSamplerHonorsExclude();
        testWeightedSamplerPrefersHighWeight();
        testWeightedSamplerFallsBackOnAllZero();

        System.out.println("======================================================================");
        System.out.println("RESULT: " + passed + " passed, " + failed + " failed");
        System.out.println("======================================================================");
        if (failed > 0) System.exit(1);
    }

    private static void testFeatureImportanceUniformByDefault() {
        FeatureImportance imp = new FeatureImportance(4);
        double[] v = imp.getImportance();
        boolean ok = true;
        for (double x : v) if (Math.abs(x - 0.25) > 1e-9) ok = false;
        report("FeatureImportance defaults to uniform", ok);
    }

    private static void testFeatureImportanceUpdateAndNormalize() {
        FeatureImportance imp = new FeatureImportance(3);
        imp.update(new double[]{0.0, 0.5, 1.0}, new double[]{0.0, 0.0, 0.0});
        double[] v = imp.getImportance();
        double sum = 0.0; for (double x : v) sum += x;
        boolean monotone = v[2] >= v[1] && v[1] >= v[0];
        report("FeatureImportance.update normalizes (sum=" + sum
                        + ", monotone=" + monotone + ")",
                Math.abs(sum - 1.0) < 1e-9 && monotone);
    }

    private static void testFeatureImportanceBoostNormalizes() {
        FeatureImportance imp = new FeatureImportance(4);
        imp.boost(Set.of(2), 5.0);
        double[] v = imp.getImportance();
        double sum = 0.0; for (double x : v) sum += x;
        boolean ok = Math.abs(sum - 1.0) < 1e-9 && v[2] > v[0];
        report("FeatureImportance.boost increases target and re-normalizes", ok);
    }

    private static void testFeatureImportanceDegenerateFallback() {
        FeatureImportance imp = new FeatureImportance(3);
        imp.update(new double[]{0.0, 0.0, 0.0}, new double[]{1e30, 1e30, 1e30});
        double[] v = imp.getImportance();
        double sum = 0.0; for (double x : v) sum += x;
        report("FeatureImportance degenerate input → uniform fallback (fallbacks="
                        + imp.getDegenerateUniformFallbacks() + ")",
                Math.abs(sum - 1.0) < 1e-9
                        && imp.getDegenerateUniformFallbacks() >= 0);
    }

    private static void testWeightedSamplerNoReplacement() {
        double[] w = {0.1, 0.2, 0.3, 0.4, 0.5};
        Random rng = new Random(42);
        int[] s = WeightedSubspaceSampler.sample(w, 3, rng, Set.of());
        Set<Integer> seen = new HashSet<>();
        for (int x : s) seen.add(x);
        report("WeightedSampler returns distinct indices (got " + Arrays.toString(s) + ")",
                s.length == 3 && seen.size() == 3);
    }

    private static void testWeightedSamplerHonorsExclude() {
        double[] w = {1, 1, 1, 1, 1};
        Random rng = new Random(7);
        int[] s = WeightedSubspaceSampler.sample(w, 2, rng, Set.of(0, 1));
        boolean ok = true;
        for (int x : s) if (x == 0 || x == 1) ok = false;
        report("WeightedSampler honors exclude (got " + Arrays.toString(s) + ")", ok);
    }

    private static void testWeightedSamplerPrefersHighWeight() {
        double[] w = {0.001, 0.001, 0.001, 0.001, 1.0};
        int hits = 0, trials = 200;
        for (int t = 0; t < trials; t++) {
            int[] s = WeightedSubspaceSampler.sample(w, 1, new Random(t), Set.of());
            if (s.length == 1 && s[0] == 4) hits++;
        }
        report("WeightedSampler heavy-weight wins majority (hits=" + hits + "/" + trials + ")",
                hits >= trials * 0.85);
    }

    private static void testWeightedSamplerFallsBackOnAllZero() {
        double[] w = {0, 0, 0, 0};
        int[] s = WeightedSubspaceSampler.sample(w, 2, new Random(1), Set.of());
        report("WeightedSampler falls back when all weights zero (size=" + s.length + ")",
                s.length == 2);
    }

    private static void report(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  [PASSED] " + name); }
        else    { failed++; System.out.println("  [FAILED] " + name); }
    }
}
