package thesis.pipeline;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.streams.InstanceStream;

import java.util.HashMap;
import java.util.Map;

public class SyntheticStreamSmokeTest {

    private static void summarize(String name, InstanceStream stream, int n) {
        System.out.println("─".repeat(70));
        System.out.println("STREAM: " + name);
        System.out.println("─".repeat(70));

        InstancesHeader h = stream.getHeader();
        int numAtts = h.numAttributes();
        int classIdx = h.classIndex();
        System.out.printf("  attributes=%d  classIdx=%d  classes=%d%n",
                numAtts, classIdx, h.numClasses());

        double[] sum = new double[numAtts];
        double[] sumSq = new double[numAtts];
        Map<String, Integer> classCounts = new HashMap<>();
        int count = 0;

        while (count < n && stream.hasMoreInstances()) {
            Instance inst = stream.nextInstance().getData();
            for (int i = 0; i < numAtts; i++) {
                if (i == classIdx) continue;
                double v = inst.value(i);
                sum[i] += v;
                sumSq[i] += v * v;
            }
            String c = h.attribute(classIdx).value((int) inst.classValue());
            classCounts.merge(c, 1, Integer::sum);
            count++;
        }

        System.out.println("  drew " + count + " instances");
        System.out.println("  class distribution: " + classCounts);
        System.out.println("  per-feature mean ± std (first 5 non-class):");
        int shown = 0;
        for (int i = 0; i < numAtts && shown < 5; i++) {
            if (i == classIdx) continue;
            double mean = count == 0 ? 0.0 : sum[i] / count;
            double var = count == 0 ? 0.0 : Math.max(0, sumSq[i] / count - mean * mean);
            System.out.printf("     [%2d] %-15s  μ=% .4f  σ=%.4f%n",
                    i, h.attribute(i).name(), mean, Math.sqrt(var));
            shown++;
        }
        System.out.println();
    }

    private static void checkLimit(String name, InstanceStream stream, int expected) {
        int count = 0;
        while (stream.hasMoreInstances()) {
            stream.nextInstance();
            count++;
            if (count > expected + 10) break;
        }
        String status = (count == expected) ? "OK" : "MISMATCH";
        System.out.printf("  LIMIT[%s] expected=%d  got=%d  → %s%n",
                name, expected, count, status);
    }

    private static void checkNoiseDeterminism(int seed) {
        InstanceStream a1 = SyntheticStreamFactory.addNoiseFeatures(
                SyntheticStreamFactory.createHyperplane(seed, 0.01, 1000), 3, seed);
        InstanceStream a2 = SyntheticStreamFactory.addNoiseFeatures(
                SyntheticStreamFactory.createHyperplane(seed, 0.01, 1000), 3, seed);

        int classIdx = a1.getHeader().classIndex();
        int numAtts = a1.getHeader().numAttributes();
        boolean equal = true;
        int compared = 0;
        while (a1.hasMoreInstances() && a2.hasMoreInstances() && compared < 200) {
            Instance i1 = a1.nextInstance().getData();
            Instance i2 = a2.nextInstance().getData();
            for (int j = 0; j < numAtts; j++) {
                if (j == classIdx) continue;
                if (Double.compare(i1.value(j), i2.value(j)) != 0) {
                    equal = false;
                    break;
                }
            }
            compared++;
        }
        System.out.printf("  NOISE DETERMINISM seed=%d  equal=%s  compared=%d%n",
                seed, equal, compared);

        InstanceStream b1 = SyntheticStreamFactory.addNoiseFeatures(
                SyntheticStreamFactory.createHyperplane(seed, 0.01, 1000), 3, seed);
        InstanceStream b2 = SyntheticStreamFactory.addNoiseFeatures(
                SyntheticStreamFactory.createHyperplane(seed, 0.01, 1000), 3, seed + 1);

        boolean differ = false;
        compared = 0;
        while (b1.hasMoreInstances() && b2.hasMoreInstances() && compared < 200) {
            Instance i1 = b1.nextInstance().getData();
            Instance i2 = b2.nextInstance().getData();
            for (int j = 0; j < numAtts; j++) {
                if (j == classIdx) continue;
                if (Double.compare(i1.value(j), i2.value(j)) != 0) {
                    differ = true;
                    break;
                }
            }
            if (differ) break;
            compared++;
        }
        System.out.printf("  NOISE SEED-SENSITIVITY seed=%d vs %d  differ=%s%n",
                seed, seed + 1, differ);
    }

    public static void main(String[] args) {
        int seed = 42;
        int n = 1000;

        summarize("SEA (drifts @25%/50%/75%)",
                SyntheticStreamFactory.createSEA(seed, 100_000), n);

        summarize("Hyperplane σ=0.001 (low)",
                SyntheticStreamFactory.createHyperplane(seed, 0.001, 100_000), n);
        summarize("Hyperplane σ=0.01 (med)",
                SyntheticStreamFactory.createHyperplane(seed, 0.01, 100_000), n);
        summarize("Hyperplane σ=0.1 (high)",
                SyntheticStreamFactory.createHyperplane(seed, 0.1, 100_000), n);

        summarize("RandomRBF speed=1e-4",
                SyntheticStreamFactory.createRandomRBF(seed, 1e-4, 100_000), n);
        summarize("RandomRBF speed=1e-3",
                SyntheticStreamFactory.createRandomRBF(seed, 1e-3, 100_000), n);
        summarize("RandomRBF speed=1e-2",
                SyntheticStreamFactory.createRandomRBF(seed, 1e-2, 100_000), n);

        summarize("STAGGER (recurring)",
                SyntheticStreamFactory.createSTAGGER(seed, 100_000), n);

        summarize("CustomFeatureDrift  k=2  σ=0.01",
                SyntheticStreamFactory.createCustomFeatureDrift(seed, 2, 0.01, 100_000), n);
        summarize("CustomFeatureDrift  k=5  σ=0.01",
                SyntheticStreamFactory.createCustomFeatureDrift(seed, 5, 0.01, 100_000), n);
        summarize("CustomFeatureDrift  k=10 σ=0.01",
                SyntheticStreamFactory.createCustomFeatureDrift(seed, 10, 0.01, 100_000), n);

        InstanceStream base = SyntheticStreamFactory.createCustomFeatureDrift(seed, 5, 0.01, 100_000);
        summarize("CustomFeatureDrift k=5 + 3 noise feats",
                SyntheticStreamFactory.addNoiseFeatures(base, 3, seed), n);

        InstanceStream base2 = SyntheticStreamFactory.createCustomFeatureDrift(seed, 5, 0.01, 100_000);
        summarize("CustomFeatureDrift k=5 + 5 noise feats",
                SyntheticStreamFactory.addNoiseFeatures(base2, 5, seed), n);

        System.out.println("─".repeat(70));
        System.out.println("LIMIT CHECKS");
        System.out.println("─".repeat(70));
        checkLimit("SEA n=5000",
                SyntheticStreamFactory.createSEA(seed, 5_000), 5_000);
        checkLimit("STAGGER n=5000",
                SyntheticStreamFactory.createSTAGGER(seed, 5_000), 5_000);
        checkLimit("Hyperplane n=2500",
                SyntheticStreamFactory.createHyperplane(seed, 0.01, 2_500), 2_500);
        checkLimit("RandomRBF n=2500",
                SyntheticStreamFactory.createRandomRBF(seed, 1e-3, 2_500), 2_500);
        checkLimit("CustomFeatureDrift n=2500",
                SyntheticStreamFactory.createCustomFeatureDrift(seed, 5, 0.01, 2_500), 2_500);
        checkLimit("Hyperplane+noise n=1500",
                SyntheticStreamFactory.addNoiseFeatures(
                        SyntheticStreamFactory.createHyperplane(seed, 0.01, 1_500), 4, seed),
                1_500);
        System.out.println();

        System.out.println("─".repeat(70));
        System.out.println("NOISE DETERMINISM CHECKS");
        System.out.println("─".repeat(70));
        checkNoiseDeterminism(1);
        checkNoiseDeterminism(7);
    }
}