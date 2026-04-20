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
            double mean = sum[i] / count;
            double var = Math.max(0, sumSq[i] / count - mean * mean);
            System.out.printf("     [%2d] %-15s  μ=% .4f  σ=%.4f%n",
                    i, h.attribute(i).name(), mean, Math.sqrt(var));
            shown++;
        }
        System.out.println();
    }

    public static void main(String[] args) {
        int seed = 42;
        int n = 1000;

        summarize("SEA (drifts @25k/50k/75k)",
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

        // Noise-wrapped variants
        InstanceStream base = SyntheticStreamFactory.createCustomFeatureDrift(seed, 5, 0.01, 100_000);
        summarize("CustomFeatureDrift k=5 + 3 noise feats",
                SyntheticStreamFactory.addNoiseFeatures(base, 3), n);

        InstanceStream base2 = SyntheticStreamFactory.createCustomFeatureDrift(seed, 5, 0.01, 100_000);
        summarize("CustomFeatureDrift k=5 + 5 noise feats",
                SyntheticStreamFactory.addNoiseFeatures(base2, 5), n);
    }
}