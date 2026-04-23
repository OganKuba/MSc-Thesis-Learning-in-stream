package thesis.selection;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

public class SelectorStrategiesSmokeTest {

    private static double[] makeRow(int y, Random rng, int d, int[] informative, double sep) {
        double[] row = new double[d];
        for (int i = 0; i < d; i++) row[i] = rng.nextGaussian();
        for (int idx : informative) {
            row[idx] = rng.nextGaussian() + (y == 1 ? sep : -sep);
        }
        return row;
    }

    private static int overlap(int[] sel, int[] truth) {
        Set<Integer> t = new HashSet<>();
        for (int x : truth) t.add(x);
        int c = 0;
        for (int x : sel) if (t.contains(x)) c++;
        return c;
    }

    private static void runStrategy(String label, FeatureSelector sel,
                                    double[][] warmup, int[] warmupLabels,
                                    int total, int driftAt, int d,
                                    int[] truthBefore, int[] truthAfter,
                                    long seed) {
        sel.initialize(warmup, warmupLabels);
        System.out.println("─".repeat(82));
        System.out.println(label);
        System.out.println("  init selection : " + Arrays.toString(sel.getSelectedFeatures()) +
                "   (truth pre-drift = " + Arrays.toString(truthBefore) + ")");

        Random rng = new Random(seed);
        int[] checkpoints = { driftAt - 1, driftAt + 200, driftAt + 600,
                driftAt + 1500, driftAt + 4000, total - 1 };
        int chkIdx = 0;

        for (int t = 0; t < total; t++) {
            int y = rng.nextInt(2);
            int[] truth = (t < driftAt) ? truthBefore : truthAfter;
            double[] row = makeRow(y, rng, d, truth, 2.0);
            boolean alarm = (t == driftAt);
            sel.update(row, y, alarm, alarm ? toSet(truthAfter) : Set.of());

            if (chkIdx < checkpoints.length && t == checkpoints[chkIdx]) {
                int[] s = sel.getCurrentSelection();
                int[] currentTruth = (t < driftAt) ? truthBefore : truthAfter;
                System.out.printf("  t=%5d  selection=%s  overlap_with_truth=%d/%d%n",
                        t, Arrays.toString(s), overlap(s, currentTruth), currentTruth.length);
                chkIdx++;
            }
        }
        System.out.println("  final selection: " + Arrays.toString(sel.getCurrentSelection()) +
                "   (truth post-drift = " + Arrays.toString(truthAfter) + ")");
    }

    private static Set<Integer> toSet(int[] a) {
        Set<Integer> s = new TreeSet<>();
        for (int x : a) s.add(x);
        return s;
    }

    public static void main(String[] args) {
        int d = 12;
        int numClasses = 2;
        int warmupN = 1500;
        int total = 10_000;
        int driftAt = 4000;
        int[] truthBefore = {0, 1, 2};
        int[] truthAfter  = {7, 8, 9};

        int[] warmupLabels = new int[warmupN];
        double[][] warmup = new double[warmupN][];
        Random rng = new Random(42);
        for (int i = 0; i < warmupN; i++) {
            int y = rng.nextInt(2);
            warmupLabels[i] = y;
            warmup[i] = makeRow(y, rng, d, truthBefore, 2.0);
        }

        System.out.println("Selector strategies S1 / S2 / S3 — relevance-swap drift at t=" + driftAt);
        System.out.println("d=" + d + "  K=" + StaticFeatureSelector.defaultK(d) +
                "  truth pre=" + Arrays.toString(truthBefore) +
                "  truth post=" + Arrays.toString(truthAfter));
        System.out.println();

        runStrategy("S1 — StaticFeatureSelector",
                new StaticFeatureSelector(d, numClasses),
                warmup, warmupLabels, total, driftAt, d, truthBefore, truthAfter, 100L);

        runStrategy("S2 — AlarmTriggeredSelector (W_postdrift=500)",
                new AlarmTriggeredSelector(d, numClasses),
                warmup, warmupLabels, total, driftAt, d, truthBefore, truthAfter, 100L);

        AlarmTriggeredSelector s2view = new AlarmTriggeredSelector(d, numClasses);
        s2view.initialize(warmup, warmupLabels);
        System.out.println("  (S2 reSelections counter exposed via getReSelections())");

        runStrategy("S3 — PeriodicSelector (N=1000, N_min=500, maxSwap=ceil(K*0.3))",
                new PeriodicSelector(d, numClasses),
                warmup, warmupLabels, total, driftAt, d, truthBefore, truthAfter, 100L);
    }
}