package thesis.models;

import java.util.Collections;
import java.util.Random;
import java.util.Set;

public final class WeightedSubspaceSampler {

    private WeightedSubspaceSampler() {}

    public static int[] sample(double[] weights, int size, Random rng, Set<Integer> exclude) {
        if (weights == null) throw new IllegalArgumentException("weights must not be null");
        if (rng == null) throw new IllegalArgumentException("rng must not be null");
        if (size < 1) throw new IllegalArgumentException("size must be >= 1");
        int n = weights.length;
        if (size > n) throw new IllegalArgumentException("size > weights.length");

        Set<Integer> excludeSet = (exclude == null) ? Collections.emptySet() : exclude;

        double[] keys = new double[n];
        boolean[] selectable = new boolean[n];
        int selectableCount = 0;
        for (int i = 0; i < n; i++) {
            if (excludeSet.contains(i)) { keys[i] = Double.NEGATIVE_INFINITY; continue; }
            double w = weights[i];
            if (!(w > 0.0) || !Double.isFinite(w)) { keys[i] = Double.NEGATIVE_INFINITY; continue; }
            double u = rng.nextDouble();
            if (u <= 0.0) u = Double.MIN_NORMAL;
            keys[i] = Math.log(u) / w;
            selectable[i] = true;
            selectableCount++;
        }

        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        sortIndicesByKeyDesc(order, keys);

        int weightedTake = Math.min(size, selectableCount);
        boolean[] picked = new boolean[n];
        int pickedCount = 0;
        for (int i = 0; i < n && pickedCount < weightedTake; i++) {
            int idx = order[i];
            if (selectable[idx]) { picked[idx] = true; pickedCount++; }
        }

        if (pickedCount < size) {
            int needed = size - pickedCount;
            int[] remaining = new int[n];
            int rc = 0;
            for (int i = 0; i < n; i++) {
                if (excludeSet.contains(i)) continue;
                if (picked[i]) continue;
                remaining[rc++] = i;
            }
            for (int i = rc - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = remaining[i]; remaining[i] = remaining[j]; remaining[j] = tmp;
            }
            for (int i = 0; i < rc && needed > 0; i++) {
                picked[remaining[i]] = true; pickedCount++; needed--;
            }
            if (needed > 0) {
                int[] excluded = new int[n];
                int ec = 0;
                for (int i = 0; i < n; i++) {
                    if (picked[i]) continue;
                    excluded[ec++] = i;
                }
                for (int i = ec - 1; i > 0; i--) {
                    int j = rng.nextInt(i + 1);
                    int tmp = excluded[i]; excluded[i] = excluded[j]; excluded[j] = tmp;
                }
                for (int i = 0; i < ec && needed > 0; i++) {
                    picked[excluded[i]] = true; pickedCount++; needed--;
                }
            }
        }

        int[] out = new int[pickedCount];
        int j = 0;
        for (int i = 0; i < n; i++) if (picked[i]) out[j++] = i;
        return out;
    }

    private static void sortIndicesByKeyDesc(int[] idx, double[] keys) {
        for (int i = 1; i < idx.length; i++) {
            int cur = idx[i]; double curKey = keys[cur]; int j = i - 1;
            while (j >= 0 && keys[idx[j]] < curKey) { idx[j + 1] = idx[j]; j--; }
            idx[j + 1] = cur;
        }
    }
}