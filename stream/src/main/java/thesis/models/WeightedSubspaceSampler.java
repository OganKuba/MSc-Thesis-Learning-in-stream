package thesis.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public final class WeightedSubspaceSampler {

    private WeightedSubspaceSampler() {}

    public static int[] sample(double[] weights, int size, Random rng, Set<Integer> exclude) {
        if (size < 1) throw new IllegalArgumentException("size must be >= 1");
        int n = weights.length;
        if (size > n) throw new IllegalArgumentException("size > weights.length");

        Set<Integer> excludeSet = (exclude == null) ? Collections.emptySet() : exclude;

        double[] keys = new double[n];
        boolean[] selectable = new boolean[n];
        int selectableCount = 0;
        for (int i = 0; i < n; i++) {
            if (excludeSet.contains(i)) {
                keys[i] = Double.NEGATIVE_INFINITY;
                continue;
            }
            double w = weights[i];
            if (!(w > 0.0) || Double.isNaN(w) || Double.isInfinite(w)) {
                keys[i] = Double.NEGATIVE_INFINITY;
                continue;
            }
            double u = rng.nextDouble();
            if (u <= 0.0) u = Double.MIN_NORMAL;
            keys[i] = Math.log(u) / w;
            selectable[i] = true;
            selectableCount++;
        }

        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(keys[b], keys[a]));

        int weightedTake = Math.min(size, selectableCount);
        Set<Integer> picked = new HashSet<>(weightedTake);
        for (int i = 0; i < n && picked.size() < weightedTake; i++) {
            int idx = order[i];
            if (selectable[idx]) picked.add(idx);
        }

        if (picked.size() < size) {
            List<Integer> remaining = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (excludeSet.contains(i)) continue;
                if (picked.contains(i)) continue;
                remaining.add(i);
            }
            Collections.shuffle(remaining, rng);
            for (int i = 0; picked.size() < size && i < remaining.size(); i++) {
                picked.add(remaining.get(i));
            }
        }

        int[] out = new int[picked.size()];
        int j = 0;
        for (int idx : picked) out[j++] = idx;
        Arrays.sort(out);
        return out;
    }
}