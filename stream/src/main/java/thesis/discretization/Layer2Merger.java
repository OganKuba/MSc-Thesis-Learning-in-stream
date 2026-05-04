package thesis.discretization;

import java.util.Arrays;

public final class Layer2Merger {

    private Layer2Merger() {}

    public static int[] merge(int[] binCounts, int[][] classCounts,
                              int b2, int numClasses) {
        int b1 = binCounts.length;
        if (b2 < 2 || b2 > b1) {
            throw new IllegalArgumentException("require 2 <= b2 <= b1");
        }

        int[] groupCount = new int[b1];
        double[][] groupClass = new double[b1][numClasses];
        for (int i = 0; i < b1; i++) {
            groupCount[i] = binCounts[i];
            for (int c = 0; c < numClasses; c++) {
                groupClass[i][c] = classCounts[i][c];
            }
        }

        int[] next = new int[b1];
        for (int i = 0; i < b1; i++) next[i] = i + 1;
        next[b1 - 1] = -1;

        boolean[] alive = new boolean[b1];
        Arrays.fill(alive, true);

        int groups = b1;
        while (groups > b2) {
            int bestA = -1;
            double bestD = Double.POSITIVE_INFINITY;
            int a = 0;
            while (a != -1) {
                int b = next[a];
                if (b == -1) break;
                double d = distance(groupCount[a], groupClass[a],
                        groupCount[b], groupClass[b], numClasses);
                if (d < bestD) { bestD = d; bestA = a; }
                a = b;
            }
            if (bestA == -1) break;
            int b = next[bestA];
            groupCount[bestA] += groupCount[b];
            for (int c = 0; c < numClasses; c++) {
                groupClass[bestA][c] += groupClass[b][c];
            }
            next[bestA] = next[b];
            alive[b] = false;
            groups--;
        }

        int[] head = new int[groups];
        int idx = 0;
        for (int i = 0; i < b1; i++) if (alive[i]) head[idx++] = i;

        int[] l1ToL2 = new int[b1];
        for (int g = 0; g < groups; g++) {
            int start = head[g];
            int end = (g == groups - 1) ? b1 : head[g + 1];
            for (int i = start; i < end; i++) l1ToL2[i] = g;
        }
        return l1ToL2;
    }

    private static double distance(int countA, double[] cA,
                                   int countB, double[] cB,
                                   int numClasses) {
        double na = countA + numClasses;
        double nb = countB + numClasses;
        double tv = 0.0;
        for (int c = 0; c < numClasses; c++) {
            double pa = (cA[c] + 1.0) / na;
            double pb = (cB[c] + 1.0) / nb;
            tv += Math.abs(pa - pb);
        }
        tv *= 0.5;
        double weight = (countA + countB) + numClasses;
        return tv * weight;
    }
}