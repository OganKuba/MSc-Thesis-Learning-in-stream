package thesis.evaluation;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

public class CDDiagramExporter {

    public static void writeRanks(Path file, List<String> methodNames, double[] averageRanks,
                                  double criticalDifference, int numDatasets, double alpha) throws IOException {
        if (file == null) throw new IllegalArgumentException("file == null");
        if (methodNames == null || averageRanks == null) throw new IllegalArgumentException("null inputs");
        if (methodNames.size() != averageRanks.length) {
            throw new IllegalArgumentException("methodNames and averageRanks must match length");
        }
        ensureParent(file);
        try (BufferedWriter bw = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
            bw.write(String.format(Locale.ROOT,
                    "# n_datasets=%d alpha=%.4f CD=%.6f", numDatasets, alpha, criticalDifference));
            bw.newLine();
            bw.write("method,avg_rank");
            bw.newLine();
            for (int i = 0; i < methodNames.size(); i++) {
                bw.write(escape(methodNames.get(i)) + "," + fmt(averageRanks[i]));
                bw.newLine();
            }
        }
    }

    public static void writeRankMatrix(Path file, List<String> datasetNames, List<String> methodNames,
                                       double[][] ranks) throws IOException {
        if (file == null) throw new IllegalArgumentException("file == null");
        if (ranks == null || ranks.length == 0) throw new IllegalArgumentException("ranks empty");
        if (ranks.length != datasetNames.size()) throw new IllegalArgumentException("rows mismatch");
        if (ranks[0].length != methodNames.size()) throw new IllegalArgumentException("cols mismatch");
        ensureParent(file);
        try (BufferedWriter bw = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
            bw.write("dataset");
            for (String m : methodNames) bw.write("," + escape(m));
            bw.newLine();
            for (int i = 0; i < ranks.length; i++) {
                if (ranks[i] == null || ranks[i].length != methodNames.size()) {
                    throw new IllegalArgumentException("ragged ranks at row " + i);
                }
                bw.write(escape(datasetNames.get(i)));
                for (int j = 0; j < ranks[i].length; j++) bw.write("," + fmt(ranks[i][j]));
                bw.newLine();
            }
        }
    }

    public static void writePairwiseSignificance(Path file, List<String> methodNames,
                                                 boolean[][] significant, double[][] rankDifferences,
                                                 double criticalDifference) throws IOException {
        if (file == null) throw new IllegalArgumentException("file == null");
        if (methodNames == null || significant == null || rankDifferences == null) {
            throw new IllegalArgumentException("null inputs");
        }
        int k = methodNames.size();
        if (significant.length != k || rankDifferences.length != k) {
            throw new IllegalArgumentException("matrix size != methodNames.size");
        }
        ensureParent(file);
        try (BufferedWriter bw = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
            bw.write(String.format(Locale.ROOT, "# CD=%.6f", criticalDifference));
            bw.newLine();
            bw.write("method_a,method_b,rank_diff,significant");
            bw.newLine();
            for (int i = 0; i < k; i++) {
                for (int j = i + 1; j < k; j++) {
                    bw.write(escape(methodNames.get(i)) + "," + escape(methodNames.get(j)) + ","
                            + fmt(rankDifferences[i][j]) + "," + significant[i][j]);
                    bw.newLine();
                }
            }
        }
    }

    public static void writeLongScores(Path file, List<String> methodNames, List<String> datasetNames,
                                       List<Integer> seeds, double[][][] scores,
                                       double[] avgRanks, double[][] perDatasetRanks,
                                       double[][] pairwisePValues) throws IOException {
        if (file == null) throw new IllegalArgumentException("file == null");
        if (scores == null) throw new IllegalArgumentException("scores == null");
        int M = methodNames.size();
        int D = datasetNames.size();
        int S = seeds == null ? 1 : seeds.size();
        if (scores.length != D) throw new IllegalArgumentException("scores rows != datasets");
        ensureParent(file);
        try (BufferedWriter bw = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
            bw.write("method,dataset,seed,score,rank,avg_rank,p_value_vs_first,significant_vs_first");
            bw.newLine();
            for (int d = 0; d < D; d++) {
                if (scores[d] == null || scores[d].length != M)
                    throw new IllegalArgumentException("scores[" + d + "] bad shape");
                for (int m = 0; m < M; m++) {
                    if (scores[d][m] == null || scores[d][m].length != S)
                        throw new IllegalArgumentException("scores[" + d + "][" + m + "] bad shape");
                    for (int s = 0; s < S; s++) {
                        double rank = perDatasetRanks == null ? Double.NaN : perDatasetRanks[d][m];
                        double avg = avgRanks == null ? Double.NaN : avgRanks[m];
                        double pv = pairwisePValues == null ? Double.NaN : pairwisePValues[m][0];
                        boolean sig = Double.isFinite(pv) && pv < 0.05;
                        bw.write(escape(methodNames.get(m)) + "," + escape(datasetNames.get(d))
                                + "," + (seeds == null ? 0 : seeds.get(s))
                                + "," + fmt(scores[d][m][s])
                                + "," + fmt(rank)
                                + "," + fmt(avg)
                                + "," + fmt(pv)
                                + "," + sig);
                        bw.newLine();
                    }
                }
            }
        }
    }

    private static void ensureParent(Path file) throws IOException {
        Path parent = file.toAbsolutePath().getParent();
        if (parent != null) Files.createDirectories(parent);
    }

    private static String fmt(double v) {
        if (!Double.isFinite(v)) return "";
        return String.format(Locale.ROOT, "%.6f", v);
    }

    private static String escape(String s) {
        if (s == null) return "";
        if (s.indexOf(',') < 0 && s.indexOf('"') < 0 && s.indexOf('\n') < 0) return s;
        return "\"" + s.replace("\"", "\"\"") + "\"";
    }
}