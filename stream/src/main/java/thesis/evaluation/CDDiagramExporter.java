package thesis.evaluation;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class CDDiagramExporter {

    public static void writeRanks(Path file, List<String> methodNames, double[] averageRanks,
                                  double criticalDifference, int numDatasets, double alpha) throws IOException {
        if (methodNames.size() != averageRanks.length) {
            throw new IllegalArgumentException("methodNames and averageRanks must match length");
        }
        Files.createDirectories(file.toAbsolutePath().getParent());
        try (BufferedWriter bw = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
            bw.write("# n_datasets=" + numDatasets + " alpha=" + alpha + " CD=" + criticalDifference);
            bw.newLine();
            bw.write("method,avg_rank");
            bw.newLine();
            for (int i = 0; i < methodNames.size(); i++) {
                bw.write(escape(methodNames.get(i)) + "," + averageRanks[i]);
                bw.newLine();
            }
        }
    }

    public static void writeRankMatrix(Path file, List<String> datasetNames, List<String> methodNames,
                                       double[][] ranks) throws IOException {
        if (ranks.length != datasetNames.size()) throw new IllegalArgumentException("rows mismatch");
        if (ranks[0].length != methodNames.size()) throw new IllegalArgumentException("cols mismatch");
        Files.createDirectories(file.toAbsolutePath().getParent());
        try (BufferedWriter bw = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
            bw.write("dataset");
            for (String m : methodNames) bw.write("," + escape(m));
            bw.newLine();
            for (int i = 0; i < ranks.length; i++) {
                bw.write(escape(datasetNames.get(i)));
                for (int j = 0; j < ranks[i].length; j++) bw.write("," + ranks[i][j]);
                bw.newLine();
            }
        }
    }

    public static void writePairwiseSignificance(Path file, List<String> methodNames,
                                                 boolean[][] significant, double[][] rankDifferences,
                                                 double criticalDifference) throws IOException {
        Files.createDirectories(file.toAbsolutePath().getParent());
        try (BufferedWriter bw = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
            bw.write("# CD=" + criticalDifference);
            bw.newLine();
            bw.write("method_a,method_b,rank_diff,significant");
            bw.newLine();
            int k = methodNames.size();
            for (int i = 0; i < k; i++) {
                for (int j = i + 1; j < k; j++) {
                    bw.write(escape(methodNames.get(i)) + "," + escape(methodNames.get(j)) + ","
                            + rankDifferences[i][j] + "," + significant[i][j]);
                    bw.newLine();
                }
            }
        }
    }

    private static String escape(String s) {
        if (s.indexOf(',') < 0 && s.indexOf('"') < 0 && s.indexOf('\n') < 0) return s;
        return "\"" + s.replace("\"", "\"\"") + "\"";
    }
}