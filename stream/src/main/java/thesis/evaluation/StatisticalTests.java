package thesis.evaluation;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class StatisticalTests {

    private final double alpha;
    private final boolean higherIsBetter;

    public StatisticalTests() { this(0.05, true); }

    public StatisticalTests(double alpha, boolean higherIsBetter) {
        if (!(alpha > 0.0 && alpha < 1.0)) throw new IllegalArgumentException("alpha out of (0,1)");
        this.alpha = alpha;
        this.higherIsBetter = higherIsBetter;
    }

    public FriedmanTest.Result friedman(double[][] matrix) {
        return new FriedmanTest(higherIsBetter).test(matrix);
    }

    public NemenyiPostHoc.Result nemenyi(FriedmanTest.Result fr) {
        return new NemenyiPostHoc(alpha).test(fr.averageRanks, fr.numDatasets);
    }

    public WilcoxonSignedRank.Result wilcoxon(double[] a, double[] b) {
        return new WilcoxonSignedRank().test(a, b);
    }

    public Report runFull(double[][] matrix, List<String> datasetNames, List<String> methodNames) {
        if (matrix == null || matrix.length == 0) throw new IllegalArgumentException("empty matrix");
        if (datasetNames == null || methodNames == null) throw new IllegalArgumentException("null names");
        if (datasetNames.size() != matrix.length)
            throw new IllegalArgumentException("datasetNames.size != rows");
        if (methodNames.size() != matrix[0].length)
            throw new IllegalArgumentException("methodNames.size != cols");
        FriedmanTest.Result fr = friedman(matrix);
        NemenyiPostHoc.Result nr = nemenyi(fr);
        double[][] pairP = pairwiseWilcoxon(matrix);
        return new Report(matrix, datasetNames, methodNames, fr, nr, pairP, alpha);
    }

    public double[][] pairwiseWilcoxon(double[][] matrix) {
        int n = matrix.length;
        int k = matrix[0].length;
        double[][] p = new double[k][k];
        for (int i = 0; i < k; i++) java.util.Arrays.fill(p[i], Double.NaN);
        WilcoxonSignedRank w = new WilcoxonSignedRank();
        for (int a = 0; a < k; a++) {
            for (int b = a + 1; b < k; b++) {
                double[] va = new double[n];
                double[] vb = new double[n];
                for (int i = 0; i < n; i++) { va[i] = matrix[i][a]; vb[i] = matrix[i][b]; }
                try {
                    WilcoxonSignedRank.Result r = w.test(va, vb);
                    p[a][b] = r.degenerate ? 1.0 : r.pValue;
                    p[b][a] = p[a][b];
                } catch (Exception ex) {
                    p[a][b] = 1.0;
                    p[b][a] = 1.0;
                }
            }
        }
        return p;
    }

    public static final class Report {
        public final double[][] matrix;
        public final List<String> datasetNames;
        public final List<String> methodNames;
        public final FriedmanTest.Result friedman;
        public final NemenyiPostHoc.Result nemenyi;
        public final double[][] pairwiseWilcoxon;
        public final double alpha;

        public Report(double[][] matrix, List<String> datasetNames, List<String> methodNames,
                      FriedmanTest.Result friedman, NemenyiPostHoc.Result nemenyi,
                      double[][] pairwiseWilcoxon, double alpha) {
            this.matrix = matrix;
            this.datasetNames = datasetNames;
            this.methodNames = methodNames;
            this.friedman = friedman;
            this.nemenyi = nemenyi;
            this.pairwiseWilcoxon = pairwiseWilcoxon;
            this.alpha = alpha;
        }

        public void exportCD(Path outDir) throws IOException {
            CDDiagramExporter.writeRanks(outDir.resolve("avg_ranks.csv"),
                    methodNames, friedman.averageRanks, nemenyi.criticalDifference,
                    friedman.numDatasets, alpha);
            CDDiagramExporter.writeRankMatrix(outDir.resolve("rank_matrix.csv"),
                    datasetNames, methodNames, friedman.ranks);
            CDDiagramExporter.writePairwiseSignificance(outDir.resolve("pairwise_significance.csv"),
                    methodNames, nemenyi.significant, nemenyi.rankDifferences, nemenyi.criticalDifference);
        }

        public String summary() {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format(Locale.ROOT,
                    "Friedman: chi2=%.4f (df=%d) p=%.4g  |  Iman-Davenport F=%.4f (df=%d,%d) p=%.4g  |  alpha=%.3f%n",
                    friedman.chiSquared, friedman.dfChi, friedman.pValueChi,
                    friedman.imanDavenport, friedman.dfF1, friedman.dfF2, friedman.pValueF, alpha));
            sb.append(String.format(Locale.ROOT,
                    "Nemenyi: q=%.4f  CD=%.4f%n", nemenyi.qAlpha, nemenyi.criticalDifference));
            for (int i = 0; i < methodNames.size(); i++) {
                sb.append(String.format(Locale.ROOT,
                        "  rank[%s] = %.3f%n", methodNames.get(i), friedman.averageRanks[i]));
            }
            sb.append("Significant pairs (|delta rank| > CD):\n");
            for (int[] p : nemenyi.significantPairs) {
                sb.append(String.format(Locale.ROOT,
                        "  %s  vs  %s   delta=%.3f%n",
                        methodNames.get(p[0]), methodNames.get(p[1]),
                        nemenyi.rankDifferences[p[0]][p[1]]));
            }
            return sb.toString();
        }
    }
}