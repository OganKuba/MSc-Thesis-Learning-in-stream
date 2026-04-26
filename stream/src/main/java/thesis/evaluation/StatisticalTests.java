package thesis.evaluation;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

public class StatisticalTests {

    private final double alpha;
    private final boolean higherIsBetter;

    public StatisticalTests() { this(0.05, true); }

    public StatisticalTests(double alpha, boolean higherIsBetter) {
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
        FriedmanTest.Result fr = friedman(matrix);
        NemenyiPostHoc.Result nr = nemenyi(fr);
        return new Report(matrix, datasetNames, methodNames, fr, nr, alpha);
    }

    public static final class Report {
        public final double[][] matrix;
        public final List<String> datasetNames;
        public final List<String> methodNames;
        public final FriedmanTest.Result friedman;
        public final NemenyiPostHoc.Result nemenyi;
        public final double alpha;

        public Report(double[][] matrix, List<String> datasetNames, List<String> methodNames,
                      FriedmanTest.Result friedman, NemenyiPostHoc.Result nemenyi, double alpha) {
            this.matrix = matrix;
            this.datasetNames = datasetNames;
            this.methodNames = methodNames;
            this.friedman = friedman;
            this.nemenyi = nemenyi;
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
            sb.append(String.format("Friedman: χ²=%.4f (df=%d) p=%.4g  |  Iman-Davenport F=%.4f (df=%d,%d) p=%.4g  |  alpha=%.3f%n",
                    friedman.chiSquared, friedman.dfChi, friedman.pValueChi,
                    friedman.imanDavenport, friedman.dfF1, friedman.dfF2, friedman.pValueF, alpha));
            sb.append(String.format("Nemenyi: q=%.4f  CD=%.4f%n", nemenyi.qAlpha, nemenyi.criticalDifference));
            for (int i = 0; i < methodNames.size(); i++) {
                sb.append(String.format("  rank[%s] = %.3f%n", methodNames.get(i), friedman.averageRanks[i]));
            }
            sb.append("Significant pairs (|Δrank| > CD):\n");
            for (int[] p : nemenyi.significantPairs) {
                sb.append(String.format("  %s  vs  %s   Δ=%.3f%n",
                        methodNames.get(p[0]), methodNames.get(p[1]),
                        nemenyi.rankDifferences[p[0]][p[1]]));
            }
            return sb.toString();
        }
    }
}