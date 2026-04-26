package thesis.evaluation;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class StatsSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) throws Exception {
        System.out.println("=".repeat(70));
        System.out.println("STATS SMOKE TESTS");
        System.out.println("=".repeat(70));

        testFriedmanRejectsClearWinner();
        testFriedmanFailsToRejectIdenticalColumns();
        testFriedmanLowerIsBetter();
        testFriedmanHandlesTies();
        testFriedmanRejectsBadInputs();

        testNemenyiCDPositiveAndDetectsBigGap();
        testNemenyiNoSignificantPairsWhenRanksClose();
        testNemenyiAlpha010LooserThan005();
        testNemenyiRejectsBadAlphaAndK();

        testWilcoxonRejectsConsistentDifference();
        testWilcoxonNoRejectOnNoise();
        testWilcoxonCountsWinsLossesTies();
        testWilcoxonRejectsBadInputs();
        testWilcoxonExactVsAsymptoticSwitch();

        testCDDiagramExporterRanks();
        testCDDiagramExporterRankMatrix();
        testCDDiagramExporterPairwiseSignificance();
        testCDDiagramExporterEscapesCommas();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    /* ---------- Friedman ---------- */

    private static void testFriedmanRejectsClearWinner() {
        double[][] m = new double[20][3];
        java.util.Random rng = new java.util.Random(1);
        for (int i = 0; i < m.length; i++) {
            m[i][0] = 0.90 + 0.01 * rng.nextGaussian();
            m[i][1] = 0.70 + 0.01 * rng.nextGaussian();
            m[i][2] = 0.50 + 0.01 * rng.nextGaussian();
        }
        FriedmanTest.Result r = new FriedmanTest(true).test(m);
        boolean ok = r.rejectsNull(0.05)
                && r.averageRanks[0] < r.averageRanks[1]
                && r.averageRanks[1] < r.averageRanks[2];
        report("Friedman rejects when one method clearly wins (chi2=" + r.chiSquared
                + ", pF=" + r.pValueF + ", ranks=" + Arrays.toString(r.averageRanks) + ")", ok);
    }

    private static void testFriedmanFailsToRejectIdenticalColumns() {
        double[][] m = new double[15][3];
        java.util.Random rng = new java.util.Random(2);
        for (int i = 0; i < m.length; i++) {
            double base = rng.nextDouble();
            m[i][0] = base + 1e-9 * rng.nextGaussian();
            m[i][1] = base + 1e-9 * rng.nextGaussian();
            m[i][2] = base + 1e-9 * rng.nextGaussian();
        }
        FriedmanTest.Result r = new FriedmanTest(true).test(m);
        boolean ok = !r.rejectsNull(0.05);
        report("Friedman does not reject when columns are equivalent (pChi="
                + r.pValueChi + ")", ok);
    }

    private static void testFriedmanLowerIsBetter() {
        double[][] m = {
                {0.1, 0.5, 0.9},
                {0.2, 0.6, 0.8},
                {0.05, 0.55, 0.95},
                {0.15, 0.45, 0.85}
        };
        FriedmanTest.Result r = new FriedmanTest(false).test(m);
        boolean ok = r.averageRanks[0] < r.averageRanks[1]
                && r.averageRanks[1] < r.averageRanks[2];
        report("Friedman lower-is-better orientation (ranks="
                + Arrays.toString(r.averageRanks) + ")", ok);
    }

    private static void testFriedmanHandlesTies() {
        double[][] m = {
                {1.0, 1.0, 0.5},
                {1.0, 1.0, 0.5},
                {1.0, 1.0, 0.5}
        };
        FriedmanTest.Result r = new FriedmanTest(true).test(m);
        boolean ok = Math.abs(r.averageRanks[0] - 1.5) < 1e-9
                && Math.abs(r.averageRanks[1] - 1.5) < 1e-9
                && Math.abs(r.averageRanks[2] - 3.0) < 1e-9;
        report("Friedman averages tied ranks (ranks="
                + Arrays.toString(r.averageRanks) + ")", ok);
    }

    private static void testFriedmanRejectsBadInputs() {
        boolean t1 = false, t2 = false, t3 = false;
        try { new FriedmanTest().test(null); } catch (IllegalArgumentException e) { t1 = true; }
        try { new FriedmanTest().test(new double[][]{{0.1}}); }
        catch (IllegalArgumentException e) { t2 = true; }
        try { new FriedmanTest().test(new double[][]{{0.1, 0.2}, {0.3}}); }
        catch (IllegalArgumentException e) { t3 = true; }
        report("Friedman rejects bad inputs", t1 && t2 && t3);
    }

    /* ---------- Nemenyi ---------- */

    private static void testNemenyiCDPositiveAndDetectsBigGap() {
        double[] avgRanks = {1.2, 2.0, 2.8};
        NemenyiPostHoc.Result r = new NemenyiPostHoc(0.05).test(avgRanks, 30);
        boolean ok = r.criticalDifference > 0.0
                && r.significant[0][2]
                && r.rankDifferences[0][2] > r.criticalDifference;
        report("Nemenyi detects big gap when N is large (CD=" + r.criticalDifference
                + ", sigPairs=" + r.significantPairs.size() + ")", ok);
    }

    private static void testNemenyiNoSignificantPairsWhenRanksClose() {
        double[] avgRanks = {2.0, 2.05, 2.1};
        NemenyiPostHoc.Result r = new NemenyiPostHoc(0.05).test(avgRanks, 5);
        boolean ok = r.significantPairs.isEmpty();
        report("Nemenyi no sig pairs when ranks nearly equal (CD="
                + r.criticalDifference + ")", ok);
    }

    private static void testNemenyiAlpha010LooserThan005() {
        double[] avgRanks = {1.5, 2.5, 3.0, 3.5};
        NemenyiPostHoc.Result r05 = new NemenyiPostHoc(0.05).test(avgRanks, 20);
        NemenyiPostHoc.Result r10 = new NemenyiPostHoc(0.10).test(avgRanks, 20);
        boolean ok = r10.criticalDifference < r05.criticalDifference;
        report("Nemenyi CD(alpha=0.10) < CD(alpha=0.05) (cd05="
                + r05.criticalDifference + ", cd10=" + r10.criticalDifference + ")", ok);
    }

    private static void testNemenyiRejectsBadAlphaAndK() {
        boolean t1 = false, t2 = false, t3 = false;
        try { new NemenyiPostHoc(0.07); } catch (IllegalArgumentException e) { t1 = true; }
        try { new NemenyiPostHoc(0.05).test(null, 10); } catch (IllegalArgumentException e) { t2 = true; }
        try { new NemenyiPostHoc(0.05).test(new double[]{1.0}, 10); }
        catch (IllegalArgumentException e) { t3 = true; }
        report("Nemenyi rejects bad alpha/k", t1 && t2 && t3);
    }

    /* ---------- Wilcoxon ---------- */

    private static void testWilcoxonRejectsConsistentDifference() {
        double[] a = new double[20];
        double[] b = new double[20];
        java.util.Random rng = new java.util.Random(11);
        for (int i = 0; i < a.length; i++) {
            a[i] = 0.85 + 0.01 * rng.nextGaussian();
            b[i] = 0.75 + 0.01 * rng.nextGaussian();
        }
        WilcoxonSignedRank.Result r = new WilcoxonSignedRank(false).test(a, b);
        boolean ok = r.rejectsNull(0.05) && r.wins > r.losses;
        report("Wilcoxon rejects null on consistent A>B (p=" + r.pValue
                + ", wins=" + r.wins + ", losses=" + r.losses + ")", ok);
    }

    private static void testWilcoxonNoRejectOnNoise() {
        double[] a = new double[25];
        double[] b = new double[25];
        java.util.Random rng = new java.util.Random(12);
        for (int i = 0; i < a.length; i++) {
            a[i] = rng.nextGaussian();
            b[i] = rng.nextGaussian();
        }
        WilcoxonSignedRank.Result r = new WilcoxonSignedRank(false).test(a, b);
        boolean ok = !r.rejectsNull(0.05);
        report("Wilcoxon does not reject null on independent noise (p=" + r.pValue + ")", ok);
    }

    private static void testWilcoxonCountsWinsLossesTies() {
        double[] a = {1, 2, 3, 4, 5};
        double[] b = {0, 2, 5, 4, 1};
        WilcoxonSignedRank.Result r = new WilcoxonSignedRank(true).test(a, b);
        boolean ok = r.wins == 2 && r.losses == 1 && r.ties == 2 && r.n == 5;
        report("Wilcoxon counts wins/losses/ties (w=" + r.wins
                + ", l=" + r.losses + ", t=" + r.ties + ")", ok);
    }

    private static void testWilcoxonRejectsBadInputs() {
        boolean t1 = false, t2 = false, t3 = false;
        try { new WilcoxonSignedRank().test(null, new double[]{1}); }
        catch (IllegalArgumentException e) { t1 = true; }
        try { new WilcoxonSignedRank().test(new double[]{1}, new double[]{1, 2}); }
        catch (IllegalArgumentException e) { t2 = true; }
        try { new WilcoxonSignedRank().test(new double[]{1}, new double[]{2}); }
        catch (IllegalArgumentException e) { t3 = true; }
        report("Wilcoxon rejects bad inputs", t1 && t2 && t3);
    }

    private static void testWilcoxonExactVsAsymptoticSwitch() {
        double[] a = new double[40];
        double[] b = new double[40];
        java.util.Random rng = new java.util.Random(13);
        for (int i = 0; i < a.length; i++) {
            a[i] = 0.5 + 0.01 * rng.nextGaussian();
            b[i] = 0.4 + 0.01 * rng.nextGaussian();
        }
        WilcoxonSignedRank.Result r = new WilcoxonSignedRank(true).test(a, b);
        boolean ok = !r.exact && r.rejectsNull(0.05);
        report("Wilcoxon switches to asymptotic when n>30 (exact=" + r.exact
                + ", p=" + r.pValue + ")", ok);
    }

    /* ---------- CDDiagramExporter ---------- */

    private static void testCDDiagramExporterRanks() throws Exception {
        Path tmp = Files.createTempFile("cd_ranks_", ".csv");
        List<String> methods = Arrays.asList("HT", "ARF", "SRP");
        double[] ranks = {2.5, 1.8, 1.7};
        CDDiagramExporter.writeRanks(tmp, methods, ranks, 0.42, 12, 0.05);
        List<String> lines = readAll(tmp);
        boolean ok = lines.size() == 5
                && lines.get(0).startsWith("# n_datasets=12")
                && lines.get(1).equals("method,avg_rank")
                && lines.get(2).startsWith("HT,")
                && lines.get(4).startsWith("SRP,");
        report("CDDiagramExporter.writeRanks emits header + rows", ok);
        Files.deleteIfExists(tmp);
    }

    private static void testCDDiagramExporterRankMatrix() throws Exception {
        Path tmp = Files.createTempFile("cd_matrix_", ".csv");
        List<String> datasets = Arrays.asList("d1", "d2");
        List<String> methods = Arrays.asList("A", "B");
        double[][] ranks = {{1.0, 2.0}, {2.0, 1.0}};
        CDDiagramExporter.writeRankMatrix(tmp, datasets, methods, ranks);
        List<String> lines = readAll(tmp);
        boolean ok = lines.size() == 3
                && lines.get(0).equals("dataset,A,B")
                && lines.get(1).equals("d1,1.0,2.0")
                && lines.get(2).equals("d2,2.0,1.0");
        report("CDDiagramExporter.writeRankMatrix shape correct", ok);
        Files.deleteIfExists(tmp);
    }

    private static void testCDDiagramExporterPairwiseSignificance() throws Exception {
        Path tmp = Files.createTempFile("cd_pairs_", ".csv");
        List<String> methods = Arrays.asList("A", "B", "C");
        boolean[][] sig = {
                {false, true, false},
                {true, false, true},
                {false, true, false}
        };
        double[][] diffs = {
                {0.0, 0.7, 0.2},
                {0.7, 0.0, 0.5},
                {0.2, 0.5, 0.0}
        };
        CDDiagramExporter.writePairwiseSignificance(tmp, methods, sig, diffs, 0.4);
        List<String> lines = readAll(tmp);
        boolean ok = lines.size() == 5
                && lines.get(0).startsWith("# CD=")
                && lines.get(1).equals("method_a,method_b,rank_diff,significant")
                && lines.get(2).equals("A,B,0.7,true")
                && lines.get(3).equals("A,C,0.2,false")
                && lines.get(4).equals("B,C,0.5,true");
        report("CDDiagramExporter.writePairwiseSignificance emits all pairs", ok);
        Files.deleteIfExists(tmp);
    }

    private static void testCDDiagramExporterEscapesCommas() throws Exception {
        Path tmp = Files.createTempFile("cd_esc_", ".csv");
        List<String> methods = Arrays.asList("plain", "has,comma", "has\"quote");
        double[] ranks = {1.0, 2.0, 3.0};
        CDDiagramExporter.writeRanks(tmp, methods, ranks, 0.1, 5, 0.05);
        List<String> lines = readAll(tmp);
        boolean ok = lines.get(2).equals("plain,1.0")
                && lines.get(3).equals("\"has,comma\",2.0")
                && lines.get(4).equals("\"has\"\"quote\",3.0");
        report("CDDiagramExporter escapes commas + quotes (line3=" + lines.get(3)
                + ", line4=" + lines.get(4) + ")", ok);
        Files.deleteIfExists(tmp);
    }

    private static List<String> readAll(Path p) throws Exception {
        List<String> out = new ArrayList<>();
        try (BufferedReader r = Files.newBufferedReader(p)) {
            String line;
            while ((line = r.readLine()) != null) out.add(line);
        }
        return out;
    }

    private static void report(String name, boolean ok) {
        if (ok) {
            passed++;
            System.out.println("  [PASSED] " + name);
        } else {
            failed++;
            System.out.println("  [FAILED] " + name);
        }
    }
}