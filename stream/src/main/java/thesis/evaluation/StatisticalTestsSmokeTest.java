package thesis.evaluation;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

public class StatisticalTestsSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) throws Exception {
        System.out.println("=".repeat(70));
        System.out.println("STATISTICAL TESTS SMOKE TESTS");
        System.out.println("=".repeat(70));

        testFriedmanRejectsEmpty();
        testFriedmanRejectsTooFewMethodsOrDatasets();
        testFriedmanRejectsNonFinite();
        testFriedmanIdenticalGivesEqualRanks();
        testFriedmanTiesGetAverageRanks();
        testFriedmanDetectsClearDifference();
        testFriedmanLowerIsBetterFlips();

        testNemenyiRejectsBadAlpha();
        testNemenyiNoFalseSigOnIdenticalRanks();
        testNemenyiCDFormula();
        testNemenyiSignificantOnlyAboveCD();

        testWilcoxonAllZerosDegenerate();
        testWilcoxonDropZeros();
        testWilcoxonDetectsConsistentWin();
        testWilcoxonRejectsNonFinite();
        testWilcoxonRejectsTooSmallN();

        testPairwiseAllIdenticalIsAllOne();
        testRunFullProducesReportAndCsvs();
        testCsvLongScoresRoundTrip();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    private static void testFriedmanRejectsEmpty() {
        boolean t = false;
        try { new FriedmanTest().test(new double[0][0]); } catch (IllegalArgumentException e) { t = true; }
        report("Friedman rejects empty matrix", t);
    }

    private static void testFriedmanRejectsTooFewMethodsOrDatasets() {
        boolean t1 = false, t2 = false;
        try { new FriedmanTest().test(new double[][]{{1}}); } catch (IllegalArgumentException e) { t1 = true; }
        try { new FriedmanTest().test(new double[][]{{1, 2}}); } catch (IllegalArgumentException e) { t2 = true; }
        report("Friedman rejects k<2 and n<2", t1 && t2);
    }

    private static void testFriedmanRejectsNonFinite() {
        boolean t = false;
        try { new FriedmanTest().test(new double[][]{{1, Double.NaN}, {2, 3}}); }
        catch (IllegalArgumentException e) { t = true; }
        report("Friedman rejects non-finite", t);
    }

    private static void testFriedmanIdenticalGivesEqualRanks() {
        double[][] m = { {0.8, 0.8, 0.8, 0.8}, {0.7, 0.7, 0.7, 0.7}, {0.9, 0.9, 0.9, 0.9} };
        FriedmanTest.Result r = new FriedmanTest().test(m);
        boolean equal = true;
        double avg = (1 + 4) / 2.0;
        for (double a : r.averageRanks) if (Math.abs(a - avg) > 1e-9) equal = false;
        report("Friedman: identical scores → all ranks=2.5 (chi2=" + r.chiSquared + ")",
                equal && r.chiSquared < 1e-9 && r.pValueChi > 0.99);
    }

    private static void testFriedmanTiesGetAverageRanks() {
        double[][] m = { {1.0, 1.0, 0.5}, {0.5, 1.0, 1.0} };
        FriedmanTest.Result r = new FriedmanTest().test(m);
        boolean ok = Math.abs(r.ranks[0][0] - 1.5) < 1e-9
                && Math.abs(r.ranks[0][1] - 1.5) < 1e-9
                && Math.abs(r.ranks[0][2] - 3.0) < 1e-9;
        report("Friedman: ties get averaged ranks", ok);
    }

    private static void testFriedmanDetectsClearDifference() {
        double[][] m = new double[10][3];
        for (int i = 0; i < 10; i++) { m[i][0] = 0.9; m[i][1] = 0.7; m[i][2] = 0.5; }
        FriedmanTest.Result r = new FriedmanTest().test(m);
        report("Friedman detects clear difference (p=" + r.pValueChi + ")", r.pValueChi < 0.01);
    }

    private static void testFriedmanLowerIsBetterFlips() {
        double[][] m = { {0.1, 0.5, 0.9} };
        double[][] m2 = { {0.1, 0.5, 0.9}, {0.2, 0.4, 0.95} };
        FriedmanTest.Result rH = new FriedmanTest(true).test(m2);
        FriedmanTest.Result rL = new FriedmanTest(false).test(m2);
        report("Friedman: higherIsBetter flips ranks (h[0]=" + rH.averageRanks[0]
                        + ", l[0]=" + rL.averageRanks[0] + ")",
                rH.averageRanks[0] > rH.averageRanks[2] && rL.averageRanks[0] < rL.averageRanks[2]);
    }

    private static void testNemenyiRejectsBadAlpha() {
        boolean t = false;
        try { new NemenyiPostHoc(0.025); } catch (IllegalArgumentException e) { t = true; }
        report("Nemenyi rejects unsupported alpha", t);
    }

    private static void testNemenyiNoFalseSigOnIdenticalRanks() {
        double[] r = {2.5, 2.5, 2.5, 2.5};
        NemenyiPostHoc.Result nr = new NemenyiPostHoc(0.05).test(r, 5);
        report("Nemenyi: identical ranks → no significant pair", nr.significantPairs.isEmpty());
    }

    private static void testNemenyiCDFormula() {
        int k = 4, n = 10;
        NemenyiPostHoc.Result nr = new NemenyiPostHoc(0.05).test(new double[]{1, 2, 3, 4}, n);
        double expected = nr.qAlpha * Math.sqrt(((double) k * (k + 1)) / (6.0 * n));
        report("Nemenyi CD = q*sqrt(k(k+1)/6n) (got " + nr.criticalDifference
                        + " expected " + expected + ")",
                Math.abs(nr.criticalDifference - expected) < 1e-9);
    }

    private static void testNemenyiSignificantOnlyAboveCD() {
        NemenyiPostHoc.Result nr = new NemenyiPostHoc(0.05).test(new double[]{1, 4}, 100);
        report("Nemenyi: large diff small CD → significant",
                nr.significantPairs.size() == 1);
    }

    private static void testWilcoxonAllZerosDegenerate() {
        double[] a = {0.5, 0.5, 0.5, 0.5};
        double[] b = {0.5, 0.5, 0.5, 0.5};
        WilcoxonSignedRank.Result r = new WilcoxonSignedRank().test(a, b);
        report("Wilcoxon: all ties → degenerate, p=1, no rejection",
                r.degenerate && Math.abs(r.pValue - 1.0) < 1e-9 && !r.rejectsNull(0.05));
    }

    private static void testWilcoxonDropZeros() {
        double[] a = {0.5, 0.6, 0.7, 0.8, 0.5};
        double[] b = {0.5, 0.4, 0.5, 0.6, 0.5};
        WilcoxonSignedRank.Result r = new WilcoxonSignedRank(true, true).test(a, b);
        report("Wilcoxon dropZeros: effectiveN=3, ties=2 (got eff="
                        + r.effectiveN + ", ties=" + r.ties + ")",
                r.effectiveN == 3 && r.ties == 2);
    }

    private static void testWilcoxonDetectsConsistentWin() {
        double[] a = {0.9, 0.85, 0.92, 0.88, 0.91, 0.87, 0.93, 0.89};
        double[] b = {0.7, 0.65, 0.72, 0.68, 0.71, 0.67, 0.73, 0.69};
        WilcoxonSignedRank.Result r = new WilcoxonSignedRank().test(a, b);
        report("Wilcoxon detects consistent win (p=" + r.pValue + ")", r.pValue < 0.05);
    }

    private static void testWilcoxonRejectsNonFinite() {
        boolean t = false;
        try { new WilcoxonSignedRank().test(new double[]{1, Double.NaN}, new double[]{1, 2}); }
        catch (IllegalArgumentException e) { t = true; }
        report("Wilcoxon rejects NaN", t);
    }

    private static void testWilcoxonRejectsTooSmallN() {
        boolean t = false;
        try { new WilcoxonSignedRank().test(new double[]{1}, new double[]{2}); }
        catch (IllegalArgumentException e) { t = true; }
        report("Wilcoxon rejects n<2", t);
    }

    private static void testPairwiseAllIdenticalIsAllOne() {
        double[][] m = new double[6][4];
        for (int i = 0; i < 6; i++) for (int j = 0; j < 4; j++) m[i][j] = 0.8;
        StatisticalTests st = new StatisticalTests();
        double[][] p = st.pairwiseWilcoxon(m);
        boolean ok = true;
        for (int i = 0; i < 4; i++)
            for (int j = i + 1; j < 4; j++)
                if (!(p[i][j] >= 0.99)) ok = false;
        report("pairwiseWilcoxon: identical methods → p≥0.99 for all pairs", ok);
    }

    private static void testRunFullProducesReportAndCsvs() throws Exception {
        double[][] m = {
                {0.90, 0.80, 0.70, 0.60},
                {0.92, 0.81, 0.69, 0.61},
                {0.91, 0.82, 0.71, 0.59},
                {0.93, 0.79, 0.70, 0.62},
                {0.94, 0.83, 0.72, 0.58}
        };
        StatisticalTests st = new StatisticalTests(0.05, true);
        StatisticalTests.Report rep = st.runFull(m,
                List.of("ds1", "ds2", "ds3", "ds4", "ds5"),
                List.of("M1", "M2", "M3", "M4"));
        Path tmp = Files.createTempDirectory("stat_smoke");
        rep.exportCD(tmp);
        boolean filesExist = Files.exists(tmp.resolve("avg_ranks.csv"))
                && Files.exists(tmp.resolve("rank_matrix.csv"))
                && Files.exists(tmp.resolve("pairwise_significance.csv"));
        report("runFull produces CD csvs (" + tmp + ")", filesExist
                && rep.friedman.pValueChi < 0.05
                && rep.summary().contains("Friedman"));
    }

    private static void testCsvLongScoresRoundTrip() throws Exception {
        Path tmp = Files.createTempDirectory("stat_long");
        Path file = tmp.resolve("long_scores.csv");
        double[][][] sc = new double[2][3][2];
        for (int d = 0; d < 2; d++)
            for (int m = 0; m < 3; m++)
                for (int s = 0; s < 2; s++) sc[d][m][s] = 0.5 + 0.1 * m + 0.01 * s + 0.02 * d;
        double[] avgRanks = {1.0, 2.0, 3.0};
        double[][] perDsRanks = {{1, 2, 3}, {1, 2, 3}};
        double[][] pairP = {{Double.NaN, 0.04, 0.001}, {0.04, Double.NaN, 0.5}, {0.001, 0.5, Double.NaN}};
        CDDiagramExporter.writeLongScores(file, List.of("M1", "M2", "M3"),
                List.of("ds1", "ds2"), List.of(7, 11), sc, avgRanks, perDsRanks, pairP);
        List<String> lines = Files.readAllLines(file);
        boolean headerOk = lines.get(0).startsWith("method,dataset,seed,score,rank,avg_rank");
        boolean rowsOk = lines.size() == 1 + 2 * 3 * 2;
        report("long scores CSV header+rows (lines=" + lines.size() + ")", headerOk && rowsOk);
    }

    private static void report(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  [PASSED] " + name); }
        else    { failed++; System.out.println("  [FAILED] " + name); }
    }
}