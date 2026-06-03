package thesis.experiments;

import thesis.evaluation.CDDiagramExporter;
import thesis.evaluation.FriedmanTest;
import thesis.evaluation.NemenyiPostHoc;
import thesis.evaluation.StatisticalTests;
import thesis.evaluation.WilcoxonSignedRank;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Per-block statistical hypothesis testing over the seed-averaged metrics produced
 * by {@link UnifiedStreamExperimentRunner}.
 *
 * <p>For each configured metric the analyser builds a {@code datasets × variants}
 * matrix (seed-averaged), runs Friedman + Nemenyi to produce average ranks and a
 * critical-difference diagram, then runs paired Wilcoxon signed-rank tests across
 * all (dataset, seed) pairs and Holm-adjusts the resulting p-values. Outputs land
 * in {@code results/<block>/stat_tests/}.
 *
 * <p>Reuses existing {@link FriedmanTest}, {@link NemenyiPostHoc},
 * {@link WilcoxonSignedRank}, {@link StatisticalTests} and {@link CDDiagramExporter}
 * machinery; no new statistics are reimplemented.
 */
public final class BlockStatisticalAnalysis {

    /** Metrics we test by default. */
    public enum Metric {
        ACCURACY("accuracy", true),
        KAPPA("kappa", true),
        KAPPA_PER("kappa_per", true),
        TEMPORAL_KAPPA("temporal_kappa", true),
        RECOVERY_TIME("recovery_time", false),
        RAM_HOURS_GB("ram_hours_gb", false);

        public final String csvName;
        public final boolean higherIsBetter;

        Metric(String csvName, boolean higherIsBetter) {
            this.csvName = csvName;
            this.higherIsBetter = higherIsBetter;
        }
    }

    private final double alpha;
    private final List<String> warnings = new ArrayList<>();

    public BlockStatisticalAnalysis() { this(0.05); }

    public BlockStatisticalAnalysis(double alpha) {
        if (!(alpha > 0.0 && alpha < 1.0))
            throw new IllegalArgumentException("alpha must be in (0,1)");
        this.alpha = alpha;
    }

    /** Public entry. Writes all stat-test artefacts for a single block. */
    public void analyseBlock(String blockId, List<UnifiedStreamExperimentRunner.RunArtifacts> runs,
                             Path outDir) throws IOException {
        if (runs == null) runs = Collections.emptyList();
        Files.createDirectories(outDir);

        // ranks.csv accumulates rows across all metrics; same for friedman/nemenyi/wilcoxon/cd.
        try (PrintWriter friedmanW = openWriter(outDir.resolve("friedman.csv"),
                "block,dataset_group,metric,num_algorithms,num_datasets,statistic,p_value,significant,alpha");
             PrintWriter nemenyiW = openWriter(outDir.resolve("nemenyi.csv"),
                     "block,metric,variant_a,variant_b,rank_diff,critical_difference,significant");
             PrintWriter wilcoxonW = openWriter(outDir.resolve("wilcoxon.csv"),
                     "block,metric,variant_a,variant_b,n,statistic,p_value,p_adjusted,significant,effect_size");
             PrintWriter ranksW = openWriter(outDir.resolve("ranks.csv"),
                     "block,metric,variant,avg_rank,mean_score,std_score,num_datasets");
             PrintWriter cdCsvW = openWriter(outDir.resolve("cd_diagram.csv"),
                     "block,metric,variant,avg_rank,critical_difference,num_datasets,alpha")) {

            // SVG/TeX are written per metric below.
            for (Metric m : Metric.values()) {
                analyseMetric(blockId, m, runs, outDir,
                        friedmanW, nemenyiW, wilcoxonW, ranksW, cdCsvW);
            }
        }

        if (!warnings.isEmpty()) {
            try (PrintWriter w = openWriter(outDir.resolve("warnings.txt"), null)) {
                for (String line : warnings) w.println(line);
            }
        }
    }

    public List<String> warnings() { return Collections.unmodifiableList(warnings); }

    // ------------------------------------------------------------------------

    private void analyseMetric(String blockId, Metric m,
                               List<UnifiedStreamExperimentRunner.RunArtifacts> runs,
                               Path outDir,
                               PrintWriter friedmanW, PrintWriter nemenyiW,
                               PrintWriter wilcoxonW, PrintWriter ranksW,
                               PrintWriter cdCsvW) throws IOException {

        // 1. Build per-(dataset, seed, variant) score table.
        TreeSet<String> variantSet = new TreeSet<>();
        TreeSet<String> datasetSet = new TreeSet<>();
        Map<String, Map<String, Double>> byVariant = new LinkedHashMap<>(); // variant -> "dataset|seed" -> score
        for (UnifiedStreamExperimentRunner.RunArtifacts ra : runs) {
            if (!ra.ok()) continue;
            UnifiedStreamExperimentRunner.RunResult r = ra.result;
            double v = extractMetric(m, ra);
            if (!Double.isFinite(v)) continue;
            variantSet.add(r.variant);
            datasetSet.add(r.dataset);
            String key = r.dataset + "|" + r.seed;
            byVariant.computeIfAbsent(r.variant, k -> new LinkedHashMap<>()).put(key, v);
        }
        List<String> variants = new ArrayList<>(variantSet);
        List<String> datasets = new ArrayList<>(datasetSet);

        if (variants.size() < 2) {
            warn(blockId, m, "fewer than 2 variants have finite values — skipping all tests");
            return;
        }

        // 2. Build per-dataset mean matrix (rows=datasets, cols=variants). Drop any
        //    dataset that lacks a value for some variant (Friedman requires complete blocks).
        List<String> usableDatasets = new ArrayList<>();
        List<double[]> rowList = new ArrayList<>();
        Map<String, double[]> perDatasetForVariant = new LinkedHashMap<>(); // variant -> per-dataset mean
        for (String v : variants) perDatasetForVariant.put(v, new double[datasets.size()]);

        for (int d = 0; d < datasets.size(); d++) {
            String ds = datasets.get(d);
            double[] row = new double[variants.size()];
            boolean ok = true;
            for (int j = 0; j < variants.size(); j++) {
                Map<String, Double> perKey = byVariant.get(variants.get(j));
                double sum = 0.0; int n = 0;
                if (perKey != null) {
                    for (Map.Entry<String, Double> e : perKey.entrySet()) {
                        if (e.getKey().startsWith(ds + "|")) { sum += e.getValue(); n++; }
                    }
                }
                if (n == 0) { ok = false; break; }
                double mean = sum / n;
                row[j] = mean;
                perDatasetForVariant.get(variants.get(j))[d] = mean;
            }
            if (ok) {
                usableDatasets.add(ds);
                rowList.add(row);
            }
        }

        // 3. ranks.csv per (metric, variant) — always written even if Friedman can't run.
        double[][] dmatrix = rowList.toArray(new double[0][]);
        double[] avgRanks = null;
        NemenyiPostHoc.Result nem = null;
        FriedmanTest.Result fr = null;
        boolean canDoFriedman = usableDatasets.size() >= 2 && variants.size() >= 2;

        if (canDoFriedman) {
            try {
                fr = new FriedmanTest(m.higherIsBetter).test(dmatrix);
                avgRanks = fr.averageRanks;
            } catch (Exception ex) {
                warn(blockId, m, "Friedman failed: " + ex.getMessage());
            }
        } else {
            warn(blockId, m, "need >= 2 complete datasets and >= 2 variants for Friedman; skipping");
        }

        // ranks: avg_rank, mean_score (across datasets), std_score (across datasets)
        for (int j = 0; j < variants.size(); j++) {
            double[] perDs = perDatasetForVariant.get(variants.get(j));
            // Restrict to usable datasets
            double[] used = new double[usableDatasets.size()];
            for (int k = 0; k < usableDatasets.size(); k++) {
                int origIdx = datasets.indexOf(usableDatasets.get(k));
                used[k] = perDs[origIdx];
            }
            double meanScore = mean(used);
            double stdScore = std(used);
            double r = (avgRanks != null) ? avgRanks[j] : Double.NaN;
            ranksW.printf(Locale.ROOT, "%s,%s,%s,%s,%s,%s,%d%n",
                    csv(blockId), m.csvName, csv(variants.get(j)),
                    fmt(r), fmt(meanScore), fmt(stdScore), usableDatasets.size());
        }

        // 4. friedman.csv
        if (fr != null) {
            // Prefer Iman-Davenport F p-value when available, fall back to chi-square p-value.
            double stat = Double.isFinite(fr.imanDavenport) ? fr.imanDavenport : fr.chiSquared;
            double p = Double.isFinite(fr.pValueF) ? fr.pValueF : fr.pValueChi;
            friedmanW.printf(Locale.ROOT, "%s,%s,%s,%d,%d,%s,%s,%d,%s%n",
                    csv(blockId), "all", m.csvName,
                    fr.numMethods, fr.numDatasets,
                    fmt(stat), fmt(p), p < alpha ? 1 : 0, fmt(alpha));
        }

        // 5. Nemenyi — needs avgRanks and the q-table only supports k <= 20.
        if (fr != null && fr.numMethods <= 20) {
            try {
                nem = new NemenyiPostHoc(alpha).test(fr.averageRanks, fr.numDatasets);
                for (int i = 0; i < variants.size(); i++) {
                    for (int j = i + 1; j < variants.size(); j++) {
                        double diff = nem.rankDifferences[i][j];
                        boolean sig = nem.significant[i][j];
                        nemenyiW.printf(Locale.ROOT, "%s,%s,%s,%s,%s,%s,%d%n",
                                csv(blockId), m.csvName,
                                csv(variants.get(i)), csv(variants.get(j)),
                                fmt(diff), fmt(nem.criticalDifference), sig ? 1 : 0);
                    }
                }
            } catch (Exception ex) {
                warn(blockId, m, "Nemenyi failed: " + ex.getMessage());
            }
        } else if (fr != null) {
            warn(blockId, m, "Nemenyi q-table tops out at k=20; got k=" + fr.numMethods + " — skipping");
        }

        // 6. cd_diagram.* — written even if Nemenyi unavailable (cd_diagram.csv records ranks only).
        writeCDDiagram(outDir, blockId, m, variants, fr, nem, cdCsvW);

        // 7. Wilcoxon — pair variants on aligned (dataset, seed) vectors. Holm-adjust within metric.
        runWilcoxonAndWrite(blockId, m, variants, byVariant, wilcoxonW);
    }

    // ------------------------------------------------------------------------
    // Per-metric helpers
    // ------------------------------------------------------------------------

    private static double extractMetric(Metric m, UnifiedStreamExperimentRunner.RunArtifacts ra) {
        UnifiedStreamExperimentRunner.RunResult r = ra.result;
        switch (m) {
            case ACCURACY:       return r.accuracy;
            case KAPPA:          return r.kappa;
            case KAPPA_PER:      return r.kappaPer;
            case TEMPORAL_KAPPA: return ra.recorder == null ? Double.NaN : ra.recorder.meanTemporalKappa();
            case RECOVERY_TIME:  return ra.recorder == null ? Double.NaN : ra.recorder.meanRecoveryLength();
            case RAM_HOURS_GB:   return r.ramHoursGB;
            default: return Double.NaN;
        }
    }

    private void runWilcoxonAndWrite(String blockId, Metric m, List<String> variants,
                                     Map<String, Map<String, Double>> byVariant,
                                     PrintWriter wilcoxonW) {
        WilcoxonSignedRank w = new WilcoxonSignedRank();
        List<double[]> pairResults = new ArrayList<>(); // [i, j, stat, p, effect, n]
        for (int i = 0; i < variants.size(); i++) {
            for (int j = i + 1; j < variants.size(); j++) {
                Map<String, Double> pa = byVariant.getOrDefault(variants.get(i), Collections.emptyMap());
                Map<String, Double> pb = byVariant.getOrDefault(variants.get(j), Collections.emptyMap());
                Set<String> commonKeys = new TreeSet<>(pa.keySet());
                commonKeys.retainAll(pb.keySet());
                if (commonKeys.size() < 2) {
                    warn(blockId, m, String.format(Locale.ROOT,
                            "Wilcoxon %s vs %s skipped: only %d aligned (dataset,seed) pairs",
                            variants.get(i), variants.get(j), commonKeys.size()));
                    pairResults.add(new double[]{i, j, Double.NaN, Double.NaN, Double.NaN, commonKeys.size()});
                    continue;
                }
                double[] aArr = new double[commonKeys.size()];
                double[] bArr = new double[commonKeys.size()];
                int p = 0;
                for (String k : commonKeys) { aArr[p] = pa.get(k); bArr[p] = pb.get(k); p++; }
                try {
                    WilcoxonSignedRank.Result wr = w.test(aArr, bArr);
                    double effect = computeRankBiserial(aArr, bArr);
                    pairResults.add(new double[]{
                            i, j, wr.statistic, wr.pValue, effect, wr.effectiveN});
                } catch (Exception ex) {
                    warn(blockId, m, String.format(Locale.ROOT,
                            "Wilcoxon %s vs %s failed: %s",
                            variants.get(i), variants.get(j), ex.getMessage()));
                    pairResults.add(new double[]{i, j, Double.NaN, Double.NaN, Double.NaN, commonKeys.size()});
                }
            }
        }
        // Holm adjustment over the finite p-values of this metric.
        double[] padj = holmAdjust(pairResults);
        for (int idx = 0; idx < pairResults.size(); idx++) {
            double[] pr = pairResults.get(idx);
            int i = (int) pr[0], j = (int) pr[1];
            double stat = pr[2], pval = pr[3], effect = pr[4];
            int n = (int) pr[5];
            wilcoxonW.printf(Locale.ROOT, "%s,%s,%s,%s,%d,%s,%s,%s,%d,%s%n",
                    csv(blockId), m.csvName,
                    csv(variants.get(i)), csv(variants.get(j)),
                    n, fmt(stat), fmt(pval), fmt(padj[idx]),
                    (Double.isFinite(padj[idx]) && padj[idx] < alpha) ? 1 : 0,
                    fmt(effect));
        }
    }

    /** Holm step-down adjustment over the p-value field of each test row (NaNs left as NaN). */
    private static double[] holmAdjust(List<double[]> tests) {
        int m = tests.size();
        Integer[] order = new Integer[m];
        for (int i = 0; i < m; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> {
            double pa = tests.get(a)[3];
            double pb = tests.get(b)[3];
            if (Double.isNaN(pa) && Double.isNaN(pb)) return 0;
            if (Double.isNaN(pa)) return 1;
            if (Double.isNaN(pb)) return -1;
            return Double.compare(pa, pb);
        });
        // count finite p-values
        int finite = 0;
        for (int i = 0; i < m; i++) if (Double.isFinite(tests.get(i)[3])) finite++;
        double[] out = new double[m];
        Arrays.fill(out, Double.NaN);
        double prev = 0.0;
        int seen = 0;
        for (int rank = 0; rank < m; rank++) {
            int origIdx = order[rank];
            double p = tests.get(origIdx)[3];
            if (!Double.isFinite(p)) continue;
            int multiplier = finite - seen;
            double adj = Math.min(1.0, p * multiplier);
            if (adj < prev) adj = prev;
            out[origIdx] = adj;
            prev = adj;
            seen++;
        }
        return out;
    }

    /** Rank-biserial effect size: (wins - losses) / (wins + losses + ties). */
    private static double computeRankBiserial(double[] a, double[] b) {
        int wins = 0, losses = 0, ties = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            if (d > 0) wins++;
            else if (d < 0) losses++;
            else ties++;
        }
        int total = wins + losses + ties;
        if (total == 0) return Double.NaN;
        return (double) (wins - losses) / total;
    }

    // ------------------------------------------------------------------------
    // CD diagram writers (CSV + SVG + TeX)
    // ------------------------------------------------------------------------

    private void writeCDDiagram(Path outDir, String blockId, Metric m, List<String> variants,
                                FriedmanTest.Result fr, NemenyiPostHoc.Result nem,
                                PrintWriter cdCsvW) throws IOException {
        double cd = nem == null ? Double.NaN : nem.criticalDifference;
        int nd = fr == null ? 0 : fr.numDatasets;
        double[] ranks = fr == null ? null : fr.averageRanks;
        for (int j = 0; j < variants.size(); j++) {
            double r = ranks == null ? Double.NaN : ranks[j];
            cdCsvW.printf(Locale.ROOT, "%s,%s,%s,%s,%s,%d,%s%n",
                    csv(blockId), m.csvName, csv(variants.get(j)),
                    fmt(r), fmt(cd), nd, fmt(alpha));
        }

        if (ranks == null || !Double.isFinite(cd)) return;
        // Per-metric SVG + TeX, only when we actually have ranks and CD.
        writeCDDiagramSvg(outDir.resolve("cd_diagram_" + m.csvName + ".svg"),
                blockId, m, variants, ranks, cd);
        writeCDDiagramTex(outDir.resolve("cd_diagram_" + m.csvName + ".tex"),
                blockId, m, variants, ranks, cd);
        // The combined ranks file used by avg_ranks.csv / rank_matrix.csv consumers
        // mirrors what CDDiagramExporter writes for downstream tooling.
        CDDiagramExporter.writeRanks(outDir.resolve("avg_ranks_" + m.csvName + ".csv"),
                variants, ranks, cd, nd, alpha);
        if (fr.ranks != null) {
            // Build dataset names list of length fr.numDatasets — use the placeholder D1..Dn since
            // the per-dataset ordering isn't carried through after the Friedman matrix is built.
            List<String> dsNames = new ArrayList<>();
            for (int i = 0; i < fr.numDatasets; i++) dsNames.add("D" + (i + 1));
            CDDiagramExporter.writeRankMatrix(outDir.resolve("rank_matrix_" + m.csvName + ".csv"),
                    dsNames, variants, fr.ranks);
        }
        if (nem != null) {
            CDDiagramExporter.writePairwiseSignificance(
                    outDir.resolve("pairwise_significance_" + m.csvName + ".csv"),
                    variants, nem.significant, nem.rankDifferences, nem.criticalDifference);
        }
    }

    private static void writeCDDiagramSvg(Path out, String blockId, Metric m,
                                          List<String> variants, double[] ranks, double cd)
            throws IOException {
        Integer[] order = new Integer[variants.size()];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(ranks[a], ranks[b]));

        double minR = ranks[order[0]];
        double maxR = ranks[order[order.length - 1]];
        double pad = 0.5;
        double xLo = Math.max(1.0, minR - pad);
        double xHi = Math.min(variants.size(), maxR + pad);

        int width = 900;
        int height = 40 + 24 * variants.size();
        double xScale = (width - 200.0) / Math.max(1e-9, xHi - xLo);

        try (PrintWriter w = openWriter(out, null)) {
            w.printf(Locale.ROOT,
                    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" "
                            + "viewBox=\"0 0 %d %d\">%n", width, height, width, height);
            w.printf(Locale.ROOT,
                    "<text x=\"10\" y=\"18\" font-family=\"sans-serif\" font-size=\"14\">"
                            + "%s — %s (alpha=%.3f, CD=%.3f)</text>%n",
                    xmlEscape(blockId), xmlEscape(m.csvName), 0.05, cd);
            // Axis
            int axisY = 50;
            w.printf(Locale.ROOT,
                    "<line x1=\"100\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"black\"/>%n",
                    axisY, width - 100, axisY);
            for (int t = (int) Math.floor(xLo); t <= (int) Math.ceil(xHi); t++) {
                double x = 100 + (t - xLo) * xScale;
                w.printf(Locale.ROOT,
                        "<line x1=\"%.1f\" y1=\"%d\" x2=\"%.1f\" y2=\"%d\" stroke=\"black\"/>%n",
                        x, axisY - 4, x, axisY + 4);
                w.printf(Locale.ROOT,
                        "<text x=\"%.1f\" y=\"%d\" font-family=\"sans-serif\" font-size=\"10\" "
                                + "text-anchor=\"middle\">%d</text>%n", x, axisY - 8, t);
            }
            // CD bar
            double cdX1 = 100;
            double cdX2 = 100 + cd * xScale;
            w.printf(Locale.ROOT,
                    "<line x1=\"%.1f\" y1=\"%d\" x2=\"%.1f\" y2=\"%d\" "
                            + "stroke=\"red\" stroke-width=\"3\"/>%n",
                    cdX1, axisY - 20, cdX2, axisY - 20);
            w.printf(Locale.ROOT,
                    "<text x=\"%.1f\" y=\"%d\" font-family=\"sans-serif\" font-size=\"10\" "
                            + "fill=\"red\">CD=%.3f</text>%n",
                    cdX1, axisY - 24, cd);
            // Variant dots/labels (lower rank == better, drawn first)
            for (int rankIdx = 0; rankIdx < order.length; rankIdx++) {
                int j = order[rankIdx];
                double x = 100 + (ranks[j] - xLo) * xScale;
                double y = axisY + 30 + rankIdx * 20;
                w.printf(Locale.ROOT,
                        "<line x1=\"%.1f\" y1=\"%d\" x2=\"%.1f\" y2=\"%.1f\" stroke=\"gray\" "
                                + "stroke-dasharray=\"3,2\"/>%n",
                        x, axisY, x, y);
                w.printf(Locale.ROOT,
                        "<circle cx=\"%.1f\" cy=\"%.1f\" r=\"4\" fill=\"steelblue\"/>%n", x, y);
                w.printf(Locale.ROOT,
                        "<text x=\"%.1f\" y=\"%.1f\" font-family=\"sans-serif\" font-size=\"11\">"
                                + "%s (%.2f)</text>%n",
                        x + 8, y + 4, xmlEscape(variants.get(j)), ranks[j]);
            }
            w.println("</svg>");
        }
    }

    private static void writeCDDiagramTex(Path out, String blockId, Metric m,
                                          List<String> variants, double[] ranks, double cd)
            throws IOException {
        try (PrintWriter w = openWriter(out, null)) {
            w.printf(Locale.ROOT, "%% CD diagram block=%s metric=%s alpha=%.3f CD=%.4f%n",
                    blockId, m.csvName, 0.05, cd);
            w.println("\\begin{tikzpicture}[xscale=2.0]");
            double minR = Double.POSITIVE_INFINITY, maxR = Double.NEGATIVE_INFINITY;
            for (double r : ranks) { if (r < minR) minR = r; if (r > maxR) maxR = r; }
            double xLo = Math.max(1.0, minR - 0.5);
            double xHi = Math.min(variants.size(), maxR + 0.5);
            w.printf(Locale.ROOT, "\\draw[->] (%.2f,0) -- (%.2f,0);%n", xLo, xHi);
            for (int t = (int) Math.floor(xLo); t <= (int) Math.ceil(xHi); t++) {
                w.printf(Locale.ROOT, "\\draw (%d,0.1) -- (%d,-0.1) node[below] {%d};%n", t, t, t);
            }
            w.printf(Locale.ROOT,
                    "\\draw[red, very thick] (%.2f,0.5) -- (%.2f,0.5) node[midway, above] {CD=%.3f};%n",
                    xLo, xLo + cd, cd);
            Integer[] order = new Integer[variants.size()];
            for (int i = 0; i < order.length; i++) order[i] = i;
            Arrays.sort(order, (a, b) -> Double.compare(ranks[a], ranks[b]));
            for (int idx = 0; idx < order.length; idx++) {
                int j = order[idx];
                w.printf(Locale.ROOT,
                        "\\draw[dashed] (%.3f,0) -- (%.3f,-%.2f) node[right] {%s (%.2f)};%n",
                        ranks[j], ranks[j], 0.6 + 0.4 * idx, escapeTex(variants.get(j)), ranks[j]);
            }
            w.println("\\end{tikzpicture}");
        }
    }

    // ------------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------------

    private static PrintWriter openWriter(Path file, String header) throws IOException {
        Files.createDirectories(file.toAbsolutePath().getParent());
        PrintWriter w = new PrintWriter(Files.newBufferedWriter(file, StandardCharsets.UTF_8));
        if (header != null) w.println(header);
        return w;
    }

    private void warn(String blockId, Metric m, String msg) {
        String line = String.format(Locale.ROOT, "[%s/%s] %s", blockId, m.csvName, msg);
        warnings.add(line);
        System.err.println("[Unified][stat] " + line);
    }

    private static double mean(double[] x) {
        if (x == null || x.length == 0) return Double.NaN;
        double s = 0.0; int n = 0;
        for (double v : x) if (Double.isFinite(v)) { s += v; n++; }
        return n == 0 ? Double.NaN : s / n;
    }

    private static double std(double[] x) {
        double m = mean(x);
        if (!Double.isFinite(m)) return Double.NaN;
        double s = 0.0; int n = 0;
        for (double v : x) if (Double.isFinite(v)) { s += (v - m) * (v - m); n++; }
        return n < 2 ? 0.0 : Math.sqrt(s / (n - 1));
    }

    private static String fmt(double v) {
        if (Double.isNaN(v)) return "NaN";
        if (Double.isInfinite(v)) return v > 0 ? "Infinity" : "-Infinity";
        return String.format(Locale.ROOT, "%.6f", v);
    }

    private static String csv(String s) {
        if (s == null) return "";
        if (s.indexOf(',') < 0 && s.indexOf('"') < 0 && s.indexOf('\n') < 0) return s;
        return "\"" + s.replace("\"", "\"\"") + "\"";
    }

    private static String xmlEscape(String s) {
        if (s == null) return "";
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '<': b.append("&lt;"); break;
                case '>': b.append("&gt;"); break;
                case '&': b.append("&amp;"); break;
                case '"': b.append("&quot;"); break;
                default: b.append(c);
            }
        }
        return b.toString();
    }

    private static String escapeTex(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\textbackslash{}")
                .replace("_", "\\_")
                .replace("%", "\\%")
                .replace("&", "\\&")
                .replace("#", "\\#")
                .replace("$", "\\$")
                .replace("{", "\\{")
                .replace("}", "\\}")
                .replace("^", "\\^{}")
                .replace("~", "\\~{}");
    }

    /** Compact summary for logging/tests. */
    public String summary() {
        if (warnings.isEmpty()) return "no warnings";
        StringWriter sw = new StringWriter();
        sw.write("warnings:\n");
        for (String w : warnings) { sw.write("  - "); sw.write(w); sw.write('\n'); }
        return sw.toString();
    }
}
