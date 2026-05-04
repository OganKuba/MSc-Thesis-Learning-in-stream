package thesis.experiments;

import thesis.experiments.E4DriftAnalysis.GeneratorSpec;
import thesis.experiments.E4DriftAnalysis.Magnitude;
import thesis.experiments.E5Detectors.Cfg;
import thesis.experiments.E5Detectors.RunResult;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public class E5PrototypeRunner {

    static final class Probe {
        RunResult r;
        boolean alarmsObserved;
        boolean kappaSane;
        boolean delayPlausible;
        boolean falseAlarmReasonable;
        String tag = "PASS";
        StringBuilder reason = new StringBuilder();
    }

    public static void main(String[] args) throws Exception {

        Path cfgPath = args.length > 0
                ? Paths.get(args[0])
                : findDefaultConfig();

        if (!Files.exists(cfgPath)) {
            throw new RuntimeException(
                    "Config not found: " + cfgPath.toAbsolutePath() +
                            "\nWorking dir: " + Paths.get(".").toAbsolutePath()
            );
        }

        Cfg cfg = Cfg.load(cfgPath);
        new E5PrototypeRunner().run(cfg);
    }

    private static Path findDefaultConfig() {
        List<Path> candidates = List.of(
                Paths.get("src/main/java/thesis/experiments/e5_prototype.json"),
                Paths.get("experiments/e5_prototype.json"),
                Paths.get("src/main/resources/e5_prototype.json")
        );

        for (Path p : candidates) {
            if (Files.exists(p)) {
                return p;
            }
        }

        return candidates.get(0);
    }

    public void run(Cfg cfg) throws Exception {
        Files.createDirectories(Paths.get(cfg.outputDir));
        Path winCsv = Paths.get(cfg.outputDir, "E5_prototype.csv");
        Path alrCsv = Paths.get(cfg.outputDir, "E5_prototype_alarms.csv");
        List<RunResult> all = new ArrayList<>();
        E5Detectors runner = new E5Detectors();

        try (PrintWriter wcsv = new PrintWriter(new FileWriter(winCsv.toFile()));
             PrintWriter acsv = new PrintWriter(new FileWriter(alrCsv.toFile()))) {
            wcsv.println(E5Detectors.windowHeader());
            acsv.println(E5Detectors.alarmHeader());
            for (GeneratorSpec g : cfg.generators) {
                for (String mag : cfg.magnitudes) {
                    Magnitude m = Magnitude.valueOf(mag);
                    for (int seed : cfg.seeds) {
                        for (String model : cfg.models) {
                            for (String det : cfg.detectors) {
                                try {
                                    RunResult r = runner.runOne(cfg, g, m, det, model, seed, wcsv);
                                    E5Detectors.writeAlarmsCsv(acsv, r);
                                    all.add(r);
                                } catch (Exception e) {
                                    System.err.printf("[proto][FAIL] %s|%s|%s|%s|seed=%d -> %s%n",
                                            g.name, det, model, mag, seed, e);
                                    e.printStackTrace(System.err);
                                }
                            }
                        }
                    }
                }
            }
        }
        validateAndSummarize(cfg, all);
        E5Detectors.writeSummary(all, Paths.get(cfg.outputDir, "E5_prototype_summary.csv"));
        E5Detectors.writeAggregated(cfg, all);
        E5Detectors.writeRanking(cfg, all);
        E5Detectors.writeSanityChecks(cfg, all);
    }

    private void validateAndSummarize(Cfg cfg, List<RunResult> all) throws Exception {
        Map<String, Long> driftByKey = new HashMap<>();
        for (RunResult r : all) {
            String k = r.generator + "|" + r.magnitude + "|" + r.seed + "|" + r.model + "|" + r.detector;
            driftByKey.put(k, r.driftCount);
        }

        Path summaryPath = Paths.get(cfg.outputDir, "E5_prototype_summary.txt");
        try (PrintWriter w = new PrintWriter(new FileWriter(summaryPath.toFile()))) {
            w.println("=== E5 Prototype Summary ===");
            int pass = 0, fail = 0, warn = 0;
            boolean anyAlarms = false, anyTP = false, anyDifferent = false;

            Map<String, Long> firstDriftPerScenario = new HashMap<>();
            for (RunResult r : all) {
                String scen = r.generator + "|" + r.magnitude + "|" + r.seed + "|" + r.model;
                Long prev = firstDriftPerScenario.put(scen, r.driftCount);
                if (prev != null && !prev.equals(r.driftCount)) anyDifferent = true;
            }

            for (RunResult r : all) {
                Probe p = new Probe();
                p.r = r;
                p.alarmsObserved = !r.alarms.isEmpty();
                p.kappaSane = r.kappa > -0.5 && r.kappa < 1.0001 && !Double.isNaN(r.kappa);
                p.delayPlausible = r.detection.tp == 0
                        || (r.detection.meanDetectionDelay > 0.5 && r.detection.meanDetectionDelay < cfg.toleranceWindow + 1);
                long mon = r.instances - cfg.warmup;
                p.falseAlarmReasonable = mon <= 0 || r.detection.fp <= mon / 50;

                if (!p.kappaSane)              p.reason.append("kappa NaN/out-of-range; ");
                if (!p.alarmsObserved && r.abrupt) p.reason.append("no alarms on abrupt generator; ");
                if (!p.delayPlausible)         p.reason.append("delay implausible (=0 with TP>0); ");
                if (!p.falseAlarmReasonable)   p.reason.append("FP > 2% of stream; ");
                if (r.abrupt && r.detection.tp == 0 && r.gtPositions.length > 0)
                    p.reason.append("zero TP on abrupt; ");

                if (!p.kappaSane) { p.tag = "FAIL"; fail++; }
                else if (!p.delayPlausible || !p.falseAlarmReasonable) { p.tag = "WARN"; warn++; }
                else if (r.abrupt && r.detection.tp == 0 && r.gtPositions.length > 0) { p.tag = "WARN"; warn++; }
                else { p.tag = "PASS"; pass++; }

                if (p.alarmsObserved) anyAlarms = true;
                if (r.detection.tp > 0) anyTP = true;

                w.printf(Locale.ROOT,
                        "%-18s %-18s %-8s %-6s seed=%d  k=%.4f  drift=%d  TP=%d FP=%d FN=%d  P=%.3f R=%.3f F1=%.3f delay=%.1f recov=%.1f  %s %s%n",
                        r.generator, r.detector, r.model, r.magnitude, r.seed, r.kappa,
                        r.driftCount, r.detection.tp, r.detection.fp, r.detection.fn,
                        r.detection.precision, r.detection.recall, r.detection.f1,
                        r.detection.meanDetectionDelay, r.recoveryTime,
                        p.tag, p.reason.toString());
            }

            w.println();
            Map<String, Map<String, Long>> diffCheck = new HashMap<>();
            for (RunResult r : all) {
                String scen = r.generator + "|" + r.magnitude + "|" + r.seed + "|" + r.model;
                diffCheck.computeIfAbsent(scen, k -> new HashMap<>()).put(r.detector, r.driftCount);
            }
            w.println("--- DETECTOR DIFFERENTIATION CHECK ---");
            for (Map.Entry<String, Map<String, Long>> e : diffCheck.entrySet()) {
                Map<String, Long> dm = e.getValue();
                long mn = Long.MAX_VALUE, mx = Long.MIN_VALUE;
                for (long v : dm.values()) { if (v < mn) mn = v; if (v > mx) mx = v; }
                if (mn == mx && dm.size() > 1) {
                    w.println("  WARN_SAME " + e.getKey() + " all detectors fired " + mn + " times");
                } else {
                    w.println("  OK " + e.getKey() + " range=[" + mn + ".." + mx + "] " + dm);
                }
            }

            w.println();
            w.println("--- GLOBAL CHECKS ---");
            w.println("any alarms across runs       : " + anyAlarms);
            w.println("any TP across runs           : " + anyTP);
            w.println("detectors differ per scenario: " + anyDifferent);
            w.printf("--- TOTAL: %d PASS / %d FAIL / %d WARN ---%n", pass, fail, warn);

            System.out.println("------------------------------------------------------------");
            System.out.printf("[proto] TOTAL: %d PASS / %d FAIL / %d WARN%n", pass, fail, warn);
            System.out.println("[proto] anyAlarms=" + anyAlarms + " anyTP=" + anyTP + " anyDifferent=" + anyDifferent);
            System.out.println("[proto] details -> " + summaryPath);
        }
    }
}
