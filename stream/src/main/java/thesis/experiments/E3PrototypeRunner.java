//package thesis.experiments;
//
//import thesis.experiments.E2AdaptiveFS.DatasetSpec;
//import thesis.experiments.E3DASRP.Cfg;
//import thesis.experiments.E3DASRP.RunResult;
//import thesis.experiments.E3DASRP.Variant;
//
//import java.io.FileWriter;
//import java.io.PrintWriter;
//import java.nio.file.Files;
//import java.nio.file.Path;
//import java.nio.file.Paths;
//import java.util.ArrayList;
//import java.util.HashMap;
//import java.util.List;
//import java.util.Locale;
//import java.util.Map;
//
//public class E3PrototypeRunner {
//
//    static final class Probe {
//        RunResult r;
//        boolean componentAActive;
//        boolean componentBActive;
//        boolean componentCActive;
//        boolean differsFromSrpS1Kappa;
//        boolean driftingFeaturesObserved;
//        String tag = "PASS";
//        StringBuilder reason = new StringBuilder();
//    }
//
//    public static void main(String[] args) throws Exception {
//
//        Path cfgPath = args.length > 0
//                ? Paths.get(args[0])
//                : Paths.get("src/main/java/thesis/experiments/e3_prototype.json");
//
//        if (!Files.exists(cfgPath)) {
//            throw new RuntimeException(
//                    "Config not found: " + cfgPath.toAbsolutePath() +
//                            "\nWorking dir: " + Paths.get(".").toAbsolutePath()
//            );
//        }
//
//        Cfg cfg = Cfg.load(cfgPath);
//        new E3PrototypeRunner().run(cfg);
//    }
//
//    public void run(Cfg cfg) throws Exception {
//        Files.createDirectories(Paths.get(cfg.outputDir));
//        Path winCsv = Paths.get(cfg.outputDir, "E3_prototype.csv");
//        List<RunResult> all = new ArrayList<>();
//        E3DASRP runner = new E3DASRP();
//
//        try (PrintWriter csv = new PrintWriter(new FileWriter(winCsv.toFile()))) {
//            csv.println(E3DASRP.windowHeader());
//            for (DatasetSpec ds : cfg.datasets) {
//                if ("arff".equalsIgnoreCase(ds.type)
//                        && (ds.path == null || !Files.exists(Paths.get(ds.path)))) {
//                    System.err.printf("[proto] skip arff %s%n", ds.name);
//                    if (cfg.skipMissingArff) continue;
//                    throw new java.io.FileNotFoundException(String.valueOf(ds.path));
//                }
//                for (int seed : cfg.seeds) {
//                    for (Variant v : cfg.variants) {
//                        try {
//                            RunResult r = runner.runOne(cfg, ds, v, seed, csv);
//                            all.add(r);
//                        } catch (Exception e) {
//                            System.err.printf("[proto][FAIL] %s|%s|seed=%d -> %s%n",
//                                    ds.name, v.name, seed, e);
//                            e.printStackTrace(System.err);
//                        }
//                    }
//                }
//            }
//        }
//        validateAndSummarize(cfg, all);
//    }
//
//    private void validateAndSummarize(Cfg cfg, List<RunResult> all) throws Exception {
//        Map<String, Double> srpS1Kappa = new HashMap<>();
//        for (RunResult r : all) {
//            if ("SRP+S1".equals(r.variant)) {
//                srpS1Kappa.put(r.dataset + "#" + r.seed, r.kappa);
//            }
//        }
//
//        try (PrintWriter w = new PrintWriter(new FileWriter(
//                Paths.get(cfg.outputDir, "E3_prototype_summary.txt").toFile()))) {
//
//            w.println("=== E3 Prototype Summary ===");
//            int pass = 0, fail = 0, warn = 0;
//            boolean anyAActive = false, anyBActive = false, anyCActive = false;
//            boolean anyDriftingObserved = false;
//
//            for (RunResult r : all) {
//                Probe p = new Probe();
//                p.r = r;
//                Double s1k = srpS1Kappa.get(r.dataset + "#" + r.seed);
//
//                boolean isDA = r.variant.startsWith("DA-SRP");
//                p.componentAActive = isDA && (r.totalSurgical > 0 || r.totalFull > 0 || r.totalKept > 0);
//                p.componentBActive = isDA && r.variant.contains("AB") && r.totalSurgical + r.totalFull > 0;
//                p.componentCActive = "DA-SRP-ABC".equals(r.variant) && r.weightedPredictions > 0;
//                p.driftingFeaturesObserved = r.driftCount > 0;
//                p.differsFromSrpS1Kappa = isDA && s1k != null && Math.abs(r.kappa - s1k) > 1e-6;
//
//                if (isDA) {
//                    if (!p.componentAActive) p.reason.append("A inactive (no surg/full/kept); ");
//                    if (r.variant.contains("AB") && r.totalSurgical == 0)
//                        p.reason.append("B suspect (no surgical updates); ");
//                    if ("DA-SRP-ABC".equals(r.variant) && r.weightedPredictions == 0)
//                        p.reason.append("C inactive (no weighted predictions); ");
//                    if (s1k != null && !p.differsFromSrpS1Kappa)
//                        p.reason.append("kappa==SRP+S1; ");
//                    if (r.driftCount == 0)
//                        p.reason.append("no drift alarms in run; ");
//                }
//
//                if (isDA) {
//                    boolean ok = p.componentAActive
//                            && (!r.variant.contains("AB") || r.totalSurgical > 0)
//                            && (!"DA-SRP-ABC".equals(r.variant) || r.weightedPredictions > 0);
//                    if (!ok) { p.tag = "FAIL"; fail++; }
//                    else if (r.driftCount == 0 || (s1k != null && !p.differsFromSrpS1Kappa)) {
//                        p.tag = "WARN"; warn++;
//                    } else { p.tag = "PASS"; pass++; }
//                } else {
//                    p.tag = "PASS"; pass++;
//                }
//
//                if (p.componentAActive) anyAActive = true;
//                if (p.componentBActive) anyBActive = true;
//                if (p.componentCActive) anyCActive = true;
//                if (p.driftingFeaturesObserved) anyDriftingObserved = true;
//
//                w.printf(Locale.ROOT,
//                        "%-14s %-22s seed=%d  acc=%.4f k=%.4f kPer=%.4f  drift=%d kept=%d surg=%d full=%d noR=%d  wPred=%d unwFb=%d  stab=%.3f  ramH=%.6f  %s %s%n",
//                        r.dataset, r.variant, r.seed,
//                        r.accuracy, r.kappa, r.kappaPer,
//                        r.driftCount, r.totalKept, r.totalSurgical, r.totalFull, r.totalNoReplacement,
//                        r.weightedPredictions, r.unweightedFallbacks,
//                        r.featureStability, r.ramHours,
//                        p.tag, p.reason.toString());
//            }
//
//            w.println();
//            w.println("--- GLOBAL CHECKS ---");
//            w.println("any DA-SRP variant has Component A active : " + anyAActive);
//            w.println("any AB variant did surgical updates        : " + anyBActive);
//            w.println("ABC variant produced weighted predictions  : " + anyCActive);
//            w.println("any run observed drift alarms              : " + anyDriftingObserved);
//            w.printf("--- TOTAL: %d PASS / %d FAIL / %d WARN ---%n", pass, fail, warn);
//
//            System.out.println("------------------------------------------------------------");
//            System.out.printf("[proto] TOTAL: %d PASS / %d FAIL / %d WARN%n", pass, fail, warn);
//            System.out.println("[proto] anyA=" + anyAActive
//                    + " anyB=" + anyBActive
//                    + " anyC=" + anyCActive
//                    + " anyDrift=" + anyDriftingObserved);
//            System.out.println("[proto] details -> " + cfg.outputDir + "/E3_prototype_summary.txt");
//        }
//    }
//}
