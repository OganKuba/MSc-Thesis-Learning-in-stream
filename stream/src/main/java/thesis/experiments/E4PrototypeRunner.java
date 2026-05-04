//package thesis.experiments;
//
//import thesis.experiments.E4DriftAnalysis.AlarmEvent;
//import thesis.experiments.E4DriftAnalysis.Cfg;
//import thesis.experiments.E4DriftAnalysis.GeneratorSpec;
//import thesis.experiments.E4DriftAnalysis.Magnitude;
//import thesis.experiments.E4DriftAnalysis.RunResult;
//
//import java.io.FileWriter;
//import java.io.PrintWriter;
//import java.nio.file.Files;
//import java.nio.file.Path;
//import java.nio.file.Paths;
//import java.util.ArrayList;
//import java.util.List;
//import java.util.Locale;
//
//public class E4PrototypeRunner {
//
//    static final class Probe {
//        RunResult r;
//        boolean detectionWorks;
//        boolean kappaSane;
//        boolean alarmsObserved;
//        boolean noFeatureDetectionForCFD;
//        String tag = "PASS";
//        StringBuilder reason = new StringBuilder();
//    }
//
//    public static void main(String[] args) throws Exception {
//
//        Path cfgPath = args.length > 0
//                ? Paths.get(args[0])
//                : findDefaultConfig();
//
//        if (!Files.exists(cfgPath)) {
//            throw new RuntimeException(
//                    "Config not found: " + cfgPath.toAbsolutePath() +
//                            "\nWorking dir: " + Paths.get(".").toAbsolutePath()
//            );
//        }
//
//        Cfg cfg = Cfg.load(cfgPath);
//        new E4PrototypeRunner().run(cfg);
//    }
//
//    private static Path findDefaultConfig() {
//        List<Path> candidates = List.of(
//                Paths.get("src/main/java/thesis/experiments/e4_prototype.json"),
//
//                Paths.get("experiments/e4_prototype.json"),
//
//                Paths.get("src/main/resources/e4_prototype.json")
//        );
//
//        for (Path p : candidates) {
//            if (Files.exists(p)) {
//                return p;
//            }
//        }
//
//        return candidates.get(0);
//    }
//
//    public void run(Cfg cfg) throws Exception {
//        Files.createDirectories(Paths.get(cfg.outputDir));
//        Path winCsv = Paths.get(cfg.outputDir, "E4_prototype.csv");
//        Path alrCsv = Paths.get(cfg.outputDir, "E4_prototype_alarms.csv");
//        List<RunResult> all = new ArrayList<>();
//        E4DriftAnalysis runner = new E4DriftAnalysis();
//
//        try (PrintWriter wcsv = new PrintWriter(new FileWriter(winCsv.toFile()));
//             PrintWriter acsv = new PrintWriter(new FileWriter(alrCsv.toFile()))) {
//            wcsv.println(E4DriftAnalysis.windowHeader());
//            acsv.println(E4DriftAnalysis.alarmHeader());
//            for (GeneratorSpec g : cfg.generators) {
//                for (String mag : cfg.magnitudes) {
//                    Magnitude m = Magnitude.valueOf(mag);
//                    for (int seed : cfg.seeds) {
//                        for (String method : cfg.methods) {
//                            try {
//                                RunResult r = runner.runOne(cfg, g, m, method, seed, wcsv);
//                                E4DriftAnalysis.writeAlarmsCsv(acsv, r);
//                                all.add(r);
//                            } catch (Exception e) {
//                                System.err.printf("[proto][FAIL] %s|%s|%s|seed=%d -> %s%n",
//                                        g.name, method, mag, seed, e);
//                                e.printStackTrace(System.err);
//                            }
//                        }
//                    }
//                }
//            }
//        }
//        validateAndSummarize(cfg, all);
//    }
//
//    private void validateAndSummarize(Cfg cfg, List<RunResult> all) throws Exception {
//        Path summaryPath = Paths.get(cfg.outputDir, "E4_prototype_summary.txt");
//        try (PrintWriter w = new PrintWriter(new FileWriter(summaryPath.toFile()))) {
//            w.println("=== E4 Prototype Summary ===");
//            int pass = 0, fail = 0, warn = 0;
//            boolean anyAlarms = false, anyDetectionWorks = false, anyFeatureDetection = false;
//
//            for (RunResult r : all) {
//                Probe p = new Probe();
//                p.r = r;
//                p.alarmsObserved = !r.alarms.isEmpty();
//                p.detectionWorks = r.detection.tp > 0 || (!r.abrupt && r.detection.f1 > 0);
//                p.kappaSane = r.kappa > -0.5 && r.kappa < 1.0001 && !Double.isNaN(r.kappa);
//                p.noFeatureDetectionForCFD = "CustomFeatureDrift".equalsIgnoreCase(r.generator)
//                        && r.featureDetection.applicable && r.featureDetection.tp == 0
//                        && r.detection.tp > 0;
//
//                if (!p.kappaSane)         p.reason.append("kappa NaN/out-of-range; ");
//                if (!p.alarmsObserved)    p.reason.append("no alarms observed; ");
//                if (r.abrupt && r.detection.tp == 0 && r.gtPositions.length > 0)
//                    p.reason.append("zero TP on abrupt generator; ");
//                if (p.noFeatureDetectionForCFD)
//                    p.reason.append("CFD with TP but feature_TP=0 (KSWIN missed all); ");
//
//                if (!p.kappaSane) { p.tag = "FAIL"; fail++; }
//                else if (r.abrupt && r.detection.tp == 0 && r.gtPositions.length > 0) {
//                    p.tag = "WARN"; warn++;
//                } else if (!p.alarmsObserved && r.abrupt) {
//                    p.tag = "WARN"; warn++;
//                } else { p.tag = "PASS"; pass++; }
//
//                if (p.alarmsObserved)    anyAlarms = true;
//                if (p.detectionWorks)    anyDetectionWorks = true;
//                if (r.featureDetection.applicable && r.featureDetection.tp > 0) anyFeatureDetection = true;
//
//                w.printf(Locale.ROOT,
//                        "%-18s %-12s %-6s seed=%d  k=%.4f  drift=%d  TP=%d FP=%d FN=%d  P=%.3f R=%.3f F1=%.3f delay=%.1f  featF1=%.3f  %s %s%n",
//                        r.generator, r.method, r.magnitude, r.seed, r.kappa,
//                        r.driftCount, r.detection.tp, r.detection.fp, r.detection.fn,
//                        r.detection.precision, r.detection.recall, r.detection.f1,
//                        r.detection.meanDetectionDelay,
//                        r.featureDetection.applicable ? r.featureDetection.f1 : Double.NaN,
//                        p.tag, p.reason.toString());
//            }
//
//            w.println();
//            w.println("--- GLOBAL CHECKS ---");
//            w.println("any run produced alarms              : " + anyAlarms);
//            w.println("any run had non-zero TP detection    : " + anyDetectionWorks);
//            w.println("any CFD run had feature TP > 0       : " + anyFeatureDetection);
//            w.printf("--- TOTAL: %d PASS / %d FAIL / %d WARN ---%n", pass, fail, warn);
//
//            System.out.println("------------------------------------------------------------");
//            System.out.printf("[proto] TOTAL: %d PASS / %d FAIL / %d WARN%n", pass, fail, warn);
//            System.out.println("[proto] anyAlarms=" + anyAlarms
//                    + " anyTP=" + anyDetectionWorks
//                    + " anyFeatTP=" + anyFeatureDetection);
//            System.out.println("[proto] details -> " + summaryPath);
//        }
//
//        E4DriftAnalysis.writeSummary(all, Paths.get(cfg.outputDir, "E4_prototype_summary.csv"));
//        E4DriftAnalysis.writeAggregated(cfg, all);
//        E4DriftAnalysis.writeMinDetectableMagnitude(cfg, all);
//        E4DriftAnalysis.writeRanking(cfg, all);
//    }
//}
