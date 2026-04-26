package thesis.models;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.streams.generators.SEAGenerator;
import thesis.discretization.PiDDiscretizer;
import thesis.selection.FeatureSelector;
import thesis.selection.InformationGainRanker;
import thesis.selection.StaticFeatureSelector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class DriftAwareSRPSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("DRIFT-AWARE SRP SMOKE TESTS");
        System.out.println("=".repeat(70));

        diagnoseSubspaceFieldName();

        testFeatureImportanceUniformAtStart();
        testFeatureImportanceWeightsHighMI();
        testFeatureImportanceStabilityFavorsLowKS();
        testFeatureImportanceProjectToReduced();
        testFeatureImportanceRejectsBadInputs();

        testWeightedSamplerRespectsExclude();
        testWeightedSamplerFavorsHighWeights();
        testWeightedSamplerFallsBackWhenAllExcluded();
        testWeightedSamplerRejectsBadParams();

        testDriftAwareSRPConstructorValidation();
        testDriftAwareSRPDelegatesPredictWhenNoImportance();
        testDriftAwareSRPWeightedPredictionMatchesAccuracy();
        testDriftAwareSRPHandleDriftAllKeepWhenNoOverlap();
        testDriftAwareSRPHandleDriftSurgicalWhenLowOverlap();
        testDriftAwareSRPHandleDriftFullWhenHighOverlap();
        testDriftAwareSRPRefreshAllSubspaces();
        testDriftAwareSRPHandleDriftRejectsNullScores();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    private static void diagnoseSubspaceFieldName() {
        int F = 3, K = 2;
        int[] y = new int[1500];
        double[][] data = sea(1500, y, 99);
        InstancesHeader h = makeHeader(F, 2, F);
        SRPWrapper srp = buildTrainedSRP(F, K, 1200, 99, data, y, h);

        Object[] ensemble = null;
        for (String name : new String[]{"ensemble", "learners", "baseLearners", "classifiers"}) {
            try {
                java.lang.reflect.Field f = findField(srp.getSRP().getClass(), name);
                if (f == null) continue;
                f.setAccessible(true);
                Object v = f.get(srp.getSRP());
                if (v != null && v.getClass().isArray()) {
                    int len = java.lang.reflect.Array.getLength(v);
                    ensemble = new Object[len];
                    for (int i = 0; i < len; i++) ensemble[i] = java.lang.reflect.Array.get(v, i);
                    break;
                }
            } catch (Exception ignored) {}
        }
        if (ensemble == null || ensemble.length == 0) {
            System.out.println("  [DIAG] no ensemble visible — SRP introspection failed entirely");
            return;
        }
        Object learner = ensemble[0];
        System.out.println("  [DIAG] learner class: " + learner.getClass().getName());
        Class<?> c = learner.getClass();
        while (c != null && c != Object.class) {
            for (java.lang.reflect.Field f : c.getDeclaredFields()) {
                String tn = f.getType().getSimpleName();
                if (tn.equals("int[]") || tn.equals("Integer[]")
                        || f.getName().toLowerCase().contains("sub")
                        || f.getName().toLowerCase().contains("ind")) {
                    System.out.println("    " + c.getSimpleName() + "." + f.getName()
                            + " : " + f.getType().getName());
                }
            }
            c = c.getSuperclass();
        }
    }

    private static java.lang.reflect.Field findField(Class<?> cls, String name) {
        Class<?> c = cls;
        while (c != null && c != Object.class) {
            try { return c.getDeclaredField(name); }
            catch (NoSuchFieldException e) { c = c.getSuperclass(); }
        }
        return null;
    }

    private static InstancesHeader makeHeader(int F, int numClasses, int classPos) {
        ArrayList<Attribute> attrs = new ArrayList<>(F + 1);
        ArrayList<String> classVals = new ArrayList<>();
        for (int c = 0; c < numClasses; c++) classVals.add("c" + c);
        for (int i = 0; i < F + 1; i++) {
            if (i == classPos) attrs.add(new Attribute("class", classVals));
            else                attrs.add(new Attribute("f" + i));
        }
        Instances ins = new Instances("synthetic", attrs, 0);
        ins.setClassIndex(classPos);
        return new InstancesHeader(ins);
    }

    private static Instance makeInstance(InstancesHeader header, double[] values) {
        Instance inst = new DenseInstance(1.0, values);
        inst.setDataset(header);
        return inst;
    }

    private static double[][] sea(int n, int[] outLabels, long seed) {
        SEAGenerator g = new SEAGenerator();
        g.instanceRandomSeedOption.setValue((int) seed);
        g.functionOption.setValue(1);
        g.balanceClassesOption.setValue(false);
        g.noisePercentageOption.setValue(10);
        g.prepareForUse();
        double[][] out = new double[n][3];
        for (int i = 0; i < n; i++) {
            Instance inst = g.nextInstance().getData();
            for (int f = 0; f < 3; f++) out[i][f] = inst.value(f);
            outLabels[i] = (int) inst.classValue();
        }
        return out;
    }

    private static FeatureSelector trainedSelector(int F, double[][] win, int[] y, int K) {
        StaticFeatureSelector sel = new StaticFeatureSelector(F, 2, K,
                new PiDDiscretizer(F, 2),
                (bins, classes) -> new InformationGainRanker(F, bins, classes));
        sel.initialize(win, y);
        return sel;
    }

    private static SRPWrapper buildTrainedSRP(int F, int K, int trainN, long seed,
                                              double[][] data, int[] y, InstancesHeader h) {
        FeatureSelector sel = trainedSelector(F, Arrays.copyOf(data, 600),
                Arrays.copyOf(y, 600), K);
        SRPWrapper srp = new SRPWrapper(sel, h, 5, 6.0, false);
        for (int i = 0; i < trainN; i++) {
            double[] vals = new double[h.numAttributes()];
            for (int f = 0; f < F; f++) vals[f] = data[i][f];
            vals[h.classIndex()] = y[i];
            srp.train(makeInstance(h, vals), y[i]);
        }
        return srp;
    }

    private static void testFeatureImportanceUniformAtStart() {
        FeatureImportance fi = new FeatureImportance(5);
        double[] imp = fi.getImportance();
        boolean ok = imp.length == 5;
        for (double v : imp) ok &= Math.abs(v - 0.2) < 1e-9;
        report("FeatureImportance uniform at construction", ok && fi.getUpdates() == 0);
    }

    private static void testFeatureImportanceWeightsHighMI() {
        FeatureImportance fi = new FeatureImportance(4, 1.0, 0.0, 1e-6, true);
        double[] mi = {0.1, 0.9, 0.05, 0.5};
        double[] ks = {0.0, 0.0, 0.0, 0.0};
        fi.update(mi, ks);
        double[] imp = fi.getImportance();
        double sum = 0; for (double v : imp) sum += v;
        boolean ok = Math.abs(sum - 1.0) < 1e-9
                && imp[1] > imp[3] && imp[3] > imp[0] && imp[0] > imp[2];
        report("FeatureImportance ranks features by MI when w2=0 (imp="
                + Arrays.toString(imp) + ")", ok);
    }

    private static void testFeatureImportanceStabilityFavorsLowKS() {
        FeatureImportance fi = new FeatureImportance(3, 0.0, 1.0, 1e-6, true);
        double[] mi = {0.5, 0.5, 0.5};
        double[] ks = {0.9, 0.1, 0.5};
        fi.update(mi, ks);
        double[] imp = fi.getImportance();
        boolean ok = imp[1] > imp[2] && imp[2] > imp[0];
        report("FeatureImportance stability term favors low KS (imp="
                + Arrays.toString(imp) + ")", ok);
    }

    private static void testFeatureImportanceProjectToReduced() {
        FeatureImportance fi = new FeatureImportance(5, 1.0, 0.0, 1e-6, false);
        fi.update(new double[]{0.1, 0.2, 0.3, 0.4, 0.5}, new double[5]);
        double[] proj = fi.projectToReduced(new int[]{4, 1, 99});
        double[] full = fi.getImportance();
        boolean ok = proj.length == 3
                && proj[0] == full[4]
                && proj[1] == full[1]
                && proj[2] == 0.0;
        report("FeatureImportance.projectToReduced maps and zero-pads", ok);
    }

    private static void testFeatureImportanceRejectsBadInputs() {
        boolean t1 = false, t2 = false, t3 = false, t4 = false, t5 = false;
        try { new FeatureImportance(0); } catch (IllegalArgumentException e) { t1 = true; }
        try { new FeatureImportance(3, -1, 1, 1e-6, true); } catch (IllegalArgumentException e) { t2 = true; }
        try { new FeatureImportance(3, 0, 0, 1e-6, true); } catch (IllegalArgumentException e) { t3 = true; }
        try { new FeatureImportance(3, 1, 1, 0, true); } catch (IllegalArgumentException e) { t4 = true; }
        try { new FeatureImportance(3).update(new double[2], new double[3]); }
        catch (IllegalArgumentException e) { t5 = true; }
        report("FeatureImportance rejects bad inputs", t1 && t2 && t3 && t4 && t5);
    }

    private static void testWeightedSamplerRespectsExclude() {
        Random rng = new Random(1);
        double[] w = {1, 1, 1, 1, 1, 1};
        Set<Integer> excl = new HashSet<>(Arrays.asList(0, 1, 2));
        int[] s = WeightedSubspaceSampler.sample(w, 3, rng, excl);
        boolean ok = s.length == 3;
        for (int i : s) ok &= !excl.contains(i);
        report("WeightedSubspaceSampler respects exclude set (got=" + Arrays.toString(s) + ")", ok);
    }

    private static void testWeightedSamplerFavorsHighWeights() {
        Random rng = new Random(2);
        double[] w = {0.001, 0.001, 0.001, 1.0, 1.0, 0.001};
        int hits3 = 0, hits4 = 0;
        int trials = 500;
        for (int t = 0; t < trials; t++) {
            int[] s = WeightedSubspaceSampler.sample(w, 2, new Random(t), null);
            for (int idx : s) {
                if (idx == 3) hits3++;
                if (idx == 4) hits4++;
            }
        }
        boolean ok = hits3 > trials / 2 && hits4 > trials / 2;
        report("WeightedSubspaceSampler favors high-weight indices ("
                + hits3 + "/" + trials + ", " + hits4 + "/" + trials + ")", ok);
    }

    private static void testWeightedSamplerFallsBackWhenAllExcluded() {
        Random rng = new Random(3);
        double[] w = {1, 1, 1, 1};
        Set<Integer> excl = new HashSet<>(Arrays.asList(0, 1, 2, 3));
        int[] s = WeightedSubspaceSampler.sample(w, 2, rng, excl);
        boolean ok = s.length == 0 || s.length == 2;
        report("WeightedSubspaceSampler degrades gracefully when all excluded "
                + "(got len=" + s.length + ")", ok);
    }

    private static void testWeightedSamplerRejectsBadParams() {
        boolean t1 = false, t2 = false;
        try { WeightedSubspaceSampler.sample(new double[]{1, 1}, 0, new Random(), null); }
        catch (IllegalArgumentException e) { t1 = true; }
        try { WeightedSubspaceSampler.sample(new double[]{1, 1}, 5, new Random(), null); }
        catch (IllegalArgumentException e) { t2 = true; }
        report("WeightedSubspaceSampler rejects bad size", t1 && t2);
    }

    private static void testDriftAwareSRPConstructorValidation() {
        boolean t1 = false, t2 = false, t3 = false;
        try { new DriftAwareSRP(null); } catch (IllegalArgumentException e) { t1 = true; }
        int F = 3;
        int[] y = new int[600];
        double[][] data = sea(600, y, 41);
        InstancesHeader h = makeHeader(F, 2, F);
        SRPWrapper srp = new SRPWrapper(
                trainedSelector(F, data, y, 2), h, 5, 6.0, false);
        try { new DriftAwareSRP(srp, 0.0, 7L); } catch (IllegalArgumentException e) { t2 = true; }
        try { new DriftAwareSRP(srp, 1.5, 7L); } catch (IllegalArgumentException e) { t3 = true; }
        report("DriftAwareSRP constructor validation", t1 && t2 && t3);
    }

    private static void testDriftAwareSRPDelegatesPredictWhenNoImportance() {
        int F = 3, N = 1500, K = 2;
        int[] y = new int[N];
        double[][] data = sea(N, y, 42);
        InstancesHeader h = makeHeader(F, 2, F);
        SRPWrapper srp = buildTrainedSRP(F, K, 1000, 42, data, y, h);
        DriftAwareSRP m = new DriftAwareSRP(srp, 0.5, 7L, null);

        double[] vals = new double[h.numAttributes()];
        for (int f = 0; f < F; f++) vals[f] = data[1000][f];
        vals[h.classIndex()] = y[1000];
        Instance inst = makeInstance(h, vals);

        double[] direct = srp.predictProba(inst);
        double[] viaWrap = m.predictProba(inst);
        boolean ok = Arrays.equals(direct, viaWrap)
                && m.getUnweightedFallbacks() >= 1
                && m.getWeightedPredictions() == 0;
        report("DriftAwareSRP delegates to SRP when importance is null", ok);
    }

    private static void testDriftAwareSRPWeightedPredictionMatchesAccuracy() {
        int F = 3, N = 2000, K = 2;
        int[] y = new int[N];
        double[][] data = sea(N, y, 43);
        InstancesHeader h = makeHeader(F, 2, F);
        SRPWrapper srp = buildTrainedSRP(F, K, 1500, 43, data, y, h);
        FeatureImportance imp = new FeatureImportance(F);
        imp.update(new double[]{0.8, 0.2, 0.1}, new double[]{0.0, 0.5, 0.5});
        DriftAwareSRP m = new DriftAwareSRP(srp, 0.5, 7L, imp);

        int correct = 0;
        for (int i = 1500; i < 2000; i++) {
            double[] vals = new double[h.numAttributes()];
            for (int f = 0; f < F; f++) vals[f] = data[i][f];
            vals[h.classIndex()] = y[i];
            if (m.predict(makeInstance(h, vals)) == y[i]) correct++;
        }
        boolean ok = correct >= 250
                && m.getWeightedPredictions() >= 1
                && m.getLastLearnerWeights().length >= 1;
        report("DriftAwareSRP weighted prediction works (acc=" + correct + "/500, "
                + "weightedCalls=" + m.getWeightedPredictions() + ")", ok);
    }

    private static void testDriftAwareSRPHandleDriftAllKeepWhenNoOverlap() {
        int F = 3, K = 2;
        int[] y = new int[1500];
        double[][] data = sea(1500, y, 44);
        InstancesHeader h = makeHeader(F, 2, F);
        SRPWrapper srp = buildTrainedSRP(F, K, 1200, 44, data, y, h);
        DriftAwareSRP m = new DriftAwareSRP(srp, 0.5, 7L);

        Set<Integer> driftingOriginal = new HashSet<>();
        for (int f = 0; f < F; f++) {
            boolean inSelection = false;
            for (int s : srp.getCurrentSelection()) if (s == f) { inSelection = true; break; }
            if (!inSelection) driftingOriginal.add(f);
        }
        double[] scores = new double[F];
        Arrays.fill(scores, 1.0);

        DriftActionSummary summary = m.handleDrift(driftingOriginal, scores);
        boolean ok = summary.getEnsembleSize() >= 1
                && summary.getKeptCount() == summary.getEnsembleSize()
                && summary.getSurgicalCount() == 0
                && summary.getFullCount() == 0;
        report("DriftAwareSRP handleDrift KEEP when no overlap (kept=" + summary.getKeptCount()
                + "/" + summary.getEnsembleSize() + ")", ok);
    }

    private static void testDriftAwareSRPHandleDriftSurgicalWhenLowOverlap() {
        int F = 5, K = 4;
        int N = 2000;
        int[] y = new int[N];
        double[][] data = sea(N, y, 45);
        for (int i = 0; i < N; i++) {
            double[] r = data[i];
            double[] grown = new double[F];
            grown[0] = r[0]; grown[1] = r[1]; grown[2] = r[2];
            grown[3] = new Random(45 + i).nextGaussian();
            grown[4] = new Random(450 + i).nextGaussian();
            data[i] = grown;
        }
        InstancesHeader h = makeHeader(F, 2, F);

        double[][] win = Arrays.copyOf(data, 600);
        int[] yWin = Arrays.copyOf(y, 600);
        FeatureSelector sel = trainedSelector(F, win, yWin, K);
        SRPWrapper srp = new SRPWrapper(sel, h, 6, 6.0, false);
        for (int i = 0; i < 1500; i++) {
            double[] vals = new double[h.numAttributes()];
            for (int f = 0; f < F; f++) vals[f] = data[i][f];
            vals[h.classIndex()] = y[i];
            srp.train(makeInstance(h, vals), y[i]);
        }
        DriftAwareSRP m = new DriftAwareSRP(srp, 0.9, 7L);
        int[] selection = srp.getCurrentSelection();
        Set<Integer> driftingOriginal = new HashSet<>();
        if (selection.length > 0) driftingOriginal.add(selection[0]);

        double[] scores = new double[F];
        Arrays.fill(scores, 1.0);
        for (int s : selection) scores[s] = 5.0;
        if (selection.length > 0) scores[selection[0]] = 0.0;

        DriftActionSummary summary = m.handleDrift(driftingOriginal, scores);
        int touched = summary.getSurgicalCount() + summary.getFullCount()
                + summary.getNoReplacementCount();
        boolean ok = summary.getEnsembleSize() >= 1
                && touched >= 1
                && summary.getFullCount() == 0;
        report("DriftAwareSRP handleDrift SURGICAL on low overlap (summary=" + summary + ")", ok);
    }

    private static void testDriftAwareSRPHandleDriftFullWhenHighOverlap() {
        int F = 6, K = 5;
        int[] y = new int[1500];
        double[][] data = sea(1500, y, 46);
        for (int i = 0; i < 1500; i++) {
            double[] grown = new double[F];
            grown[0] = data[i][0]; grown[1] = data[i][1]; grown[2] = data[i][2];
            grown[3] = new Random(46 + i).nextGaussian();
            grown[4] = new Random(460 + i).nextGaussian();
            grown[5] = new Random(4600 + i).nextGaussian();
            data[i] = grown;
        }
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = trainedSelector(F,
                Arrays.copyOf(data, 600), Arrays.copyOf(y, 600), K);
        SRPWrapper srp = new SRPWrapper(sel, h, 5, 6.0, false);
        for (int i = 0; i < 1200; i++) {
            double[] vals = new double[h.numAttributes()];
            for (int f = 0; f < F; f++) vals[f] = data[i][f];
            vals[h.classIndex()] = y[i];
            srp.train(makeInstance(h, vals), y[i]);
        }
        DriftAwareSRP m = new DriftAwareSRP(srp, 0.5, 7L);

        Set<Integer> driftingOriginal = new HashSet<>();
        for (int s : srp.getCurrentSelection()) driftingOriginal.add(s);

        double[] scores = new double[F];
        Arrays.fill(scores, 1.0);
        DriftActionSummary summary = m.handleDrift(driftingOriginal, scores);
        int touchedDestructively = summary.getFullCount();
        boolean ok = summary.getEnsembleSize() >= 1
                && touchedDestructively >= 1
                && summary.getKeptCount() < summary.getEnsembleSize();
        report("DriftAwareSRP handleDrift FULL when whole selection drifts "
                + "(full=" + summary.getFullCount() + "/" + summary.getEnsembleSize()
                + ", surgical=" + summary.getSurgicalCount() + ")", ok);
    }

    private static void testDriftAwareSRPRefreshAllSubspaces() {
        int F = 6, K = 4;
        int[] y = new int[1500];
        double[][] data = sea(1500, y, 47);
        for (int i = 0; i < 1500; i++) {
            double[] grown = new double[F];
            grown[0] = data[i][0]; grown[1] = data[i][1]; grown[2] = data[i][2];
            grown[3] = new Random(47 + i).nextGaussian();
            grown[4] = new Random(470 + i).nextGaussian();
            grown[5] = new Random(4700 + i).nextGaussian();
            data[i] = grown;
        }
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = trainedSelector(F,
                Arrays.copyOf(data, 600), Arrays.copyOf(y, 600), K);
        SRPWrapper srp = new SRPWrapper(sel, h, 5, 6.0, false);
        for (int i = 0; i < 1200; i++) {
            double[] vals = new double[h.numAttributes()];
            for (int f = 0; f < F; f++) vals[f] = data[i][f];
            vals[h.classIndex()] = y[i];
            srp.train(makeInstance(h, vals), y[i]);
        }
        DriftAwareSRP m = new DriftAwareSRP(srp, 0.5, 7L);

        DriftAwareSRP.RefreshSummary rs = m.refreshAllSubspaces();
        boolean ok = rs.ensembleSize >= 1
                && rs.refreshedCount >= 1
                && m.getRefreshCalls() == 1
                && m.getTotalRefreshed() == rs.refreshedCount;
        report("DriftAwareSRP.refreshAllSubspaces refreshes ensemble (refreshed="
                + rs.refreshedCount + "/" + rs.ensembleSize + ")", ok);
    }

    private static void testDriftAwareSRPHandleDriftRejectsNullScores() {
        int F = 3, K = 2;
        int[] y = new int[600];
        double[][] data = sea(600, y, 48);
        InstancesHeader h = makeHeader(F, 2, F);
        SRPWrapper srp = buildTrainedSRP(F, K, 500, 48, data, y, h);
        DriftAwareSRP m = new DriftAwareSRP(srp, 0.5, 7L);
        boolean threw = false;
        try { m.handleDrift(Set.of(), null); }
        catch (IllegalArgumentException e) { threw = true; }
        report("DriftAwareSRP.handleDrift rejects null featureScores", threw);
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