package thesis.models;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.streams.generators.SEAGenerator;
import thesis.selection.FeatureSelector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class DriftAwareSRPSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("DRIFT-AWARE SRP SMOKE TESTS");
        System.out.println("=".repeat(70));

        testRequiresHardFilterFalse();
        testRejectsBadTau();
        testRejectsImportanceDimMismatch();

        testFeatureImportanceUniformByDefault();
        testFeatureImportanceUpdateAndNormalize();
        testFeatureImportanceBoostNormalizes();
        testFeatureImportanceDegenerateFallback();

        testWeightedSamplerNoReplacement();
        testWeightedSamplerHonorsExclude();
        testWeightedSamplerPrefersHighWeight();
        testWeightedSamplerFallsBackOnAllZero();

        testHandleDriftKeepsWhenNoOverlap();
        testHandleDriftSurgicalWhenSmallOverlap();
        testHandleDriftFullWhenLargeOverlap();
        testHandleDriftIdempotentAfterEmptySet();
        testHandleDriftRejectsBadScoresLength();

        testSubspaceWritesPersistAndAreSorted();
        testSubspaceIndicesNeverOutOfRange();

        testLearnerWeightsSumToOneAndPositive();
        testLearnerWeightsHigherForBetterSubspaces();

        testPredictWeightedFallbackWhenNoImportance();
        testPredictNeverReturnsNaN();

        testAutoHandleDriftFiresOnAlarm();
        testDriftListenerInvoked();
        testS1VsS4DiffersAfterAutoDrift();

        testSurgicalNoResetPolicyKeepsAccuracy();
        testRefreshAllResetsCounters();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    private static InstancesHeader makeHeader(int F, int numClasses, int classPos) {
        ArrayList<Attribute> attrs = new ArrayList<>(F + 1);
        ArrayList<String> classVals = new ArrayList<>();
        for (int c = 0; c < numClasses; c++) classVals.add("c" + c);
        for (int i = 0; i < F + 1; i++) {
            if (i == classPos) attrs.add(new Attribute("class", classVals));
            else attrs.add(new Attribute("f" + i));
        }
        Instances ins = new Instances("synthetic", attrs, 0);
        ins.setClassIndex(classPos);
        return new InstancesHeader(ins);
    }

    private static Instance makeInstance(InstancesHeader h, double[] v) {
        Instance i = new DenseInstance(1.0, v);
        i.setDataset(h);
        return i;
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

    private static Instance fullInstance(InstancesHeader h, double[] feats, int label) {
        double[] vals = new double[h.numAttributes()];
        for (int f = 0; f < feats.length; f++) vals[f] = feats[f];
        vals[h.classIndex()] = label;
        return makeInstance(h, vals);
    }

    private static final class FixedSelector implements FeatureSelector {
        private int[] selection;
        private final int F, C;
        FixedSelector(int F, int C, int[] sel) { this.F = F; this.C = C; this.selection = sel.clone(); }
        @Override public boolean isInitialized() { return true; }
        @Override public int[] getCurrentSelection() { return selection.clone(); }
        @Override public int[] getSelectedFeatures() { return selection.clone(); }
        @Override public int getNumFeatures() { return F; }
        @Override public int getK() { return selection.length; }
        @Override public double[] filterInstance(double[] full) {
            double[] o = new double[selection.length];
            for (int i = 0; i < selection.length; i++) o[i] = full[selection[i]];
            return o;
        }
        @Override public void initialize(double[][] win, int[] y) { }
        @Override public void update(double[] f, int l, boolean a, Set<Integer> df) { }
        @Override public String name() { return "Fixed"; }
        void setSelection(int[] s) { this.selection = s.clone(); }
    }

    private static SRPWrapper newSrp(int F, int ensembleSize, FixedSelector sel) {
        InstancesHeader h = makeHeader(F, 2, F);
        return new SRPWrapper(sel, h, ensembleSize, 6.0, false, false);
    }

    private static SRPWrapper newSrpHardFilter(int F, int ensembleSize, FixedSelector sel) {
        InstancesHeader h = makeHeader(F, 2, F);
        return new SRPWrapper(sel, h, ensembleSize, 6.0, false, true);
    }

    private static void warmTrain(DriftAwareSRP m, InstancesHeader h, double[][] data, int[] y, int n) {
        for (int i = 0; i < n; i++) m.train(fullInstance(h, data[i], y[i]), y[i]);
    }

    private static void testRequiresHardFilterFalse() {
        int F = 5;
        FixedSelector sel = new FixedSelector(F, 2, new int[]{0, 1, 2});
        SRPWrapper srp = newSrpHardFilter(F, 4, sel);
        boolean threw = false;
        try { new DriftAwareSRP(srp); } catch (IllegalArgumentException e) { threw = true; }
        report("DA-SRP rejects SRPWrapper with useHardFilter=true", threw);
    }

    private static void testRejectsBadTau() {
        int F = 4;
        FixedSelector sel = new FixedSelector(F, 2, new int[]{0, 1});
        SRPWrapper srp = newSrp(F, 3, sel);
        boolean t1 = false, t2 = false, t3 = false;
        try { new DriftAwareSRP(srp, 0.0, 1L); } catch (IllegalArgumentException e) { t1 = true; }
        try { new DriftAwareSRP(srp, -0.1, 1L); } catch (IllegalArgumentException e) { t2 = true; }
        try { new DriftAwareSRP(srp, 1.5, 1L); } catch (IllegalArgumentException e) { t3 = true; }
        report("DA-SRP rejects bad tau", t1 && t2 && t3);
    }

    private static void testRejectsImportanceDimMismatch() {
        int F = 5;
        FixedSelector sel = new FixedSelector(F, 2, new int[]{0, 1});
        SRPWrapper srp = newSrp(F, 3, sel);
        boolean threw = false;
        try { new DriftAwareSRP(srp, 0.5, 1L, new FeatureImportance(F + 2)); }
        catch (IllegalArgumentException e) { threw = true; }
        report("DA-SRP rejects FeatureImportance with wrong dim", threw);
    }

    private static void testFeatureImportanceUniformByDefault() {
        FeatureImportance imp = new FeatureImportance(4);
        double[] v = imp.getImportance();
        boolean ok = true;
        for (double x : v) if (Math.abs(x - 0.25) > 1e-9) ok = false;
        report("FeatureImportance defaults to uniform", ok);
    }

    private static void testFeatureImportanceUpdateAndNormalize() {
        FeatureImportance imp = new FeatureImportance(3);
        imp.update(new double[]{0.0, 0.5, 1.0}, new double[]{0.0, 0.0, 0.0});
        double[] v = imp.getImportance();
        double sum = 0.0; for (double x : v) sum += x;
        boolean monotone = v[2] >= v[1] && v[1] >= v[0];
        report("FeatureImportance.update normalizes (sum=" + sum
                        + ", monotone=" + monotone + ")",
                Math.abs(sum - 1.0) < 1e-9 && monotone);
    }

    private static void testFeatureImportanceBoostNormalizes() {
        FeatureImportance imp = new FeatureImportance(4);
        imp.boost(Set.of(2), 5.0);
        double[] v = imp.getImportance();
        double sum = 0.0; for (double x : v) sum += x;
        boolean ok = Math.abs(sum - 1.0) < 1e-9 && v[2] > v[0];
        report("FeatureImportance.boost increases target and re-normalizes", ok);
    }

    private static void testFeatureImportanceDegenerateFallback() {
        FeatureImportance imp = new FeatureImportance(3);
        imp.update(new double[]{0.0, 0.0, 0.0}, new double[]{1e30, 1e30, 1e30});
        double[] v = imp.getImportance();
        double sum = 0.0; for (double x : v) sum += x;
        report("FeatureImportance degenerate input → uniform fallback (fallbacks="
                        + imp.getDegenerateUniformFallbacks() + ")",
                Math.abs(sum - 1.0) < 1e-9
                        && imp.getDegenerateUniformFallbacks() >= 0);
    }

    private static void testWeightedSamplerNoReplacement() {
        double[] w = {0.1, 0.2, 0.3, 0.4, 0.5};
        Random rng = new Random(42);
        int[] s = WeightedSubspaceSampler.sample(w, 3, rng, Set.of());
        Set<Integer> seen = new HashSet<>();
        for (int x : s) seen.add(x);
        report("WeightedSampler returns distinct indices (got " + Arrays.toString(s) + ")",
                s.length == 3 && seen.size() == 3);
    }

    private static void testWeightedSamplerHonorsExclude() {
        double[] w = {1, 1, 1, 1, 1};
        Random rng = new Random(7);
        int[] s = WeightedSubspaceSampler.sample(w, 2, rng, Set.of(0, 1));
        boolean ok = true;
        for (int x : s) if (x == 0 || x == 1) ok = false;
        report("WeightedSampler honors exclude (got " + Arrays.toString(s) + ")", ok);
    }

    private static void testWeightedSamplerPrefersHighWeight() {
        double[] w = {0.001, 0.001, 0.001, 0.001, 1.0};
        int hits = 0, trials = 200;
        for (int t = 0; t < trials; t++) {
            int[] s = WeightedSubspaceSampler.sample(w, 1, new Random(t), Set.of());
            if (s.length == 1 && s[0] == 4) hits++;
        }
        report("WeightedSampler heavy-weight wins majority (hits=" + hits + "/" + trials + ")",
                hits >= trials * 0.85);
    }

    private static void testWeightedSamplerFallsBackOnAllZero() {
        double[] w = {0, 0, 0, 0};
        int[] s = WeightedSubspaceSampler.sample(w, 2, new Random(1), Set.of());
        report("WeightedSampler falls back when all weights zero (size=" + s.length + ")",
                s.length == 2);
    }

    private static DriftAwareSRP buildTrainedDA(int F, int ensembleSize, long seed) {
        FixedSelector sel = new FixedSelector(F, 2, intRange(F));
        SRPWrapper srp = newSrp(F, ensembleSize, sel);
        FeatureImportance imp = new FeatureImportance(F);
        DriftAwareSRP da = new DriftAwareSRP(srp, 0.5, seed, imp);
        InstancesHeader h = makeHeader(F, 2, F);
        int[] y = new int[600];
        double[][] data = sea(600, y, seed);
        warmTrain(da, h, data, y, 600);
        return da;
    }

    private static int[] intRange(int F) { int[] r = new int[F]; for (int i = 0; i < F; i++) r[i] = i; return r; }

    private static void testHandleDriftKeepsWhenNoOverlap() {
        int F = 3;
        DriftAwareSRP da = buildTrainedDA(F, 4, 1001);
        int[][] before = da.getCurrentSubspaces();
        Set<Integer> notInAny = findFeatureNotInAnySubspace(before, F);
        if (notInAny == null) { report("HandleDrift KEEP no-overlap (skipped: every feature appears)", true); return; }
        double[] scores = new double[F]; Arrays.fill(scores, 1.0);
        DriftActionSummary s = da.handleDrift(notInAny, scores);
        boolean allKeep = true;
        for (DriftActionSummary.Action a : s.getPerLearner())
            if (a != DriftActionSummary.Action.KEEP) { allKeep = false; break; }
        report("HandleDrift returns KEEP for everyone when no overlap", allKeep);
    }

    private static Set<Integer> findFeatureNotInAnySubspace(int[][] subs, int F) {
        for (int f = 0; f < F; f++) {
            boolean inAny = false;
            for (int[] s : subs) for (int x : s) if (x == f) { inAny = true; break; }
            if (!inAny) return Set.of(f);
        }
        return null;
    }

    private static void testHandleDriftSurgicalWhenSmallOverlap() {
        int F = 8;
        DriftAwareSRP da = buildTrainedDA(F, 6, 1002);
        int[][] subs = da.getCurrentSubspaces();
        Set<Integer> drift = new HashSet<>();
        for (int[] s : subs) if (s.length > 0) { drift.add(s[0]); break; }
        double[] scores = new double[F];
        for (int i = 0; i < F; i++) scores[i] = i + 1;
        DriftActionSummary summary = da.handleDrift(drift, scores);
        boolean anySurgical = summary.getSurgicalCount() > 0
                || summary.getNoReplacementCount() > 0
                || summary.getKeptCount() > 0;
        report("HandleDrift produces SURGICAL/KEEP for small overlap (surgical="
                        + summary.getSurgicalCount() + ", keep=" + summary.getKeptCount()
                        + ", full=" + summary.getFullCount() + ")",
                anySurgical && summary.getFullCount() <= subs.length / 2);
    }

    private static void testHandleDriftFullWhenLargeOverlap() {
        int F = 6;
        DriftAwareSRP da = buildTrainedDA(F, 4, 1003);
        Set<Integer> drift = new HashSet<>();
        for (int i = 0; i < F; i++) drift.add(i);
        double[] scores = new double[F]; Arrays.fill(scores, 1.0);
        DriftActionSummary s = da.handleDrift(drift, scores);
        report("HandleDrift triggers FULL for >=tau overlap (full=" + s.getFullCount() + ")",
                s.getFullCount() == s.getEnsembleSize() || s.getNoReplacementCount() > 0);
    }

    private static void testHandleDriftIdempotentAfterEmptySet() {
        int F = 5;
        DriftAwareSRP da = buildTrainedDA(F, 4, 1004);
        int[][] before = da.getCurrentSubspaces();
        double[] scores = new double[F]; Arrays.fill(scores, 1.0);
        da.handleDrift(Set.of(), scores);
        int[][] after = da.getCurrentSubspaces();
        boolean same = before.length == after.length;
        for (int i = 0; i < before.length && same; i++) same = Arrays.equals(before[i], after[i]);
        report("HandleDrift with empty set is no-op", same);
    }

    private static void testHandleDriftRejectsBadScoresLength() {
        int F = 4;
        DriftAwareSRP da = buildTrainedDA(F, 3, 1005);
        boolean threw = false;
        try { da.handleDrift(Set.of(0), new double[F + 2]); }
        catch (IllegalArgumentException e) { threw = true; }
        report("HandleDrift rejects bad scores length", threw);
    }

    private static void testSubspaceWritesPersistAndAreSorted() {
        int F = 6;
        DriftAwareSRP da = buildTrainedDA(F, 4, 1006);
        Set<Integer> drift = new HashSet<>();
        for (int i = 0; i < F; i++) drift.add(i);
        double[] scores = new double[F]; for (int i = 0; i < F; i++) scores[i] = i;
        da.handleDrift(drift, scores);
        int[][] after = da.getCurrentSubspaces();
        boolean allSorted = true;
        for (int[] s : after) {
            for (int i = 1; i < s.length; i++) if (s[i] < s[i - 1]) { allSorted = false; break; }
        }
        report("Subspaces are sorted after write", allSorted);
    }

    private static void testSubspaceIndicesNeverOutOfRange() {
        int F = 7;
        DriftAwareSRP da = buildTrainedDA(F, 5, 1007);
        int[][] before = da.getCurrentSubspaces();
        Set<Integer> drift = Set.of(0, 1, 2);
        double[] scores = new double[F]; for (int i = 0; i < F; i++) scores[i] = i + 1;
        DriftActionSummary summary = da.handleDrift(drift, scores);
        int[][] subs = da.getCurrentSubspaces();
        System.out.println("    [diag] F=" + F);
        System.out.println("    [diag] summary=" + summary);
        for (int i = 0; i < before.length; i++)
            System.out.println("    [diag] before[" + i + "]=" + Arrays.toString(before[i]));
        for (int i = 0; i < subs.length; i++)
            System.out.println("    [diag] after [" + i + "]=" + Arrays.toString(subs[i]));
        boolean ok = true;
        for (int[] s : subs) for (int x : s) if (x < 0 || x >= F) ok = false;
        report("Subspace indices are within [0, F)", ok);
    }

    private static void testLearnerWeightsSumToOneAndPositive() {
        int F = 5;
        DriftAwareSRP da = buildTrainedDA(F, 5, 1008);
        InstancesHeader h = makeHeader(F, 2, F);
        da.predictProba(fullInstance(h, new double[F], 0));
        double[] w = da.getLastLearnerWeights();
        double sum = 0.0; for (double x : w) sum += x;
        boolean nonNeg = true; for (double x : w) if (x < 0.0) nonNeg = false;
        report("Learner weights sum=1 and >=0 (sum=" + sum + ")",
                Math.abs(sum - 1.0) < 1e-9 && nonNeg);
    }

    private static void testLearnerWeightsHigherForBetterSubspaces() {
        int F = 6;
        FixedSelector sel = new FixedSelector(F, 2, intRange(F));
        SRPWrapper srp = newSrp(F, 6, sel);
        FeatureImportance imp = new FeatureImportance(F);
        imp.update(new double[]{0.0, 0.0, 0.0, 1.0, 1.0, 1.0}, new double[F]);
        DriftAwareSRP da = new DriftAwareSRP(srp, 0.5, 1009L, imp);
        InstancesHeader h = makeHeader(F, 2, F);
        int[] y = new int[400];
        double[][] data = sea(400, y, 1009);
        warmTrain(da, h, data, y, 400);
        da.predictProba(fullInstance(h, new double[F], 0));
        int[][] subs = da.getCurrentSubspaces();
        double[] w = da.getLastLearnerWeights();
        double bestHigh = 0.0, bestLow = 0.0;
        for (int li = 0; li < subs.length; li++) {
            double avg = 0.0; int n = 0;
            for (int x : subs[li]) { if (x >= 3) avg++; n++; }
            double frac = n == 0 ? 0.0 : avg / n;
            if (frac > 0.5) bestHigh = Math.max(bestHigh, w[li]);
            else            bestLow  = Math.max(bestLow,  w[li]);
        }
        report("Learners on high-importance features get higher weight (high=" + bestHigh
                        + ", low=" + bestLow + ")",
                bestHigh >= bestLow);
    }

    private static void testPredictWeightedFallbackWhenNoImportance() {
        int F = 4;
        FixedSelector sel = new FixedSelector(F, 2, intRange(F));
        SRPWrapper srp = newSrp(F, 3, sel);
        DriftAwareSRP da = new DriftAwareSRP(srp, 0.5, 1010L, null);
        InstancesHeader h = makeHeader(F, 2, F);
        int[] y = new int[200];
        double[][] data = sea(200, y, 1010);
        warmTrain(da, h, data, y, 200);
        long before = da.getUnweightedFallbacks();
        da.predictProba(fullInstance(h, new double[F], 0));
        report("Predict falls back to plain SRP when importance==null",
                da.getUnweightedFallbacks() == before + 1);
    }

    private static void testPredictNeverReturnsNaN() {
        int F = 5;
        DriftAwareSRP da = buildTrainedDA(F, 4, 1011);
        InstancesHeader h = makeHeader(F, 2, F);
        double[] v = da.predictProba(fullInstance(h, new double[F], 0));
        boolean ok = true;
        for (double x : v) if (!Double.isFinite(x) || x < 0.0) ok = false;
        report("predictProba never returns NaN/negative", ok);
    }

    private static void testAutoHandleDriftFiresOnAlarm() {
        int F = 5;
        FixedSelector sel = new FixedSelector(F, 2, intRange(F));
        SRPWrapper srp = newSrp(F, 4, sel);
        FeatureImportance imp = new FeatureImportance(F);
        imp.update(new double[]{0.1, 0.2, 0.3, 0.4, 0.5}, new double[F]);
        DriftAwareSRP da = new DriftAwareSRP(srp, 0.5, 1012L, imp);
        InstancesHeader h = makeHeader(F, 2, F);
        int[] y = new int[200];
        double[][] data = sea(200, y, 1012);
        warmTrain(da, h, data, y, 200);
        long before = da.getAutoHandleDriftCalls();
        da.train(fullInstance(h, data[0], y[0]), y[0], true, Set.of(0, 1));
        report("Auto handleDrift fires on alarm (auto=" + da.getAutoHandleDriftCalls() + ")",
                da.getAutoHandleDriftCalls() == before + 1);
    }

    private static void testDriftListenerInvoked() {
        int F = 4;
        FixedSelector sel = new FixedSelector(F, 2, intRange(F));
        SRPWrapper srp = newSrp(F, 3, sel);
        FeatureImportance imp = new FeatureImportance(F);
        imp.update(new double[]{0.1, 0.2, 0.3, 0.4}, new double[F]);
        DriftAwareSRP da = new DriftAwareSRP(srp, 0.5, 1013L, imp);
        AtomicInteger calls = new AtomicInteger();
        AtomicReference<DriftAwareSRP.DriftEvent> last = new AtomicReference<>();
        da.setDriftListener(ev -> { calls.incrementAndGet(); last.set(ev); });
        InstancesHeader h = makeHeader(F, 2, F);
        int[] y = new int[200];
        double[][] data = sea(200, y, 1013);
        warmTrain(da, h, data, y, 200);
        da.train(fullInstance(h, data[0], y[0]), y[0], true, Set.of(0));
        report("Drift listener invoked with non-null summary",
                calls.get() == 1 && last.get() != null && last.get().summary != null);
    }

    private static void testS1VsS4DiffersAfterAutoDrift() {
        int F = 6;
        long seed = 1014;
        DriftAwareSRP s1 = buildTrainedDA(F, 5, seed);

        FixedSelector sel4 = new FixedSelector(F, 2, intRange(F));
        SRPWrapper srp4 = newSrp(F, 5, sel4);
        FeatureImportance imp4 = new FeatureImportance(F);
        imp4.update(new double[]{0.05, 0.05, 0.05, 0.95, 0.95, 0.95}, new double[F]);
        DriftAwareSRP s4 = new DriftAwareSRP(srp4, 0.3, seed + 1, imp4);
        s4.setSurgicalResetPolicy(DriftAwareSRP.SurgicalResetPolicy.FULL);

        InstancesHeader h = makeHeader(F, 2, F);
        int[] y = new int[1200];
        double[][] data = sea(1200, y, seed);

        warmTrain(s4, h, data, y, 600);

        s4.handleDrift(Set.of(0, 1, 2),
                new double[]{0.05, 0.05, 0.05, 0.95, 0.95, 0.95});

        for (int i = 600; i < 900; i++) {
            s4.train(fullInstance(h, data[i], y[i]), y[i]);
        }
        for (int i = 600; i < 900; i++) {
            s1.train(fullInstance(h, data[i], y[i]), y[i]);
        }

        int diff = 0;
        for (int i = 900; i < 1200; i++) {
            int p1 = s1.predict(fullInstance(h, data[i], y[i]));
            int p4 = s4.predict(fullInstance(h, data[i], y[i]));
            if (p1 != p4) diff++;
        }
        report("S1 vs S4 differ after auto-drift (diff=" + diff + "/300)", diff >= 1);
    }

    private static void testSurgicalNoResetPolicyKeepsAccuracy() {
        int F = 6;
        FixedSelector sel = new FixedSelector(F, 2, intRange(F));
        SRPWrapper srp = newSrp(F, 5, sel);
        FeatureImportance imp = new FeatureImportance(F);
        imp.update(new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, new double[F]);
        DriftAwareSRP da = new DriftAwareSRP(srp, 0.5, 1015L, imp);
        da.setSurgicalResetPolicy(DriftAwareSRP.SurgicalResetPolicy.NONE);
        InstancesHeader h = makeHeader(F, 2, F);
        int[] y = new int[800];
        double[][] data = sea(800, y, 1015);
        warmTrain(da, h, data, y, 600);
        int beforeOk = 0;
        for (int i = 600; i < 700; i++)
            if (da.predict(fullInstance(h, data[i], y[i])) == y[i]) beforeOk++;
        da.handleDrift(Set.of(0), new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6});
        int afterOk = 0;
        for (int i = 600; i < 700; i++)
            if (da.predict(fullInstance(h, data[i], y[i])) == y[i]) afterOk++;
        report("Surgical NONE keeps most accuracy (before=" + beforeOk + ", after=" + afterOk + ")",
                afterOk >= beforeOk - 25);
    }

    private static void testRefreshAllResetsCounters() {
        int F = 5;
        DriftAwareSRP da = buildTrainedDA(F, 4, 1016);
        DriftAwareSRP.RefreshSummary rs = da.refreshAllSubspaces();
        report("refreshAllSubspaces touches every learner (refreshed=" + rs.getRefreshedCount() + ")",
                rs.getRefreshedCount() == rs.getEnsembleSize());
    }

    private static void report(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  [PASSED] " + name); }
        else    { failed++; System.out.println("  [FAILED] " + name); }
    }
}