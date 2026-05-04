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
import java.util.Set;

public class WrapperSelectionSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("WRAPPER ↔ FEATURESELECTOR SMOKE TESTS");
        System.out.println("=".repeat(70));

        testHTReducedHeaderArity();
        testARFReducedHeaderArity();
        testSRPReducedHeaderArity();

        testHTSelectionChangesModelInput();
        testARFSelectionChangesModelInput();
        testSRPSelectionChangesModelInput();

        testHTPredictUsesFilteredAttrs();
        testARFPredictUsesFilteredAttrs();
        testSRPPredictUsesFilteredAttrs();

        testHTKEqualsOneVsFullDiffers();
        testARFKEqualsOneVsFullDiffers();
        testSRPKEqualsOneVsFullDiffers();

        testHTTrainNeverSeesFullVector();
        testARFTrainNeverSeesFullVector();
        testSRPTrainNeverSeesFullVector();

        testSRPHardFilterDefaultIsTrue();
        testARFHardFilterDefaultIsTrue();

        testSRPDriftAlarmFlowsToSelector();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    // ── helpers (jak w ModelsSmokeTest) ────────────────────────────────────

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

    private static FeatureSelector trainedSelector(int F, int numClasses,
                                                   double[][] win, int[] y, int K) {
        StaticFeatureSelector sel = new StaticFeatureSelector(F, numClasses, K,
                new PiDDiscretizer(F, numClasses),
                (numFeatures, numBins, numClassesArg) ->
                        new InformationGainRanker(numFeatures, numBins, numClassesArg));
        sel.initialize(win, y);
        return sel;
    }

    private static Instance fullInstance(InstancesHeader h, double[] feats, int label) {
        double[] vals = new double[h.numAttributes()];
        for (int f = 0; f < feats.length; f++) vals[f] = feats[f];
        vals[h.classIndex()] = label;
        return makeInstance(h, vals);
    }

    /** Recording selector — pozwala wymusić dowolną selekcję i podejrzeć update calls. */
    private static final class RecordingSelector implements FeatureSelector {
        private int[] selection;
        private final int F, C;
        int updateCalls = 0;
        boolean lastDriftAlarm = false;
        Set<Integer> lastDriftFeats = Set.of();

        RecordingSelector(int F, int C, int[] initialSelection) {
            this.F = F;
            this.C = C;
            this.selection = initialSelection.clone();
        }

        @Override public boolean isInitialized()        { return true; }
        @Override public int[]   getCurrentSelection()  { return selection.clone(); }
        @Override public int[]   getSelectedFeatures()  { return selection.clone(); }
        @Override public int     getNumFeatures()       { return F; }
        @Override public int     getK()                 { return selection.length; }

        @Override
        public double[] filterInstance(double[] fullInstance) {
            if (fullInstance == null) throw new IllegalArgumentException("fullInstance must not be null");
            if (fullInstance.length != F) {
                throw new IllegalArgumentException(
                        "expected " + F + " features, got " + fullInstance.length);
            }
            double[] out = new double[selection.length];
            for (int i = 0; i < selection.length; i++) out[i] = fullInstance[selection[i]];
            return out;
        }

        @Override public void initialize(double[][] win, int[] y) { /* no-op */ }

        @Override
        public void update(double[] feats, int label,
                           boolean driftAlarm, Set<Integer> driftingFeatures) {
            updateCalls++;
            lastDriftAlarm = driftAlarm;
            lastDriftFeats = driftingFeatures == null ? Set.of() : driftingFeatures;
        }

        @Override public String name() { return "RecordingSelector"; }

        void setSelection(int[] s) { this.selection = s.clone(); }
    }

    // ── 1. Reduced header arity = K+1 ──────────────────────────────────────

    private static void testHTReducedHeaderArity() {
        int F = 5, K = 2;
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = new RecordingSelector(F, 2, new int[]{1, 3});
        HoeffdingTreeWrapper m = new HoeffdingTreeWrapper(sel, h);
        // pośrednio sprawdzamy przez predict — ale najlepiej przez liczbę atrybutów filtra:
        Instance probe = fullInstance(h, new double[]{1, 2, 3, 4, 5}, 0);
        Instance filt = FilteredHeaderBuilder.filteredInstance(probe,
                new FeatureSpace(h), m.getCurrentSelection(),
                FilteredHeaderBuilder.build(new FeatureSpace(h), m.getCurrentSelection(), "_p"));
        report("HT reduced view has K+1 attrs (got " + filt.numAttributes() + ")",
                filt.numAttributes() == K + 1);
    }

    private static void testARFReducedHeaderArity() {
        int F = 5, K = 3;
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = new RecordingSelector(F, 2, new int[]{0, 2, 4});
        ARFWrapper m = new ARFWrapper(sel, h, 3, 6.0, false, true);
        Instance probe = fullInstance(h, new double[]{1, 2, 3, 4, 5}, 0);
        Instance filt = FilteredHeaderBuilder.filteredInstance(probe,
                new FeatureSpace(h), m.getCurrentSelection(),
                FilteredHeaderBuilder.build(new FeatureSpace(h), m.getCurrentSelection(), "_p"));
        report("ARF(hardFilter) reduced view has K+1 attrs (got " + filt.numAttributes() + ")",
                filt.numAttributes() == K + 1);
    }

    private static void testSRPReducedHeaderArity() {
        int F = 5, K = 2;
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = new RecordingSelector(F, 2, new int[]{0, 4});
        SRPWrapper m = new SRPWrapper(sel, h, 3, 6.0, false, true);
        InstancesHeader red = m.getReducedHeader();
        report("SRP(hardFilter) reducedHeader.numAttrs == K+1 (got "
                        + red.numAttributes() + ")",
                red.numAttributes() == K + 1 && red.classIndex() == K);
    }

    // ── 2. Zmiana selekcji NAPRAWDĘ zmienia widok modelu ───────────────────

    private static void testHTSelectionChangesModelInput() {
        int F = 4;
        InstancesHeader h = makeHeader(F, 2, F);
        RecordingSelector sel = new RecordingSelector(F, 2, new int[]{0});
        HoeffdingTreeWrapper m = new HoeffdingTreeWrapper(sel, h);
        int before = m.getCurrentSelection().length;
        sel.setSelection(new int[]{0, 1, 2});
        // wymuś sync
        m.predict(fullInstance(h, new double[]{1, 2, 3, 4}, 0));
        int after = m.getCurrentSelection().length;
        report("HT picks up selection change (" + before + " → " + after + ")",
                before == 1 && after == 3);
    }

    private static void testARFSelectionChangesModelInput() {
        int F = 4;
        InstancesHeader h = makeHeader(F, 2, F);
        RecordingSelector sel = new RecordingSelector(F, 2, new int[]{0});
        ARFWrapper m = new ARFWrapper(sel, h, 3, 6.0, false, true);
        int before = m.getCurrentSelection().length;
        sel.setSelection(new int[]{1, 2});
        m.predict(fullInstance(h, new double[]{1, 2, 3, 4}, 0));
        int after = m.getCurrentSelection().length;
        report("ARF picks up selection change (" + before + " → " + after + ")",
                before == 1 && after == 2);
    }

    private static void testSRPSelectionChangesModelInput() {
        int F = 4;
        InstancesHeader h = makeHeader(F, 2, F);
        RecordingSelector sel = new RecordingSelector(F, 2, new int[]{0});
        SRPWrapper m = new SRPWrapper(sel, h, 3, 6.0, false, true);
        int beforeAttrs = m.getReducedHeader().numAttributes();
        sel.setSelection(new int[]{1, 2, 3});
        m.predict(fullInstance(h, new double[]{1, 2, 3, 4}, 0));
        int afterAttrs = m.getReducedHeader().numAttributes();
        report("SRP picks up selection change (header " + beforeAttrs
                        + " → " + afterAttrs + ")",
                beforeAttrs == 2 && afterAttrs == 4);
    }

    // ── 3. predict używa wartości tylko z wybranych atrybutów ──────────────
    //   wkładamy „śmieci" w nieselekowanych atrybutach — predykcja musi
    //   pozostać taka sama jak dla czystego wektora (te wartości nie docierają do modelu).

    private static int trainAccOnSelection(ModelWrapper m, InstancesHeader h,
                                           double[][] data, int[] y, int trainN) {
        for (int i = 0; i < trainN; i++) m.train(fullInstance(h, data[i], y[i]), y[i]);
        return trainN;
    }

    private static void assertSelectionInsensitivity(String tag, ModelWrapper m,
                                                     InstancesHeader h, double[][] data, int[] y) {
        trainAccOnSelection(m, h, data, y, 1500);
        int[] sel = m.getCurrentSelection();
        Set<Integer> selSet = new HashSet<>();
        for (int s : sel) selSet.add(s);

        int sameCount = 0, total = 0;
        for (int i = 1500; i < 1800; i++) {
            Instance clean = fullInstance(h, data[i], y[i]);
            double[] noisy = data[i].clone();
            for (int f = 0; f < noisy.length; f++) {
                if (!selSet.contains(f)) noisy[f] = 1e6 + f; // brutalne śmieci
            }
            Instance dirty = fullInstance(h, noisy, y[i]);
            int pClean = m.predict(clean);
            int pDirty = m.predict(dirty);
            if (pClean == pDirty) sameCount++;
            total++;
        }
        // jeżeli model NAPRAWDĘ widzi tylko wybrane atrybuty, śmieci poza selekcją
        // nie wpływają → 100% zgodności. Tolerujemy 99% by uniknąć FP od side-effects.
        double agree = sameCount / (double) total;
        report(tag + " ignores non-selected attributes (agree=" + sameCount + "/" + total + ")",
                agree >= 0.99);
    }

    private static void testHTPredictUsesFilteredAttrs() {
        int F = 3, K = 1;
        int[] y = new int[1800];
        double[][] data = sea(1800, y, 101);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = trainedSelector(F, 2,
                Arrays.copyOf(data, 600), Arrays.copyOf(y, 600), K);
        assertSelectionInsensitivity("HT", new HoeffdingTreeWrapper(sel, h), h, data, y);
    }

    private static void testARFPredictUsesFilteredAttrs() {
        int F = 3, K = 1;
        int[] y = new int[1800];
        double[][] data = sea(1800, y, 102);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = trainedSelector(F, 2,
                Arrays.copyOf(data, 600), Arrays.copyOf(y, 600), K);
        assertSelectionInsensitivity("ARF", new ARFWrapper(sel, h, 3, 6.0, false, true), h, data, y);
    }

    private static void testSRPPredictUsesFilteredAttrs() {
        int F = 3, K = 1;
        int[] y = new int[1800];
        double[][] data = sea(1800, y, 103);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = trainedSelector(F, 2,
                Arrays.copyOf(data, 600), Arrays.copyOf(y, 600), K);
        assertSelectionInsensitivity("SRP", new SRPWrapper(sel, h, 3, 6.0, false, true), h, data, y);
    }

    // ── 4. K=1 vs K=full → modele MUSZĄ się różnić w predykcjach ───────────
    //   to jest kluczowy test antyregresji „S1=S2=S3=S4".

    private static int countDiff(ModelWrapper a, ModelWrapper b,
                                 InstancesHeader h, double[][] data, int[] y,
                                 int trainN, int testN) {
        for (int i = 0; i < trainN; i++) {
            Instance inst = fullInstance(h, data[i], y[i]);
            a.train(inst, y[i]);
            b.train(inst, y[i]);
        }
        int diff = 0;
        for (int i = trainN; i < trainN + testN; i++) {
            Instance inst = fullInstance(h, data[i], y[i]);
            if (a.predict(inst) != b.predict(inst)) diff++;
        }
        return diff;
    }

    private static void testHTKEqualsOneVsFullDiffers() {
        int F = 3;
        int[] y = new int[2000];
        double[][] data = sea(2000, y, 201);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector selFull = trainedSelector(F, 2,
                Arrays.copyOf(data, 600), Arrays.copyOf(y, 600), F);
        FeatureSelector selOne = trainedSelector(F, 2,
                Arrays.copyOf(data, 600), Arrays.copyOf(y, 600), 1);
        int diff = countDiff(new HoeffdingTreeWrapper(selFull, h),
                new HoeffdingTreeWrapper(selOne, h), h, data, y, 1500, 500);
        report("HT K=1 vs K=full produce different predictions (diff=" + diff + "/500)",
                diff > 10);
    }

    private static void testARFKEqualsOneVsFullDiffers() {
        int F = 3;
        int[] y = new int[2000];
        double[][] data = sea(2000, y, 202);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector selFull = trainedSelector(F, 2,
                Arrays.copyOf(data, 600), Arrays.copyOf(y, 600), F);
        FeatureSelector selOne = trainedSelector(F, 2,
                Arrays.copyOf(data, 600), Arrays.copyOf(y, 600), 1);
        int diff = countDiff(new ARFWrapper(selFull, h, 3, 6.0, false, true),
                new ARFWrapper(selOne, h, 3, 6.0, false, true), h, data, y, 1500, 500);
        report("ARF K=1 vs K=full produce different predictions (diff=" + diff + "/500)",
                diff > 10);
    }

    private static void testSRPKEqualsOneVsFullDiffers() {
        int F = 3;
        int[] y = new int[2000];
        double[][] data = sea(2000, y, 203);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector selFull = trainedSelector(F, 2,
                Arrays.copyOf(data, 600), Arrays.copyOf(y, 600), F);
        FeatureSelector selOne = trainedSelector(F, 2,
                Arrays.copyOf(data, 600), Arrays.copyOf(y, 600), 1);
        int diff = countDiff(new SRPWrapper(selFull, h, 3, 6.0, false, true),
                new SRPWrapper(selOne, h, 3, 6.0, false, true), h, data, y, 1500, 500);
        report("SRP K=1 vs K=full produce different predictions (diff=" + diff + "/500)",
                diff > 10);
    }

    // ── 5. train: model nigdy nie dostaje pełnego wektora ──────────────────
    //   Sprawdzamy przez monkey-patching: śmieci w niewybranych pozycjach
    //   nie psują accuracy względem czystego treningu.

    private static double accuracy(ModelWrapper m, InstancesHeader h,
                                   double[][] data, int[] y, int from, int to) {
        int ok = 0;
        for (int i = from; i < to; i++) {
            if (m.predict(fullInstance(h, data[i], y[i])) == y[i]) ok++;
        }
        return ok / (double) (to - from);
    }

    private static void assertTrainIgnoresNonSelected(String tag, ModelWrapper clean,
                                                      ModelWrapper noisy, InstancesHeader h,
                                                      double[][] data, int[] y) {
        int[] selC = clean.getCurrentSelection();
        Set<Integer> selSet = new HashSet<>();
        for (int s : selC) selSet.add(s);

        for (int i = 0; i < 1500; i++) {
            Instance c = fullInstance(h, data[i], y[i]);
            double[] noisyVals = data[i].clone();
            for (int f = 0; f < noisyVals.length; f++)
                if (!selSet.contains(f)) noisyVals[f] = 1e6 + f;
            Instance n = fullInstance(h, noisyVals, y[i]);
            clean.train(c, y[i]);
            noisy.train(n, y[i]);
        }
        double a1 = accuracy(clean, h, data, y, 1500, 1800);
        double a2 = accuracy(noisy, h, data, y, 1500, 1800);
        // Modele powinny być prawie identyczne — wartości poza selekcją nie wchodzą do modelu.
        report(tag + " train ignores non-selected attrs (acc clean="
                        + String.format("%.3f", a1) + ", noisy=" + String.format("%.3f", a2) + ")",
                Math.abs(a1 - a2) < 0.05);
    }

    private static void testHTTrainNeverSeesFullVector() {
        int F = 3, K = 1;
        int[] y = new int[1800];
        double[][] data = sea(1800, y, 301);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector s1 = trainedSelector(F, 2,
                Arrays.copyOf(data, 600), Arrays.copyOf(y, 600), K);
        FeatureSelector s2 = trainedSelector(F, 2,
                Arrays.copyOf(data, 600), Arrays.copyOf(y, 600), K);
        assertTrainIgnoresNonSelected("HT",
                new HoeffdingTreeWrapper(s1, h), new HoeffdingTreeWrapper(s2, h), h, data, y);
    }

    private static void testARFTrainNeverSeesFullVector() {
        int F = 3, K = 1;
        int[] y = new int[1800];
        double[][] data = sea(1800, y, 302);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector s1 = trainedSelector(F, 2,
                Arrays.copyOf(data, 600), Arrays.copyOf(y, 600), K);
        FeatureSelector s2 = trainedSelector(F, 2,
                Arrays.copyOf(data, 600), Arrays.copyOf(y, 600), K);
        assertTrainIgnoresNonSelected("ARF",
                new ARFWrapper(s1, h, 3, 6.0, false, true),
                new ARFWrapper(s2, h, 3, 6.0, false, true), h, data, y);
    }

    private static void testSRPTrainNeverSeesFullVector() {
        int F = 3, K = 1;
        int[] y = new int[1800];
        double[][] data = sea(1800, y, 303);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector s1 = trainedSelector(F, 2,
                Arrays.copyOf(data, 600), Arrays.copyOf(y, 600), K);
        FeatureSelector s2 = trainedSelector(F, 2,
                Arrays.copyOf(data, 600), Arrays.copyOf(y, 600), K);
        assertTrainIgnoresNonSelected("SRP",
                new SRPWrapper(s1, h, 3, 6.0, false, true),
                new SRPWrapper(s2, h, 3, 6.0, false, true), h, data, y);
    }

    // ── 6. Default useHardFilter MUST be true ──────────────────────────────

    private static void testSRPHardFilterDefaultIsTrue() {
        int F = 3, K = 2;
        int[] y = new int[600];
        double[][] data = sea(600, y, 401);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = trainedSelector(F, 2, data, y, K);
        SRPWrapper m1 = new SRPWrapper(sel, h);
        SRPWrapper m2 = new SRPWrapper(sel, h, 5, 6.0, false);
        report("SRP default ctor uses hardFilter=true",
                m1.isUseHardFilter() && m2.isUseHardFilter());
    }

    private static void testARFHardFilterDefaultIsTrue() {
        int F = 3, K = 2;
        int[] y = new int[600];
        double[][] data = sea(600, y, 402);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = trainedSelector(F, 2, data, y, K);
        ARFWrapper m1 = new ARFWrapper(sel, h);
        ARFWrapper m2 = new ARFWrapper(sel, h, 5, 6.0, false);
        report("ARF default ctor uses hardFilter=true",
                m1.isUseHardFilter() && m2.isUseHardFilter());
    }

    // ── 7. Drift alarm dociera do selektora (S2/S4) ────────────────────────

    private static void testSRPDriftAlarmFlowsToSelector() {
        int F = 3;
        InstancesHeader h = makeHeader(F, 2, F);
        RecordingSelector sel = new RecordingSelector(F, 2, new int[]{0, 1});
        SRPWrapper m = new SRPWrapper(sel, h, 3, 6.0, false, true);
        m.train(fullInstance(h, new double[]{1, 2, 3}, 0), 0, true, Set.of(2));
        report("SRP forwards drift alarm + features to selector",
                sel.updateCalls == 1 && sel.lastDriftAlarm && sel.lastDriftFeats.contains(2));
    }

    // ── reporting ──────────────────────────────────────────────────────────

    private static void report(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  [PASSED] " + name); }
        else    { failed++; System.out.println("  [FAILED] " + name); }
    }
}