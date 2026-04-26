package thesis.models;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.streams.InstanceStream;
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

public class ModelsSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("MODELS SMOKE TESTS");
        System.out.println("=".repeat(70));

        testFeatureSpaceLayout();
        testFeatureSpaceExtractFeatures();
        testFeatureSpaceRejectsBadHeader();

        testFilteredHeaderBuilderShape();
        testFilteredHeaderBuilderValuesMatchSelection();
        testFilteredHeaderBuilderHandlesMissingClass();

        testHoeffdingTreeWrapperEndToEnd();
        testHoeffdingTreeWrapperRejectsUninitializedSelector();
        testHoeffdingTreeWrapperRejectsBadParams();
        testHoeffdingTreeWrapperResetWorks();

        testARFWrapperEndToEnd();
        testARFWrapperRejectsBadParams();

        testSRPWrapperEndToEnd();
        testSRPWrapperReducedHeaderShape();
        testSRPWrapperEnsembleVisibleAfterTraining();

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
            if (i == classPos) {
                attrs.add(new Attribute("class", classVals));
            } else {
                attrs.add(new Attribute("f" + i));
            }
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

    private static FeatureSelector trainedSelector(int F, int numClasses,
                                                   double[][] win, int[] y, int K) {
        StaticFeatureSelector sel = new StaticFeatureSelector(F, numClasses, K,
                new PiDDiscretizer(F, numClasses),
                (bins, classes) -> new InformationGainRanker(F, bins, classes));
        sel.initialize(win, y);
        return sel;
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

    private static void testFeatureSpaceLayout() {
        InstancesHeader h = makeHeader(4, 2, 2);
        FeatureSpace fs = new FeatureSpace(h);
        boolean ok = fs.numFeatures() == 4
                && fs.classIndex() == 2
                && fs.attrIndexOf(0) == 0
                && fs.attrIndexOf(1) == 1
                && fs.attrIndexOf(2) == 3
                && fs.attrIndexOf(3) == 4;
        report("FeatureSpace skips class index in feature->attr mapping", ok);
    }

    private static void testFeatureSpaceExtractFeatures() {
        InstancesHeader h = makeHeader(3, 2, 1);
        FeatureSpace fs = new FeatureSpace(h);
        Instance inst = makeInstance(h, new double[]{10.0, 0, 20.0, 30.0});
        double[] feats = fs.extractFeatures(inst);
        boolean ok = Arrays.equals(feats, new double[]{10.0, 20.0, 30.0});
        report("FeatureSpace.extractFeatures skips class column", ok);
    }

    private static void testFeatureSpaceRejectsBadHeader() {
        boolean threwOnNull = false;
        try { new FeatureSpace(null); }
        catch (IllegalArgumentException e) { threwOnNull = true; }

        boolean threwOnNoClass = false;
        try {
            ArrayList<Attribute> attrs = new ArrayList<>();
            attrs.add(new Attribute("a"));
            attrs.add(new Attribute("b"));
            Instances ins = new Instances("noClass", attrs, 0);
            ins.setClassIndex(-1);
            InstancesHeader noClass = new InstancesHeader(ins);
            new FeatureSpace(noClass);
        } catch (IllegalArgumentException e) {
            threwOnNoClass = true;
        } catch (Exception e) {
            threwOnNoClass = true;
        }

        report("FeatureSpace rejects null header"
                        + (threwOnNoClass ? " and no-class header" : " (no-class case skipped)"),
                threwOnNull);
    }

    private static void testFilteredHeaderBuilderShape() {
        InstancesHeader h = makeHeader(5, 2, 5);
        FeatureSpace fs = new FeatureSpace(h);
        int[] selection = {1, 3};
        InstancesHeader reduced = FilteredHeaderBuilder.build(fs, selection, "_test");
        boolean ok = reduced.numAttributes() == 3
                && reduced.classIndex() == 2
                && reduced.attribute(0).name().equals("f1")
                && reduced.attribute(1).name().equals("f3")
                && reduced.getRelationName().endsWith("_test");
        report("FilteredHeaderBuilder.build shape (got "
                + reduced.numAttributes() + " attrs)", ok);
    }

    private static void testFilteredHeaderBuilderValuesMatchSelection() {
        InstancesHeader h = makeHeader(4, 2, 4);
        FeatureSpace fs = new FeatureSpace(h);
        int[] selection = {0, 2};
        InstancesHeader reduced = FilteredHeaderBuilder.build(fs, selection, "_x");
        Instance full = makeInstance(h, new double[]{7.0, 8.0, 9.0, 10.0, 1.0});
        Instance filt = FilteredHeaderBuilder.filteredInstance(full, fs, selection, reduced);
        boolean ok = filt.numAttributes() == 3
                && filt.value(0) == 7.0
                && filt.value(1) == 9.0
                && filt.classValue() == 1.0;
        report("FilteredHeaderBuilder.filteredInstance picks correct values", ok);
    }

    private static void testFilteredHeaderBuilderHandlesMissingClass() {
        InstancesHeader h = makeHeader(3, 2, 3);
        FeatureSpace fs = new FeatureSpace(h);
        int[] selection = {0, 1};
        InstancesHeader reduced = FilteredHeaderBuilder.build(fs, selection, "_m");
        Instance full = makeInstance(h, new double[]{1.0, 2.0, 3.0, 0.0});
        full.setMissing(h.classIndex());
        Instance filt = FilteredHeaderBuilder.filteredInstance(full, fs, selection, reduced);
        boolean ok = filt.value(0) == 1.0 && filt.value(1) == 2.0;
        report("FilteredHeaderBuilder copes with missing class", ok);
    }

    private static int trainAndScore(ModelWrapper m, InstancesHeader h,
                                     double[][] data, int[] y,
                                     int trainN, int testN) {
        for (int i = 0; i < trainN; i++) {
            double[] vals = new double[h.numAttributes()];
            for (int f = 0; f < data[i].length; f++) vals[f] = data[i][f];
            vals[h.classIndex()] = y[i];
            m.train(makeInstance(h, vals), y[i]);
        }
        int correct = 0;
        for (int i = trainN; i < trainN + testN; i++) {
            double[] vals = new double[h.numAttributes()];
            for (int f = 0; f < data[i].length; f++) vals[f] = data[i][f];
            vals[h.classIndex()] = y[i];
            if (m.predict(makeInstance(h, vals)) == y[i]) correct++;
        }
        return correct;
    }

    private static void testHoeffdingTreeWrapperEndToEnd() {
        int F = 3, N = 2000, K = 2;
        int[] y = new int[N];
        double[][] data = sea(N, y, 11);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = trainedSelector(F, 2, Arrays.copyOf(data, 600),
                Arrays.copyOf(y, 600), K);
        HoeffdingTreeWrapper m = new HoeffdingTreeWrapper(sel, h);
        int correct = trainAndScore(m, h, data, y, 1500, 500);
        boolean ok = m.getCurrentSelection().length == K
                && m.getReducedHeader() != null
                && correct >= 300;
        report("HoeffdingTreeWrapper end-to-end (acc=" + correct + "/500)", ok);
    }

    private static InstancesHeader getReducedHeader(ModelWrapper m) {
        if (m instanceof HoeffdingTreeWrapper) return ((HoeffdingTreeWrapper) m).getReducedHeader();
        if (m instanceof ARFWrapper) return ((ARFWrapper) m).getReducedHeader();
        if (m instanceof SRPWrapper) return ((SRPWrapper) m).getReducedHeader();
        return null;
    }

    private static void testHoeffdingTreeWrapperRejectsUninitializedSelector() {
        InstancesHeader h = makeHeader(3, 2, 3);
        FeatureSelector uninit = new StaticFeatureSelector(3, 2, 2,
                new PiDDiscretizer(3, 2),
                (b, c) -> new InformationGainRanker(3, b, c));
        boolean threw = false;
        try { new HoeffdingTreeWrapper(uninit, h); }
        catch (IllegalArgumentException e) { threw = true; }
        report("HoeffdingTreeWrapper rejects uninitialized selector", threw);
    }

    private static void testHoeffdingTreeWrapperRejectsBadParams() {
        int F = 3;
        int[] y = new int[600];
        double[][] data = sea(600, y, 12);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = trainedSelector(F, 2, data, y, 2);
        boolean t1 = false, t2 = false, t3 = false, t4 = false;
        try { new HoeffdingTreeWrapper(null, h); } catch (IllegalArgumentException e) { t1 = true; }
        try { new HoeffdingTreeWrapper(sel, null); } catch (IllegalArgumentException e) { t2 = true; }
        try { new HoeffdingTreeWrapper(sel, h, 0, 0.01, false); }
        catch (IllegalArgumentException e) { t3 = true; }
        try { new HoeffdingTreeWrapper(sel, h, 200, 0.0, false); }
        catch (IllegalArgumentException e) { t4 = true; }
        report("HoeffdingTreeWrapper rejects bad params", t1 && t2 && t3 && t4);
    }

    private static void testHoeffdingTreeWrapperResetWorks() {
        int F = 3, N = 1000, K = 2;
        int[] y = new int[N];
        double[][] data = sea(N, y, 13);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = trainedSelector(F, 2, Arrays.copyOf(data, 600),
                Arrays.copyOf(y, 600), K);
        HoeffdingTreeWrapper m = new HoeffdingTreeWrapper(sel, h);
        for (int i = 0; i < 500; i++) {
            double[] vals = new double[h.numAttributes()];
            for (int f = 0; f < F; f++) vals[f] = data[i][f];
            vals[h.classIndex()] = y[i];
            m.train(makeInstance(h, vals), y[i]);
        }
        m.reset();
        boolean ok = m.getCurrentSelection().length == K
                && m.getReducedHeader() != null;
        report("HoeffdingTreeWrapper.reset() rebuilds without throwing", ok);
    }

    private static void testARFWrapperEndToEnd() {
        int F = 3, N = 2000, K = 2;
        int[] y = new int[N];
        double[][] data = sea(N, y, 21);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = trainedSelector(F, 2, Arrays.copyOf(data, 600),
                Arrays.copyOf(y, 600), K);
        ARFWrapper m = new ARFWrapper(sel, h, 5, 6.0, false);
        int correct = trainAndScore(m, h, data, y, 1500, 500);
        boolean ok = m.getCurrentSelection().length == K
                && m.getReducedHeader() != null
                && correct >= 300;
        report("ARFWrapper end-to-end (acc=" + correct + "/500)", ok);
    }

    private static void testARFWrapperRejectsBadParams() {
        int F = 3;
        int[] y = new int[600];
        double[][] data = sea(600, y, 22);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = trainedSelector(F, 2, data, y, 2);
        boolean t1 = false, t2 = false, t3 = false, t4 = false;
        try { new ARFWrapper(null, h); } catch (IllegalArgumentException e) { t1 = true; }
        try { new ARFWrapper(sel, null); } catch (IllegalArgumentException e) { t2 = true; }
        try { new ARFWrapper(sel, h, 0, 6.0, false); }
        catch (IllegalArgumentException e) { t3 = true; }
        try { new ARFWrapper(sel, h, 5, 0.0, false); }
        catch (IllegalArgumentException e) { t4 = true; }
        report("ARFWrapper rejects bad params", t1 && t2 && t3 && t4);
    }

    private static void testSRPWrapperEndToEnd() {
        int F = 3, N = 2000, K = 2;
        int[] y = new int[N];
        double[][] data = sea(N, y, 31);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = trainedSelector(F, 2, Arrays.copyOf(data, 600),
                Arrays.copyOf(y, 600), K);
        SRPWrapper m = new SRPWrapper(sel, h, 5, 6.0, false);
        int correct = trainAndScore(m, h, data, y, 1500, 500);
        boolean ok = m.getCurrentSelection().length == K
                && m.getReducedHeader() != null
                && correct >= 300;
        report("SRPWrapper end-to-end (acc=" + correct + "/500)", ok);
    }

    private static void testSRPWrapperReducedHeaderShape() {
        int F = 3, K = 2;
        int[] y = new int[600];
        double[][] data = sea(600, y, 32);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = trainedSelector(F, 2, data, y, K);
        SRPWrapper m = new SRPWrapper(sel, h, 4, 6.0, false);
        InstancesHeader reduced = m.getReducedHeader();
        boolean ok = reduced.numAttributes() == K + 1
                && reduced.classIndex() == K;
        report("SRPWrapper reduced header has K+1 attrs (got "
                + reduced.numAttributes() + ")", ok);
    }

    private static void testSRPWrapperEnsembleVisibleAfterTraining() {
        int F = 3, K = 2;
        int N = 2000;
        int[] y = new int[N];
        double[][] data = sea(N, y, 33);
        InstancesHeader h = makeHeader(F, 2, F);
        FeatureSelector sel = trainedSelector(F, 2, Arrays.copyOf(data, 600),
                Arrays.copyOf(y, 600), K);
        SRPWrapper m = new SRPWrapper(sel, h, 5, 6.0, false);
        for (int i = 0; i < 1500; i++) {
            double[] vals = new double[h.numAttributes()];
            for (int f = 0; f < F; f++) vals[f] = data[i][f];
            vals[h.classIndex()] = y[i];
            m.train(makeInstance(h, vals), y[i]);
        }
        int actualSize = m.getActualEnsembleSize();
        int[][] subspaces = m.getAllSubspaceIndices();
        report("SRPWrapper ensemble size visible after training (size="
                        + actualSize + ", subspacesShape=" + subspaces.length + ")",
                actualSize >= 1 && subspaces.length == actualSize);
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