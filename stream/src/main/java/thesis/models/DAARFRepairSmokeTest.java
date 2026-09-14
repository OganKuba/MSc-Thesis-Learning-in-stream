package thesis.models;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import thesis.selection.FeatureSelector;

import java.util.ArrayList;
import java.util.Random;
import java.util.Set;

public class DAARFRepairSmokeTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("DA-ARF REPAIR SMOKE TESTS (A2/A3/A4)");
        System.out.println("=".repeat(70));

        testSurgicalModeNoResetsFullAndTreeStaysValid();
        testResetModeStillFullReplaces();
        testIntrinsicOffDisablesIntrinsicDrift();
        testGatingRunsWithoutCrash();

        System.out.println("=".repeat(70));
        System.out.printf("RESULT: %d passed, %d failed%n", passed, failed);
        System.out.println("=".repeat(70));
        if (failed > 0) System.exit(1);
    }

    private static final int F = 12;
    private static final int C = 2;

    private static InstancesHeader makeHeader() {
        ArrayList<Attribute> attrs = new ArrayList<>(F + 1);
        ArrayList<String> classVals = new ArrayList<>();
        for (int c = 0; c < C; c++) classVals.add("c" + c);
        for (int i = 0; i < F; i++) attrs.add(new Attribute("f" + i));
        attrs.add(new Attribute("class", classVals));
        Instances ins = new Instances("synthetic", attrs, 0);
        ins.setClassIndex(F);
        return new InstancesHeader(ins);
    }

    private static Instance inst(InstancesHeader h, Random rng, int[] label) {
        double[] vals = new double[F + 1];
        double sum = 0.0;
        for (int i = 0; i < F; i++) { vals[i] = rng.nextDouble(); sum += (i < 6 ? vals[i] : 0.0); }
        int y = sum > 3.0 ? 1 : 0;
        vals[F] = y;
        label[0] = y;
        Instance in = new DenseInstance(1.0, vals);
        in.setDataset(h);
        return in;
    }

    private static DAARFWrapper build(InstancesHeader h, DAARFWrapper.ExternalActionMode mode,
                                      boolean intrinsic, boolean gate) {
        FeatureImportance imp = new FeatureImportance(F);
        DAARFWrapper da = new DAARFWrapper(new IdentitySelector(), h, C,
                /*ensemble=*/6, /*subspace=*/4, /*lambda=*/6.0,
                /*accWindow=*/200, /*topK=*/0.5,
                /*power=*/2.0, /*beta=*/0.7,
                /*useBkg=*/true, /*warnDelta=*/1e-3, /*driftDelta=*/1e-4,
                /*seed=*/7L, imp);
        da.setExternalActionMode(mode);
        da.setIntrinsicDriftEnabled(intrinsic);
        da.setGateExternalOnPendingBackground(gate);
        da.setExternalResetFraction(0.5);
        return da;
    }

    private static void warmAndAlarm(DAARFWrapper da, InstancesHeader h, int warm, int alarms) {
        Random rng = new Random(1);
        int[] y = new int[1];
        for (int i = 0; i < warm; i++) {
            Instance in = inst(h, rng, y);
            da.train(in, y[0]);
        }
        // Drift alarms with several overlapping, low-importance drifting
        Set<Integer> drifting = Set.of(0, 1, 2, 3, 4, 5, 6, 7);
        for (int a = 0; a < alarms; a++) {
            for (int i = 0; i < 50; i++) {
                Instance in = inst(h, rng, y);
                da.train(in, y[0], i == 0, drifting);
                double[] p = da.predictProba(in);
                assertProbaValid(p);
            }
        }
    }

    private static void testSurgicalModeNoResetsFullAndTreeStaysValid() {
        InstancesHeader h = makeHeader();
        DAARFWrapper da = build(h, DAARFWrapper.ExternalActionMode.SURGICAL, true, false);
        warmAndAlarm(da, h, 400, 6);
        check("surgical: extFullCount stays 0", da.getExtFullCount() == 0);
        check("surgical: some surgical or no-replacement events happened",
                da.getExtSurgicalCount() + da.getExtNoReplacementCount() > 0);
        // Prediction still valid after all the header rebuilds.
        Random rng = new Random(99);
        int[] y = new int[1];
        assertProbaValid(da.predictProba(inst(h, rng, y)));
        check("surgical: no crash after repeated swaps + training", true);
    }

    private static void testResetModeStillFullReplaces() {
        InstancesHeader h = makeHeader();
        DAARFWrapper da = build(h, DAARFWrapper.ExternalActionMode.RESET, true, false);
        warmAndAlarm(da, h, 400, 6);
        check("reset: extFullCount > 0", da.getExtFullCount() > 0);
        check("reset: extSurgicalCount == 0", da.getExtSurgicalCount() == 0);
    }

    private static void testIntrinsicOffDisablesIntrinsicDrift() {
        InstancesHeader h = makeHeader();
        DAARFWrapper da = build(h, DAARFWrapper.ExternalActionMode.RESET, false, false);
        warmAndAlarm(da, h, 800, 6);
        check("intrinsic-off: driftCount == 0", da.getDriftCount() == 0);
        check("intrinsic-off: bkgPromotions == 0", da.getBkgPromotions() == 0);
        check("intrinsic-off: external layer still fires", da.getExtFullCount() > 0);
    }

    private static void testGatingRunsWithoutCrash() {
        InstancesHeader h = makeHeader();
        DAARFWrapper da = build(h, DAARFWrapper.ExternalActionMode.RESET, true, true);
        warmAndAlarm(da, h, 400, 6);
        check("gating: runs without crash", true);
    }

    private static void assertProbaValid(double[] p) {
        if (p == null) { check("proba non-null", false); return; }
        for (double v : p) {
            if (!Double.isFinite(v) || v < 0.0) { check("proba finite & non-negative", false); return; }
        }
    }

    private static void check(String name, boolean ok) {
        if (ok) { passed++; System.out.println("  [PASS] " + name); }
        else { failed++; System.out.println("  [FAIL] " + name); }
    }

    private static final class IdentitySelector implements FeatureSelector {
        private final int[] sel;
        IdentitySelector() { sel = new int[F]; for (int i = 0; i < F; i++) sel[i] = i; }
        @Override public boolean isInitialized() { return true; }
        @Override public int[] getCurrentSelection() { return sel.clone(); }
        @Override public int[] getSelectedFeatures() { return sel.clone(); }
        @Override public int getNumFeatures() { return F; }
        @Override public int getK() { return sel.length; }
        @Override public double[] filterInstance(double[] full) { return full.clone(); }
        @Override public void initialize(double[][] win, int[] y) { }
        @Override public void update(double[] f, int l, boolean a, Set<Integer> df) { }
        @Override public String name() { return "Identity"; }
    }
}
