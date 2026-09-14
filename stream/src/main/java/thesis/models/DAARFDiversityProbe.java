package thesis.models;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.streams.InstanceStream;
import thesis.pipeline.SyntheticStreamFactory;
import thesis.selection.FeatureSelector;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class DAARFDiversityProbe {

    public static void main(String[] args) {
        // Hyperplane (15 signal) + 5 noise = 20 features.
        InstanceStream base = SyntheticStreamFactory.createHyperplane(1, 0.01, 20000);
        InstanceStream stream = SyntheticStreamFactory.addNoiseFeatures(base, 5, 1);
        InstancesHeader header = stream.getHeader();
        int F = header.numAttributes() - 1;
        int C = header.numClasses();

        System.out.println("Hyperplane+noise: F=" + F + ", ensemble N=10");
        probe("DA-ARF narrow (m=ceil(sqrt(F)))", header, C, stream,
                (int) Math.ceil(Math.sqrt(F)));
        probe("DA-ARF wide   (m=ceil(0.75F))  ", header, C, stream,
                (int) Math.ceil(0.75 * F));
    }

    private static void probe(String label, InstancesHeader header, int C,
                              InstanceStream stream, int subspace) {
        FeatureImportance imp = new FeatureImportance(header.numAttributes() - 1);
        DAARFWrapper da = new DAARFWrapper(new IdentitySel(header.numAttributes() - 1), header, C,
                10, subspace, 6.0, 1000, 0.5, 2.0, 0.7, true, 1e-4, 1e-5, 7L, imp);
        // Warm up so trees exist and (uniform) subspaces are set.
        stream.restart();
        for (int i = 0; i < 3000 && stream.hasMoreInstances(); i++) {
            Instance x = stream.nextInstance().getData();
            da.train(x, (int) x.classValue());
        }
        int[][] subs = da.getAllSubspaces();
        int F = header.numAttributes() - 1;
        boolean[] covered = new boolean[F];
        int[] featCount = new int[F];
        for (int[] s : subs) for (int f : s) { covered[f] = true; featCount[f]++; }
        int cov = 0;
        for (boolean b : covered) if (b) cov++;
        // Mean pairwise Jaccard overlap.
        double jsum = 0; int pairs = 0;
        for (int a = 0; a < subs.length; a++)
            for (int b = a + 1; b < subs.length; b++) {
                jsum += jaccard(subs[a], subs[b]);
                pairs++;
            }
        int distinct = countDistinct(subs);
        System.out.printf("  %s  m=%2d | union coverage=%d/%d (%.0f%%) | mean pairwise Jaccard=%.2f | distinct subspaces=%d/10 | max feat reuse=%d/10%n",
                label, subspace, cov, F, 100.0 * cov / F, jsum / pairs, distinct, max(featCount));
    }

    private static double jaccard(int[] a, int[] b) {
        Set<Integer> sa = new HashSet<>();
        for (int x : a) sa.add(x);
        int inter = 0;
        Set<Integer> sb = new HashSet<>();
        for (int x : b) { sb.add(x); if (sa.contains(x)) inter++; }
        int uni = sa.size() + sb.size() - inter;
        return uni == 0 ? 0.0 : (double) inter / uni;
    }

    private static int countDistinct(int[][] subs) {
        Set<String> seen = new HashSet<>();
        for (int[] s : subs) { int[] c = s.clone(); Arrays.sort(c); seen.add(Arrays.toString(c)); }
        return seen.size();
    }

    private static int max(int[] a) { int m = 0; for (int x : a) m = Math.max(m, x); return m; }

    private static final class IdentitySel implements FeatureSelector {
        private final int[] sel;
        IdentitySel(int F) { sel = new int[F]; for (int i = 0; i < F; i++) sel[i] = i; }
        @Override public boolean isInitialized() { return true; }
        @Override public int[] getCurrentSelection() { return sel.clone(); }
        @Override public int[] getSelectedFeatures() { return sel.clone(); }
        @Override public int getNumFeatures() { return sel.length; }
        @Override public int getK() { return sel.length; }
        @Override public double[] filterInstance(double[] full) { return full.clone(); }
        @Override public void initialize(double[][] w, int[] y) { }
        @Override public void update(double[] f, int l, boolean a, Set<Integer> d) { }
        @Override public String name() { return "Identity"; }
    }
}
