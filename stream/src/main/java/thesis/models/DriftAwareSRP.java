package thesis.models;

import com.yahoo.labs.samoa.instances.Instance;
import lombok.Getter;
import moa.classifiers.Classifier;
import thesis.selection.FeatureSelector;

import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class DriftAwareSRP  implements ModelWrapper {

    private static final String[] ENSEMBLE_FIELD_CANDIDATES = {
            "ensemble", "learners", "baseLearners", "classifiers"
    };
    private static final String[] SUBSPACE_FIELD_CANDIDATES = {
            "featureIndexes",
            "subSpaceIndexes", "subspaceIndexes", "subSpace", "subspace",
            "indices", "subSpaceIndices", "subspaceIndices"
    };
    private static final String[] CLASSIFIER_FIELD_CANDIDATES = {
            "classifier", "learner", "baseLearner", "model"
    };

    @Override
    public void train(Instance full, int classLabel) {
        train(full, classLabel, false, Set.of());
    }

    @Override
    public FeatureSelector getSelector()  { return srpWrapper.getSelector(); }

    @Override
    public int[] getCurrentSelection()    { return srpWrapper.getCurrentSelection(); }

    @Override
    public void reset()                   { srpWrapper.reset(); }

    private final SRPWrapper srpWrapper;
    private final double tau;
    private final Random rng;

    private FeatureImportance importance;

    private long handleDriftCalls;
    private long totalKept;
    private long totalSurgical;
    private long totalFull;
    private long totalNoReplacement;
    private long refreshCalls;
    private long totalRefreshed;
    private long weightedPredictions;
    private long unweightedFallbacks;
    private DriftActionSummary lastSummary;
    private double[] lastLearnerWeights;

    public DriftAwareSRP(SRPWrapper srpWrapper) {
        this(srpWrapper, 0.5, 7L, null);
    }

    public DriftAwareSRP(SRPWrapper srpWrapper, double tau, long seed) {
        this(srpWrapper, tau, seed, null);
    }

    public DriftAwareSRP(SRPWrapper srpWrapper, double tau, long seed,
                         FeatureImportance importance) {
        if (srpWrapper == null) throw new IllegalArgumentException("srpWrapper must not be null");
        if (!(tau > 0.0 && tau <= 1.0)) {
            throw new IllegalArgumentException("tau must be in (0,1], got " + tau);
        }
        this.srpWrapper = srpWrapper;
        this.tau = tau;
        this.rng = new Random(seed);
        this.importance = importance;
    }

    public void setFeatureImportance(FeatureImportance importance) { this.importance = importance; }
    public FeatureImportance getFeatureImportance()                 { return importance; }

    public void train(Instance full, int classLabel,
                      boolean driftAlarm, Set<Integer> driftingFeatures) {
        srpWrapper.train(full, classLabel, driftAlarm, driftingFeatures);
    }

    public int predict(Instance full) {
        double[] votes = predictProba(full);
        int best = 0;
        for (int i = 1; i < votes.length; i++) if (votes[i] > votes[best]) best = i;
        return best;
    }

    public double[] predictProba(Instance full) {
        if (importance == null) {
            unweightedFallbacks++;
            return srpWrapper.predictProba(full);
        }
        return predictProbaWeighted(full);
    }

    public double[] predictProbaWeighted(Instance full) {
        Object[] ensemble = readEnsembleArray();
        if (ensemble == null || ensemble.length == 0) {
            unweightedFallbacks++;
            return srpWrapper.predictProba(full);
        }

        int[] selection = srpWrapper.getCurrentSelection();
        double[] impOrig = (importance != null) ? importance.getImportance() : null;
        double[] weights = computeLearnerWeights(ensemble, selection, impOrig);
        lastLearnerWeights = weights.clone();

        Instance reduced = srpWrapper.buildFilteredInstance(full);
        int numClasses = srpWrapper.getReducedHeader().numClasses();
        double[] aggregated = new double[Math.max(numClasses, 2)];
        boolean any = false;

        for (int li = 0; li < ensemble.length; li++) {
            if (!(weights[li] > 0.0)) continue;
            Classifier c = asClassifier(ensemble[li]);
            if (c == null) continue;
            double[] votes;
            try {
                votes = c.getVotesForInstance(reduced);
            } catch (Exception e) {
                continue;
            }
            if (votes == null || votes.length == 0) continue;
            if (votes.length > aggregated.length) {
                double[] grown = new double[votes.length];
                System.arraycopy(aggregated, 0, grown, 0, aggregated.length);
                aggregated = grown;
            }
            for (int ci = 0; ci < votes.length; ci++) {
                aggregated[ci] += weights[li] * votes[ci];
            }
            any = true;
        }

        if (!any) {
            unweightedFallbacks++;
            return srpWrapper.predictProba(full);
        }

        double sum = 0.0;
        for (double v : aggregated) sum += v;
        if (sum > 0.0) {
            for (int i = 0; i < aggregated.length; i++) aggregated[i] /= sum;
        }
        weightedPredictions++;
        return aggregated;
    }

    public DriftActionSummary handleDrift(Set<Integer> driftingFeaturesOriginal,
                                          double[] featureScoresOriginal) {
        if (driftingFeaturesOriginal == null) driftingFeaturesOriginal = Set.of();
        if (featureScoresOriginal == null) {
            throw new IllegalArgumentException("featureScoresOriginal must not be null");
        }

        int[] selection = srpWrapper.getCurrentSelection();
        int reducedDim = selection.length;

        Set<Integer> driftingReduced = new HashSet<>();
        for (int r = 0; r < reducedDim; r++) {
            if (driftingFeaturesOriginal.contains(selection[r])) driftingReduced.add(r);
        }

        double[] scoresReduced = new double[reducedDim];
        for (int r = 0; r < reducedDim; r++) {
            int origIdx = selection[r];
            scoresReduced[r] = (origIdx < featureScoresOriginal.length)
                    ? featureScoresOriginal[origIdx] : 0.0;
        }

        Object[] ensemble = readEnsembleArray();
        if (ensemble == null || ensemble.length == 0) {
            handleDriftCalls++;
            lastSummary = new DriftActionSummary(0);
            return lastSummary;
        }

        DriftActionSummary summary = new DriftActionSummary(ensemble.length);
        for (int li = 0; li < ensemble.length; li++) {
            int[] sub = readSubspace(ensemble[li]);
            if (sub.length == 0) {
                summary.record(li, DriftActionSummary.Action.KEEP, 0, 0);
                continue;
            }

            Set<Integer> overlap = new HashSet<>();
            for (int s : sub) if (driftingReduced.contains(s)) overlap.add(s);
            int overlapCount = overlap.size();
            double frac = overlapCount / (double) sub.length;

            if (overlapCount == 0) {
                summary.record(li, DriftActionSummary.Action.KEEP, 0, sub.length);
            } else if (frac < tau) {
                int[] newSub = surgicalReplace(sub, overlap, scoresReduced,
                        driftingReduced, reducedDim);
                if (Arrays.equals(newSub, sub)) {
                    summary.record(li, DriftActionSummary.Action.NO_REPLACEMENT,
                            overlapCount, sub.length);
                } else {
                    writeSubspace(ensemble[li], newSub);
                    summary.record(li, DriftActionSummary.Action.SURGICAL,
                            overlapCount, sub.length);
                }
            } else {
                int[] newSub = generateSubspace(sub.length, reducedDim, driftingReduced);
                writeSubspace(ensemble[li], newSub);
                resetLearner(ensemble[li]);
                summary.record(li, DriftActionSummary.Action.FULL,
                        overlapCount, sub.length);
            }
        }

        handleDriftCalls++;
        totalKept           += summary.getKeptCount();
        totalSurgical       += summary.getSurgicalCount();
        totalFull           += summary.getFullCount();
        totalNoReplacement  += summary.getNoReplacementCount();
        lastSummary = summary;
        return summary;
    }

    public RefreshSummary refreshAllSubspaces() {
        Object[] ensemble = readEnsembleArray();
        if (ensemble == null || ensemble.length == 0) {
            refreshCalls++;
            return new RefreshSummary(0, 0);
        }
        int[] selection = srpWrapper.getCurrentSelection();
        int reducedDim = selection.length;
        int refreshed = 0;
        for (int li = 0; li < ensemble.length; li++) {
            int[] sub = readSubspace(ensemble[li]);
            if (sub.length == 0) continue;
            int[] newSub = generateSubspace(sub.length, reducedDim, Set.of());
            writeSubspace(ensemble[li], newSub);
            resetLearner(ensemble[li]);
            refreshed++;
        }
        refreshCalls++;
        totalRefreshed += refreshed;
        return new RefreshSummary(ensemble.length, refreshed);
    }

    private int[] surgicalReplace(int[] currentSub, Set<Integer> driftingInSub,
                                  double[] scoresReduced, Set<Integer> driftingReduced,
                                  int reducedDim) {
        Set<Integer> currentSet = new HashSet<>();
        for (int s : currentSub) currentSet.add(s);

        List<Integer> candidates = new ArrayList<>();
        for (int i = 0; i < reducedDim; i++) {
            if (currentSet.contains(i)) continue;
            if (driftingReduced.contains(i)) continue;
            candidates.add(i);
        }
        candidates.sort((a, b) -> Double.compare(scoresReduced[b], scoresReduced[a]));

        int[] result = currentSub.clone();
        int candIdx = 0;
        for (int i = 0; i < result.length; i++) {
            if (driftingInSub.contains(result[i]) && candIdx < candidates.size()) {
                int replacement = candidates.get(candIdx++);
                if (scoresReduced[replacement] > scoresReduced[result[i]]) {
                    result[i] = replacement;
                }
            }
        }
        return result;
    }

    private int[] generateSubspace(int size, int reducedDim, Set<Integer> avoid) {
        int[] selection = srpWrapper.getCurrentSelection();
        if (importance != null && selection.length == reducedDim) {
            double[] weightsReduced = importance.projectToReduced(selection);
            return WeightedSubspaceSampler.sample(weightsReduced, size, rng, avoid);
        }
        return uniformRandomSubspace(size, reducedDim, avoid);
    }

    private int[] uniformRandomSubspace(int size, int reducedDim, Set<Integer> avoid) {
        List<Integer> pool = new ArrayList<>();
        for (int i = 0; i < reducedDim; i++) {
            if (!avoid.contains(i)) pool.add(i);
        }
        if (pool.size() < size) {
            for (int i = 0; i < reducedDim && pool.size() < size; i++) {
                if (!pool.contains(i)) pool.add(i);
            }
        }
        for (int i = pool.size() - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = pool.get(i); pool.set(i, pool.get(j)); pool.set(j, tmp);
        }
        int[] out = new int[size];
        for (int i = 0; i < size; i++) out[i] = pool.get(i);
        Arrays.sort(out);
        return out;
    }

    private double[] computeLearnerWeights(Object[] ensemble, int[] selection, double[] impOrig) {
        double[] weights = new double[ensemble.length];
        double sum = 0.0;
        for (int li = 0; li < ensemble.length; li++) {
            int[] sub = readSubspace(ensemble[li]);
            if (sub.length == 0) { weights[li] = 0.0; continue; }
            double w = 0.0;
            for (int s : sub) {
                if (s < 0 || s >= selection.length) continue;
                int orig = selection[s];
                double imp = (impOrig != null && orig >= 0 && orig < impOrig.length) ? impOrig[orig] : 1.0;
                w += imp;
            }
            weights[li] = w / sub.length;
            sum += weights[li];
        }
        if (sum <= 0.0) {
            double uniform = 1.0 / ensemble.length;
            Arrays.fill(weights, uniform);
        } else {
            for (int li = 0; li < ensemble.length; li++) weights[li] /= sum;
        }
        return weights;
    }

    private static Classifier asClassifier(Object learner) {
        if (learner == null) return null;
        if (learner instanceof Classifier) return (Classifier) learner;
        for (String name : CLASSIFIER_FIELD_CANDIDATES) {
            Object v = readField(learner, name);
            if (v instanceof Classifier) return (Classifier) v;
        }
        return null;
    }

    private Object[] readEnsembleArray() {
        Object srp = srpWrapper.getSRP();
        for (String name : ENSEMBLE_FIELD_CANDIDATES) {
            Object v = readField(srp, name);
            if (v == null) continue;
            if (v.getClass().isArray()) {
                int len = Array.getLength(v);
                Object[] out = new Object[len];
                for (int i = 0; i < len; i++) out[i] = Array.get(v, i);
                return out;
            }
        }
        return null;
    }

    private static int[] readSubspace(Object learner) {
        if (learner == null) return new int[0];
        for (String name : SUBSPACE_FIELD_CANDIDATES) {
            Object v = readField(learner, name);
            if (v instanceof int[]) return (int[]) v;
            if (v instanceof Integer[]) {
                Integer[] boxed = (Integer[]) v;
                int[] out = new int[boxed.length];
                for (int i = 0; i < boxed.length; i++) out[i] = boxed[i];
                return out;
            }
        }
        return new int[0];
    }

    private static boolean writeSubspace(Object learner, int[] newSub) {
        if (learner == null) return false;
        for (String name : SUBSPACE_FIELD_CANDIDATES) {
            if (writeField(learner, name, newSub)) return true;
        }
        return false;
    }

    private static void resetLearner(Object learner) {
        if (learner == null) return;
        if (learner instanceof Classifier) {
            ((Classifier) learner).resetLearning();
            return;
        }
        for (String name : CLASSIFIER_FIELD_CANDIDATES) {
            Object inner = readField(learner, name);
            if (inner instanceof Classifier) {
                ((Classifier) inner).resetLearning();
                return;
            }
        }
        try {
            Method m = learner.getClass().getMethod("resetLearning");
            m.invoke(learner);
        } catch (Exception ignored) { }
    }

    private static Object readField(Object obj, String name) {
        Class<?> c = obj.getClass();
        while (c != null && c != Object.class) {
            try {
                Field f = c.getDeclaredField(name);
                f.setAccessible(true);
                return f.get(obj);
            } catch (NoSuchFieldException e) {
                c = c.getSuperclass();
            } catch (IllegalAccessException e) {
                return null;
            }
        }
        return null;
    }

    private static boolean writeField(Object obj, String name, Object value) {
        Class<?> c = obj.getClass();
        while (c != null && c != Object.class) {
            try {
                Field f = c.getDeclaredField(name);
                f.setAccessible(true);
                f.set(obj, value);
                return true;
            } catch (NoSuchFieldException e) {
                c = c.getSuperclass();
            } catch (IllegalAccessException e) {
                return false;
            }
        }
        return false;
    }

    public SRPWrapper getSRPWrapper()       { return srpWrapper; }
    public double getTau()                  { return tau; }
    public long getHandleDriftCalls()       { return handleDriftCalls; }
    public long getTotalKept()              { return totalKept; }
    public long getTotalSurgical()          { return totalSurgical; }
    public long getTotalFull()              { return totalFull; }
    public long getTotalNoReplacement()     { return totalNoReplacement; }
    public long getRefreshCalls()           { return refreshCalls; }
    public long getTotalRefreshed()         { return totalRefreshed; }
    public long getWeightedPredictions()    { return weightedPredictions; }
    public long getUnweightedFallbacks()    { return unweightedFallbacks; }
    public double[] getLastLearnerWeights() {
        return lastLearnerWeights == null ? new double[0] : lastLearnerWeights.clone();
    }
    public DriftActionSummary getLastSummary() { return lastSummary; }

    public String name() {
        return "DriftAwareSRP(tau=" + tau +
                (importance == null ? ", uniform" : ", importance(w1=" + importance.getW1() +
                        ",w2=" + importance.getW2() + ")") +
                ", weighted-vote) + " + srpWrapper.name();
    }

    public static final class RefreshSummary {
        public final int ensembleSize;
        public final int refreshedCount;
        public RefreshSummary(int ensembleSize, int refreshedCount) {
            this.ensembleSize = ensembleSize;
            this.refreshedCount = refreshedCount;
        }
        @Override public String toString() {
            return "RefreshSummary{ensemble=" + ensembleSize +
                    ", refreshed=" + refreshedCount + "}";
        }
    }
}