package thesis.models;

import com.yahoo.labs.samoa.instances.Instance;
import lombok.Getter;
import moa.classifiers.Classifier;
import thesis.selection.FeatureSelector;

import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.Consumer;

public class DriftAwareSRP implements ModelWrapper {

    public enum SurgicalResetPolicy { NONE, FULL }

    @FunctionalInterface
    public interface ScoreProvider { double[] currentFeatureScores(); }

    public static final class DriftEvent {
        public final long instanceIdx;
        public final boolean alarm;
        public final Set<Integer> driftingFeatures;
        public final DriftActionSummary summary;
        public final double[] importanceSnapshot;
        public final double[] learnerWeightsSnapshot;
        public DriftEvent(long instanceIdx, boolean alarm, Set<Integer> df,
                          DriftActionSummary s, double[] imp, double[] lw) {
            this.instanceIdx = instanceIdx; this.alarm = alarm;
            this.driftingFeatures = df; this.summary = s;
            this.importanceSnapshot = imp; this.learnerWeightsSnapshot = lw;
        }
        @Override public String toString() {
            return "DriftEvent{i=" + instanceIdx + ", alarm=" + alarm
                    + ", drifting=" + driftingFeatures
                    + ", " + (summary == null ? "no-summary" : summary.toString()) + "}";
        }
    }

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
    private static final Map<Class<?>, Map<String, Field>> FIELD_CACHE = new HashMap<>();

    private final SRPWrapper srpWrapper;
    private final int origDim;
    @Getter private final double tau;
    private final Random rng;

    @Getter private FeatureImportance importance;
    @Getter private SurgicalResetPolicy surgicalResetPolicy = SurgicalResetPolicy.NONE;
    private ScoreProvider scoreProvider;
    private Consumer<DriftEvent> driftListener;

    @Getter private long handleDriftCalls;
    @Getter private long autoHandleDriftCalls;
    @Getter private long totalKept;
    @Getter private long totalSurgical;
    @Getter private long totalFull;
    @Getter private long totalNoReplacement;
    @Getter private long totalSwapsPerformed;
    @Getter private long refreshCalls;
    @Getter private long totalRefreshed;
    @Getter private long weightedPredictions;
    @Getter private long unweightedFallbacks;
    @Getter private long nonFiniteVoteSkips;
    @Getter private DriftActionSummary lastSummary;
    private double[] lastLearnerWeights;
    private long instanceCounter;

    public DriftAwareSRP(SRPWrapper srpWrapper) { this(srpWrapper, 0.5, 7L, null); }
    public DriftAwareSRP(SRPWrapper srpWrapper, double tau, long seed) { this(srpWrapper, tau, seed, null); }

    public DriftAwareSRP(SRPWrapper srpWrapper, double tau, long seed, FeatureImportance importance) {
        if (srpWrapper == null) throw new IllegalArgumentException("srpWrapper must not be null");
        if (srpWrapper.isUseHardFilter()) {
            throw new IllegalArgumentException(
                    "DriftAwareSRP requires SRPWrapper with useHardFilter=false");
        }
        if (!(tau > 0.0 && tau <= 1.0)) {
            throw new IllegalArgumentException("tau must be in (0,1], got " + tau);
        }
        this.srpWrapper = srpWrapper;
        this.origDim = srpWrapper.getFeatureSpace().numFeatures();
        if (importance != null && importance.getNumFeatures() != origDim) {
            throw new IllegalArgumentException("FeatureImportance dim mismatch");
        }
        this.tau = tau;
        this.rng = new Random(seed);
        this.importance = importance;
    }

    public void setFeatureImportance(FeatureImportance importance) {
        if (importance != null && importance.getNumFeatures() != origDim) {
            throw new IllegalArgumentException("FeatureImportance dim mismatch");
        }
        this.importance = importance;
    }

    public void setSurgicalResetPolicy(SurgicalResetPolicy p) {
        if (p == null) throw new IllegalArgumentException("policy must not be null");
        this.surgicalResetPolicy = p;
    }

    public void setScoreProvider(ScoreProvider sp) { this.scoreProvider = sp; }
    public void setDriftListener(Consumer<DriftEvent> l) { this.driftListener = l; }

    @Override
    public void train(Instance full, int classLabel) { train(full, classLabel, false, Set.of()); }

    @Override
    public void train(Instance full, int classLabel, boolean driftAlarm, Set<Integer> driftingFeatures) {
        srpWrapper.train(full, classLabel, driftAlarm, driftingFeatures);
        instanceCounter++;
        if (driftAlarm) {
            double[] scores = resolveScores();
            if (scores != null) {
                Set<Integer> df = (driftingFeatures == null) ? Set.of() : driftingFeatures;
                DriftActionSummary s = handleDrift(df, scores);
                autoHandleDriftCalls++;
                if (driftListener != null) {
                    driftListener.accept(new DriftEvent(instanceCounter, true, df, s,
                            importance == null ? null : importance.getImportance(),
                            lastLearnerWeights == null ? null : lastLearnerWeights.clone()));
                }
            }
        }
    }

    private double[] resolveScores() {
        if (scoreProvider != null) {
            double[] s = scoreProvider.currentFeatureScores();
            if (s != null && s.length == origDim) return s;
        }
        if (importance != null) {
            double[] mi = importance.getMIScores();
            if (mi != null && mi.length == origDim && hasAnyPositive(mi)) return mi;
            return importance.getImportance();
        }
        return null;
    }

    private static boolean hasAnyPositive(double[] x) {
        for (double v : x) if (v > 0.0 && Double.isFinite(v)) return true;
        return false;
    }

    @Override
    public int predict(Instance full) {
        double[] votes = predictProba(full);
        if (votes == null || votes.length == 0) return 0;
        int best = 0;
        for (int i = 1; i < votes.length; i++) if (votes[i] > votes[best]) best = i;
        return best;
    }

    @Override
    public double[] predictProba(Instance full) {
        if (importance == null) { unweightedFallbacks++; return srpWrapper.predictProba(full); }
        return predictProbaWeighted(full);
    }

    public double[] predictProbaWeighted(Instance full) {
        Object[] ensemble = readEnsembleArray();
        if (ensemble == null || ensemble.length == 0) {
            unweightedFallbacks++;
            return srpWrapper.predictProba(full);
        }
        double[] impOrig = importance.getImportance();
        if (impOrig == null || !isAllFinite(impOrig)) {
            unweightedFallbacks++;
            return srpWrapper.predictProba(full);
        }
        double[] weights = computeLearnerWeights(ensemble, impOrig);
        lastLearnerWeights = weights.clone();
        if (!isAllFinite(weights)) {
            unweightedFallbacks++;
            return srpWrapper.predictProba(full);
        }

        double cv = coefficientOfVariation(weights);
        if (cv < 0.10) {
            unweightedFallbacks++;
            return srpWrapper.predictProba(full);
        }

        int numClasses = srpWrapper.getReducedHeader().numClasses();
        if (numClasses < 2) {
            unweightedFallbacks++;
            return srpWrapper.predictProba(full);
        }

        double[] weightedAgg = new double[numClasses];
        boolean any = false;
        for (int li = 0; li < ensemble.length; li++) {
            if (!(weights[li] > 0.0)) continue;
            Classifier c = asClassifier(ensemble[li]);
            if (c == null) continue;
            double[] votes;
            try { votes = c.getVotesForInstance(full); }
            catch (Exception e) { continue; }
            if (votes == null || votes.length == 0) continue;
            if (!isAllFinite(votes)) { nonFiniteVoteSkips++; continue; }
            double vsum = 0.0;
            for (double vv : votes) vsum += vv;
            if (!(vsum > 0.0) || !Double.isFinite(vsum)) continue;
            int upTo = Math.min(votes.length, weightedAgg.length);
            for (int ci = 0; ci < upTo; ci++) weightedAgg[ci] += weights[li] * (votes[ci] / vsum);
            any = true;
        }
        if (!any) { unweightedFallbacks++; return srpWrapper.predictProba(full); }

        double wsum = 0.0;
        for (double v : weightedAgg) wsum += v;
        if (!Double.isFinite(wsum) || wsum <= 0.0) {
            unweightedFallbacks++;
            return srpWrapper.predictProba(full);
        }
        for (int i = 0; i < weightedAgg.length; i++) weightedAgg[i] /= wsum;

        double alpha = blendAlpha(cv);
        double[] srpProba = srpWrapper.predictProba(full);
        if (srpProba == null || srpProba.length == 0 || !isAllFinite(srpProba)) {
            weightedPredictions++;
            return weightedAgg;
        }
        double srpSum = 0.0;
        for (double v : srpProba) srpSum += v;
        if (!Double.isFinite(srpSum) || srpSum <= 0.0) {
            weightedPredictions++;
            return weightedAgg;
        }
        int len = Math.min(weightedAgg.length, srpProba.length);
        double[] blended = new double[weightedAgg.length];
        for (int i = 0; i < len; i++) {
            blended[i] = alpha * weightedAgg[i] + (1.0 - alpha) * (srpProba[i] / srpSum);
        }
        double bsum = 0.0;
        for (double v : blended) bsum += v;
        if (Double.isFinite(bsum) && bsum > 0.0) {
            for (int i = 0; i < blended.length; i++) blended[i] /= bsum;
        } else {
            weightedPredictions++;
            return weightedAgg;
        }
        weightedPredictions++;

        if (weightedPredictions <= 5 || weightedPredictions % 1000 == 0) {
            System.out.printf("[ABC-DBG] wPred=%d cv=%.3f alpha=%.2f impHead=%s wHead=%s%n",
                    weightedPredictions, cv, alpha,
                    Arrays.toString(Arrays.copyOf(impOrig, Math.min(5, impOrig.length))),
                    Arrays.toString(Arrays.copyOf(weights, Math.min(5, weights.length))));
        }

        return blended;
    }

    private static double coefficientOfVariation(double[] w) {
        if (w == null || w.length == 0) return 0.0;
        double sum = 0.0;
        int n = 0;
        for (double v : w) { if (Double.isFinite(v)) { sum += v; n++; } }
        if (n == 0) return 0.0;
        double mean = sum / n;
        if (!(mean > 0.0)) return 0.0;
        double sq = 0.0;
        for (double v : w) if (Double.isFinite(v)) { double d = v - mean; sq += d * d; }
        double std = Math.sqrt(sq / n);
        return std / mean;
    }

    private static double blendAlpha(double cv) {
        double lo = 0.10, hi = 0.40;
        if (cv <= lo) return 0.0;
        if (cv >= hi) return 1.0;
        return (cv - lo) / (hi - lo);
    }


    private static boolean isEffectivelyUniform(double[] w, double relTol) {
        double mn = Double.POSITIVE_INFINITY, mx = Double.NEGATIVE_INFINITY;
        int positive = 0;
        for (double v : w) {
            if (v > 0.0) { positive++; if (v < mn) mn = v; if (v > mx) mx = v; }
        }
        if (positive == 0) return true;
        if (mn <= 0.0) return false;
        return ((mx - mn) / mx) < relTol;
    }


    public DriftActionSummary handleDrift(Set<Integer> driftingFeaturesOriginal,
                                          double[] featureScoresOriginal) {
        if (driftingFeaturesOriginal == null) driftingFeaturesOriginal = Set.of();
        if (featureScoresOriginal == null)
            throw new IllegalArgumentException("featureScoresOriginal must not be null");
        if (featureScoresOriginal.length != origDim)
            throw new IllegalArgumentException("featureScoresOriginal length mismatch");

        boolean[] driftingMask = new boolean[origDim];
        for (int idx : driftingFeaturesOriginal) {
            if (idx >= 0 && idx < origDim) driftingMask[idx] = true;
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
                summary.record(li, DriftActionSummary.Action.KEEP, 0, 0, 0);
                continue;
            }
            int overlapCount = 0;
            for (int s : sub) if (s >= 0 && s < origDim && driftingMask[s]) overlapCount++;
            double frac = overlapCount / (double) sub.length;

            if (overlapCount == 0) {
                summary.record(li, DriftActionSummary.Action.KEEP, 0, sub.length, 0);
                continue;
            }

            if (frac < tau) {
                int[] result = surgicalReplace(sub, featureScoresOriginal, driftingMask);
                int swaps = countDifferences(result, sub);
                if (swaps == 0) {
                    summary.record(li, DriftActionSummary.Action.NO_REPLACEMENT,
                            overlapCount, sub.length, 0);
                    continue;
                }
                if (result.length != sub.length) {
                    throw new IllegalStateException("surgicalReplace changed subspace length: "
                            + sub.length + " -> " + result.length);
                }
                Arrays.sort(result);
                writeSubspaceSafely(ensemble[li], result);
                if (surgicalResetPolicy == SurgicalResetPolicy.FULL) {
                    resetLearner(ensemble[li]);
                }
                summary.record(li, DriftActionSummary.Action.SURGICAL,
                        overlapCount, sub.length, swaps);
            } else {
                Set<Integer> avoid = driftingFeaturesOriginal;
                if (origDim - driftingFeaturesOriginal.size() < sub.length) {
                    avoid = Set.of();
                }
                int[] newSub = generateSubspace(sub.length, avoid);
                if (newSub.length != sub.length) {
                    throw new IllegalStateException("generateSubspace changed length: "
                            + sub.length + " -> " + newSub.length);
                }
                Arrays.sort(newSub);
                writeSubspaceSafely(ensemble[li], newSub);
                resetLearner(ensemble[li]);
                summary.record(li, DriftActionSummary.Action.FULL,
                        overlapCount, sub.length, sub.length);
            }
        }

        handleDriftCalls++;
        totalKept          += summary.getKeptCount();
        totalSurgical      += summary.getSurgicalCount();
        totalFull          += summary.getFullCount();
        totalNoReplacement += summary.getNoReplacementCount();
        for (int s : summary.getSwapCounts()) totalSwapsPerformed += s;
        lastSummary = summary;
        return summary;
    }

    private static final String[] STALE_HEADER_FIELDS = {
            "subspaceInstances", "subspaceInstance",
            "headerSubspace", "subspaceHeader",
            "subSpaceInstances", "subSpaceInstance",
            "subspaceHeaderTemplate"
    };

    private static boolean writeSubspaceSafely(Object learner, int[] newSub) {
        boolean wrote = writeSubspace(learner, newSub);
        if (!wrote) return false;
        for (String stale : STALE_HEADER_FIELDS) {
            writeField(learner, stale, null);
        }
        return true;
    }

    public RefreshSummary refreshAllSubspaces() {
        Object[] ensemble = readEnsembleArray();
        if (ensemble == null || ensemble.length == 0) {
            refreshCalls++;
            return new RefreshSummary(0, 0);
        }
        int refreshed = 0;
        for (int li = 0; li < ensemble.length; li++) {
            int[] sub = readSubspace(ensemble[li]);
            if (sub.length == 0) continue;
            int[] newSub = generateSubspace(sub.length, Set.of());
            if (newSub.length != sub.length) {
                throw new IllegalStateException("generateSubspace changed length");
            }
            Arrays.sort(newSub);
            writeSubspaceSafely(ensemble[li], newSub);
            resetLearner(ensemble[li]);
            refreshed++;
        }
        refreshCalls++;
        totalRefreshed += refreshed;
        return new RefreshSummary(ensemble.length, refreshed);
    }

    private int[] surgicalReplace(int[] currentSub, double[] scoresOrig, boolean[] driftingMask) {
        boolean[] inSub = new boolean[origDim];
        for (int s : currentSub) if (s >= 0 && s < origDim) inSub[s] = true;

        int candCount = 0;
        for (int i = 0; i < origDim; i++) if (!inSub[i] && !driftingMask[i]) candCount++;
        if (candCount == 0) return currentSub.clone();

        int[] candidates = new int[candCount];
        int j = 0;
        for (int i = 0; i < origDim; i++) if (!inSub[i] && !driftingMask[i]) candidates[j++] = i;
        sortIndicesByScoreDesc(candidates, scoresOrig);

        int[] driftingPositions = new int[currentSub.length];
        int dpCount = 0;
        for (int i = 0; i < currentSub.length; i++) {
            int s = currentSub[i];
            if (s >= 0 && s < origDim && driftingMask[s]) driftingPositions[dpCount++] = i;
        }
        int[] sortedDriftPos = Arrays.copyOf(driftingPositions, dpCount);
        sortPositionsByCurrentScoreAsc(sortedDriftPos, currentSub, scoresOrig);

        int[] result = currentSub.clone();
        int candIdx = 0;
        for (int p = 0; p < sortedDriftPos.length && candIdx < candidates.length; p++) {
            int pos = sortedDriftPos[p];
            int replacement = candidates[candIdx];
            if (scoresOrig[replacement] > scoresOrig[result[pos]]) {
                result[pos] = replacement;
                candIdx++;
            } else {
                break;
            }
        }
        return result;
    }

    private int[] generateSubspace(int size, Set<Integer> avoid) {
        int[] s;
        if (importance != null) {
            s = WeightedSubspaceSampler.sample(importance.getImportance(), size, rng, avoid);
        } else {
            s = uniformRandomSubspace(size, avoid);
        }
        if (s.length != size) {
            throw new IllegalStateException("subspace size mismatch: got " + s.length
                    + " expected " + size);
        }
        return s;
    }

    private int[] uniformRandomSubspace(int size, Set<Integer> avoid) {
        boolean[] inPool = new boolean[origDim];
        int poolCount = 0;
        for (int i = 0; i < origDim; i++) {
            if (!avoid.contains(i)) { inPool[i] = true; poolCount++; }
        }
        if (poolCount < size) {
            for (int i = 0; i < origDim && poolCount < size; i++) {
                if (!inPool[i]) { inPool[i] = true; poolCount++; }
            }
        }
        int[] pool = new int[poolCount];
        int j = 0;
        for (int i = 0; i < origDim; i++) if (inPool[i]) pool[j++] = i;
        for (int i = pool.length - 1; i > 0; i--) {
            int k = rng.nextInt(i + 1);
            int tmp = pool[i]; pool[i] = pool[k]; pool[k] = tmp;
        }
        int[] out = Arrays.copyOf(pool, size);
        Arrays.sort(out);
        return out;
    }

    private double[] computeLearnerWeights(Object[] ensemble, double[] impOrig) {
        double[] raw = new double[ensemble.length];
        int nonZero = 0;
        for (int li = 0; li < ensemble.length; li++) {
            int[] sub = readSubspace(ensemble[li]);
            if (sub.length == 0) { raw[li] = 0.0; continue; }
            double w = 0.0;
            int counted = 0;
            for (int s : sub) {
                if (s < 0 || s >= origDim) continue;
                w += impOrig[s];
                counted++;
            }
            raw[li] = (counted == 0) ? 0.0 : w / counted;
            if (raw[li] > 0.0) nonZero++;
        }
        double[] weights = new double[ensemble.length];
        if (nonZero == 0) {
            Arrays.fill(weights, 1.0 / ensemble.length);
            return weights;
        }
        double temperature = 2.0;
        double smoothedSum = 0.0;
        for (int li = 0; li < ensemble.length; li++) {
            weights[li] = (raw[li] > 0.0) ? Math.pow(raw[li], 1.0 / temperature) : 0.0;
            smoothedSum += weights[li];
        }
        if (!Double.isFinite(smoothedSum) || smoothedSum <= 0.0) {
            Arrays.fill(weights, 1.0 / ensemble.length);
            return weights;
        }
        for (int li = 0; li < ensemble.length; li++) weights[li] /= smoothedSum;
        double uniformBlend = 0.7;
        double uniform = 1.0 / ensemble.length;
        double finalSum = 0.0;
        for (int li = 0; li < ensemble.length; li++) {
            weights[li] = uniformBlend * uniform + (1.0 - uniformBlend) * weights[li];
            finalSum += weights[li];
        }
        if (Double.isFinite(finalSum) && finalSum > 0.0) {
            for (int li = 0; li < ensemble.length; li++) weights[li] /= finalSum;
        } else {
            Arrays.fill(weights, 1.0 / ensemble.length);
        }
        return weights;
    }



    private static int countDifferences(int[] a, int[] b) {
        int d = 0; for (int i = 0; i < a.length; i++) if (a[i] != b[i]) d++; return d;
    }
    private static boolean isAllFinite(double[] v) {
        for (double x : v) if (!Double.isFinite(x)) return false; return true;
    }
    private static void sortIndicesByScoreDesc(int[] idx, double[] scores) {
        for (int i = 1; i < idx.length; i++) {
            int cur = idx[i]; double curScore = scores[cur]; int j = i - 1;
            while (j >= 0 && scores[idx[j]] < curScore) { idx[j + 1] = idx[j]; j--; }
            idx[j + 1] = cur;
        }
    }
    private static void sortPositionsByCurrentScoreAsc(int[] positions, int[] currentSub, double[] scores) {
        for (int i = 1; i < positions.length; i++) {
            int curPos = positions[i]; double curScore = scores[currentSub[curPos]]; int j = i - 1;
            while (j >= 0 && scores[currentSub[positions[j]]] > curScore) {
                positions[j + 1] = positions[j]; j--;
            }
            positions[j + 1] = curPos;
        }
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
        Object srp = extractInnerSRP(srpWrapper);
        if (srp == null) return null;
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
    private static Object extractInnerSRP(SRPWrapper wrapper) {
        Class<?> c = wrapper.getClass();
        while (c != null && c != Object.class) {
            for (Field f : c.getDeclaredFields()) {
                if (moa.classifiers.meta.StreamingRandomPatches.class.isAssignableFrom(f.getType())) {
                    try { f.setAccessible(true); return f.get(wrapper); }
                    catch (IllegalAccessException e) { return null; }
                }
            }
            c = c.getSuperclass();
        }
        return null;
    }
    private int[] readSubspace(Object learner) {
        if (learner == null) return new int[0];
        for (String name : SUBSPACE_FIELD_CANDIDATES) {
            Object v = readField(learner, name);
            if (v instanceof int[]) {
                int[] raw = (int[]) v;
                if (raw.length == 0) continue;
                return raw.clone();
            }
            if (v instanceof Integer[]) {
                Integer[] boxed = (Integer[]) v;
                if (boxed.length == 0) continue;
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
        try {
            Method m = learner.getClass().getMethod("resetLearning");
            m.invoke(learner);
            return;
        } catch (Exception ignored) { }
        for (String name : CLASSIFIER_FIELD_CANDIDATES) {
            Object inner = readField(learner, name);
            if (inner instanceof Classifier) {
                ((Classifier) inner).resetLearning();
                return;
            }
        }
    }
    private static Field lookupField(Class<?> rootClass, String name) {
        Map<String, Field> perClass;
        synchronized (FIELD_CACHE) {
            perClass = FIELD_CACHE.computeIfAbsent(rootClass, k -> new HashMap<>());
            if (perClass.containsKey(name)) return perClass.get(name);
        }
        Field found = null;
        Class<?> c = rootClass;
        while (c != null && c != Object.class) {
            try { Field f = c.getDeclaredField(name); f.setAccessible(true); found = f; break; }
            catch (NoSuchFieldException e) { c = c.getSuperclass(); }
        }
        synchronized (FIELD_CACHE) { perClass.put(name, found); }
        return found;
    }
    private static Object readField(Object obj, String name) {
        Field f = lookupField(obj.getClass(), name);
        if (f == null) return null;
        try { return f.get(obj); } catch (IllegalAccessException e) { return null; }
    }
    private static boolean writeField(Object obj, String name, Object value) {
        Field f = lookupField(obj.getClass(), name);
        if (f == null) return false;
        try { f.set(obj, value); return true; } catch (IllegalAccessException e) { return false; }
    }

    @Override public FeatureSelector getSelector() { return srpWrapper.getSelector(); }
    @Override public int[] getCurrentSelection()   { return srpWrapper.getCurrentSelection(); }

    @Override
    public void reset() {
        srpWrapper.reset();
        if (importance != null) importance.resetToUniform();
        handleDriftCalls = autoHandleDriftCalls = totalKept = totalSurgical = totalFull = totalNoReplacement = 0;
        totalSwapsPerformed = refreshCalls = totalRefreshed = 0;
        weightedPredictions = unweightedFallbacks = nonFiniteVoteSkips = 0;
        lastSummary = null;
        lastLearnerWeights = null;
        instanceCounter = 0;
    }

    public SRPWrapper getSRPWrapper() { return srpWrapper; }
    public double[] getLastLearnerWeights() {
        return lastLearnerWeights == null ? new double[0] : lastLearnerWeights.clone();
    }

    public int[][] getCurrentSubspaces() {
        Object[] ens = readEnsembleArray();
        if (ens == null) return new int[0][];
        int[][] out = new int[ens.length][];
        for (int i = 0; i < ens.length; i++) {
            int[] raw = readSubspace(ens[i]);
            int validCount = 0;
            for (int x : raw) if (x >= 0 && x < origDim) validCount++;
            if (validCount == raw.length) {
                out[i] = raw;
            } else {
                int[] clean = new int[validCount];
                int j = 0;
                for (int x : raw) if (x >= 0 && x < origDim) clean[j++] = x;
                out[i] = clean;
            }
        }
        return out;
    }

    @Override
    public String name() {
        return "DriftAwareSRP[fullDim=" + origDim + ", tau=" + tau
                + ", surgicalReset=" + surgicalResetPolicy
                + (importance == null ? ", uniform" : ", importance")
                + ", weighted-vote] + " + srpWrapper.name();
    }

    public static final class RefreshSummary {
        @Getter private final int ensembleSize;
        @Getter private final int refreshedCount;
        public RefreshSummary(int ensembleSize, int refreshedCount) {
            this.ensembleSize = ensembleSize;
            this.refreshedCount = refreshedCount;
        }
        @Override public String toString() {
            return "RefreshSummary{ensemble=" + ensembleSize + ", refreshed=" + refreshedCount + "}";
        }
    }
}