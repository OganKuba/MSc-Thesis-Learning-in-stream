package thesis.models;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import lombok.Getter;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.classifiers.trees.ARFHoeffdingTree;
import thesis.selection.FeatureSelector;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * Drift-Adaptive Adaptive Random Forest (DA-ARF).
 *
 * <p>Custom ensemble of {@link ARFHoeffdingTree} base learners with three modifications:
 * <ul>
 *   <li><b>Component A — drift adaptation.</b> Each learner owns a per-tree ADWIN that watches its
 *       training error: on <i>warning</i> a background learner is spawned with a freshly resampled
 *       importance-weighted subspace; on <i>drift</i> the background replaces the foreground (or
 *       full reset if no background exists). External {@code train(..., driftAlarm, drifting)}
 *       calls trigger a conservative KEEP/FULL pass: only learners whose subspace overlaps
 *       low-importance drifting features are eligible for reset, and the number of resets per
 *       alarm is capped; high-importance drifting features are treated as unstable but still
 *       potentially predictive.</li>
 *   <li><b>Component B — importance-weighted sampling.</b> Per-learner subspaces are drawn from
 *       {@link WeightedSubspaceSampler} using a {@link FeatureImportance} pool (sharpened by
 *       {@code importancePower} and blended toward uniform by {@code samplingBeta}). When no
 *       importance is provided the sampler falls back to uniform.</li>
 *   <li><b>Component C — top-K rank-weighted voting.</b> {@link #predictProba} ranks learners by
 *       recent sliding-window accuracy and aggregates the top {@code ceil(topKFraction · N)}
 *       learners with descending integer weights {@code w_r = (K - r)}.</li>
 * </ul>
 *
 * <p>The wrapper is feature-pool aware: the supplied {@link FeatureSelector} is kept for
 * interface compatibility but the model itself routes the FULL feature space — each base learner
 * applies its own filtered header. This mirrors the {@code useHardFilter=false} contract of
 * {@link DriftAwareSRP}.
 */
public class DAARFWrapper implements ModelWrapper {

    @Getter private final FeatureSelector selector;
    private final FeatureSpace space;
    private final int origDim;
    private final int numClasses;
    @Getter private final int ensembleSize;
    @Getter private final int subspaceSize;
    @Getter private final double lambda;
    @Getter private final int accWindow;
    @Getter private final double topKFraction;
    @Getter private final double importancePower;
    @Getter private final double samplingBeta;
    @Getter private final boolean useBackgroundLearner;
    @Getter private final double warningDelta;
    @Getter private final double driftDelta;
    @Getter private double unstableImportanceQuantile = 0.50;
    @Getter private double externalResetFraction = 0.20;

    /** How the external (detector-driven) KEEP/FULL pass replaces an eligible learner. */
    public enum ExternalActionMode {
        /** Original behaviour: rebuild the learner from scratch (fresh subspace + tree). */
        RESET,
        /** Surgical: swap only the drifting features in the subspace, keep the trained tree. */
        SURGICAL
    }

    @Getter private ExternalActionMode externalActionMode = ExternalActionMode.RESET;
    /** A4: when false, the intrinsic per-tree ADWIN drift/warning channel is disabled entirely. */
    @Getter private boolean intrinsicDriftEnabled = true;
    /** A3: when true, the external pass skips learners that already have a pending background. */
    @Getter private boolean gateExternalOnPendingBackground = false;
    /** Score tolerance for accepting a surgical replacement (mirrors DriftAwareSRP). */
    @Getter private double surgicalReplacementTolerance = 0.95;

    // A7: base-tree hyperparameters. Defaults match MOA AdaptiveRandomForest's tree config
    // ("ARFHoeffdingTree -e 2000000 -g 50 -c 0.01"). The old code left HoeffdingTree defaults
    // (grace=200, confidence=1e-7), producing shallow trees that collapsed to majority on
    // imbalanced streams and handicapped DA-ARF vs the ARF baseline it is compared against.
    @Getter private int treeGracePeriod = 50;
    @Getter private double treeSplitConfidence = 0.01;

    private final Random rng;
    private FeatureImportance importance;

    private BaseLearner[] ensemble;
    private long instancesSeen;

    @Getter private long extKeepCount;
    @Getter private long extFullCount;
    @Getter private long extSurgicalCount;        // A2: external surgical swaps performed
    @Getter private long extNoReplacementCount;   // A2: surgical pass found no acceptable swap
    @Getter private long extGatedSkipCount;       // A3: external resets suppressed by gating
    @Getter private long bkgPromotions;           // intrinsic: background promoted to foreground
    @Getter private long intrinsicFullResetCount; // intrinsic: drift w/o background -> full reset
    @Getter private long warningCount;
    @Getter private long driftCount;

    public DAARFWrapper(FeatureSelector selector, InstancesHeader fullHeader,
                        int numClasses, int ensembleSize, int subspaceSize,
                        double lambda, int accWindow, double topKFraction,
                        double importancePower, double samplingBeta,
                        boolean useBackgroundLearner,
                        double warningDelta, double driftDelta,
                        long seed, FeatureImportance importance) {
        if (selector == null) throw new IllegalArgumentException("selector must not be null");
        if (fullHeader == null) throw new IllegalArgumentException("fullHeader must not be null");
        if (!selector.isInitialized())
            throw new IllegalArgumentException("selector must be initialized before wrapping");
        if (numClasses < 2) throw new IllegalArgumentException("numClasses must be >= 2");
        if (ensembleSize < 1) throw new IllegalArgumentException("ensembleSize must be >= 1");
        if (subspaceSize < 1) throw new IllegalArgumentException("subspaceSize must be >= 1");
        if (lambda <= 0.0) throw new IllegalArgumentException("lambda must be > 0");
        if (accWindow < 1) throw new IllegalArgumentException("accWindow must be >= 1");
        if (!(topKFraction > 0.0 && topKFraction <= 1.0))
            throw new IllegalArgumentException("topKFraction must be in (0,1]");
        if (!(importancePower > 0.0)) throw new IllegalArgumentException("importancePower must be > 0");
        if (!(samplingBeta >= 0.0 && samplingBeta <= 1.0))
            throw new IllegalArgumentException("samplingBeta must be in [0,1]");
        if (!(warningDelta > 0.0 && warningDelta < 1.0))
            throw new IllegalArgumentException("warningDelta must be in (0,1)");
        if (!(driftDelta > 0.0 && driftDelta < 1.0))
            throw new IllegalArgumentException("driftDelta must be in (0,1)");
        if (driftDelta > warningDelta)
            throw new IllegalArgumentException("driftDelta must be <= warningDelta");

        this.selector = selector;
        this.space = new FeatureSpace(fullHeader);
        this.origDim = space.numFeatures();
        if (subspaceSize > origDim)
            throw new IllegalArgumentException("subspaceSize > origDim");
        if (importance != null && importance.getNumFeatures() != origDim)
            throw new IllegalArgumentException("FeatureImportance dim mismatch");
        this.numClasses = numClasses;
        this.ensembleSize = ensembleSize;
        this.subspaceSize = subspaceSize;
        this.lambda = lambda;
        this.accWindow = accWindow;
        this.topKFraction = topKFraction;
        this.importancePower = importancePower;
        this.samplingBeta = samplingBeta;
        this.useBackgroundLearner = useBackgroundLearner;
        this.warningDelta = warningDelta;
        this.driftDelta = driftDelta;
        this.rng = new Random(seed);
        this.importance = importance;
        buildEnsemble();
    }

    public static DAARFWrapper defaults(FeatureSelector selector, InstancesHeader header,
                                        int numClasses, long seed, FeatureImportance importance) {
        int origDim = header.numAttributes() - 1;
        int sub = Math.max(2, (int) Math.ceil(Math.sqrt(origDim)));
        return new DAARFWrapper(selector, header, numClasses,
                /*ensemble=*/10, /*subspace=*/sub, /*lambda=*/6.0,
                /*accWindow=*/1000, /*topK=*/0.5,
                /*power=*/2.0, /*beta=*/0.7,
                /*useBkg=*/true, /*warnDelta=*/1e-4, /*driftDelta=*/1e-5,
                seed, importance);
    }

    public void setFeatureImportance(FeatureImportance imp) {
        if (imp != null && imp.getNumFeatures() != origDim)
            throw new IllegalArgumentException("FeatureImportance dim mismatch");
        this.importance = imp;
    }

    public FeatureImportance getFeatureImportance() { return importance; }

    public void setUnstableImportanceQuantile(double quantile) {
        if (!(quantile >= 0.0 && quantile <= 1.0) || !Double.isFinite(quantile))
            throw new IllegalArgumentException("unstableImportanceQuantile must be in [0, 1]");
        this.unstableImportanceQuantile = quantile;
    }

    public void setExternalResetFraction(double fraction) {
        if (!(fraction >= 0.0 && fraction <= 1.0) || !Double.isFinite(fraction))
            throw new IllegalArgumentException("externalResetFraction must be in [0, 1]");
        this.externalResetFraction = fraction;
    }

    public void setExternalActionMode(ExternalActionMode mode) {
        if (mode == null) throw new IllegalArgumentException("externalActionMode must not be null");
        this.externalActionMode = mode;
    }

    public void setIntrinsicDriftEnabled(boolean enabled) {
        this.intrinsicDriftEnabled = enabled;
    }

    public void setGateExternalOnPendingBackground(boolean gate) {
        this.gateExternalOnPendingBackground = gate;
    }

    public void setSurgicalReplacementTolerance(double tolerance) {
        if (!(tolerance >= 0.0 && tolerance <= 1.0) || !Double.isFinite(tolerance))
            throw new IllegalArgumentException("surgicalReplacementTolerance must be in [0, 1]");
        this.surgicalReplacementTolerance = tolerance;
    }

    /**
     * A7: override base-tree split hyperparameters and rebuild the ensemble. Must be called
     * before any training (the runner calls it right after construction). Defaults already
     * match MOA ARF, so this is only needed for sensitivity analysis.
     */
    public void setTreeParams(int gracePeriod, double splitConfidence) {
        if (gracePeriod < 1) throw new IllegalArgumentException("gracePeriod must be >= 1");
        if (!(splitConfidence > 0.0 && splitConfidence < 1.0))
            throw new IllegalArgumentException("splitConfidence must be in (0,1)");
        boolean changed = gracePeriod != this.treeGracePeriod
                || splitConfidence != this.treeSplitConfidence;
        this.treeGracePeriod = gracePeriod;
        this.treeSplitConfidence = splitConfidence;
        if (changed) buildEnsemble();  // rebuild initial learners with the new tree config
    }

    private void buildEnsemble() {
        ensemble = new BaseLearner[ensembleSize];
        for (int i = 0; i < ensembleSize; i++) ensemble[i] = newLearner(Set.of());
    }

    private BaseLearner newLearner(Set<Integer> avoid) {
        int[] sub = sampleSubspace(avoid);
        InstancesHeader reduced = FilteredHeaderBuilder.build(space, sub, "_daarf");
        ARFHoeffdingTree tree = newTree(sub.length, reduced);
        return new BaseLearner(sub, reduced, tree, accWindow,
                useBackgroundLearner ? new ADWIN(warningDelta) : null,
                new ADWIN(driftDelta));
    }

    private ARFHoeffdingTree newTree(int dim, InstancesHeader header) {
        ARFHoeffdingTree t = new ARFHoeffdingTree();
        t.subspaceSizeOption.setValue(dim);
        // A7: match MOA AdaptiveRandomForest's tree config so DA-ARF is not handicapped vs the
        // ARF baseline it is compared to. HoeffdingTree defaults (grace=200, confidence=1e-7)
        // build far too shallow trees -> majority-collapse on imbalanced streams (NHTS).
        t.gracePeriodOption.setValue(treeGracePeriod);
        t.splitConfidenceOption.setValue(treeSplitConfidence);
        t.maxByteSizeOption.setValue(2000000);
        t.prepareForUse();
        t.setModelContext(header);
        return t;
    }

    private int[] sampleSubspace(Set<Integer> avoid) {
        int effectiveSize = Math.min(subspaceSize, origDim - avoid.size());
        if (effectiveSize < 1) effectiveSize = Math.min(subspaceSize, origDim);
        Set<Integer> exclude = avoid;
        if (origDim - avoid.size() < effectiveSize) exclude = Set.of();
        if (importance != null) {
            double[] w = sharpenAndBlend(importance.getImportance(), importancePower, samplingBeta);
            int[] s = WeightedSubspaceSampler.sample(w, effectiveSize, rng, exclude);
            Arrays.sort(s);
            return s;
        }
        return uniformSubspace(effectiveSize, exclude);
    }

    private int[] uniformSubspace(int size, Set<Integer> avoid) {
        int[] pool = new int[origDim];
        int p = 0;
        for (int i = 0; i < origDim; i++) if (!avoid.contains(i)) pool[p++] = i;
        if (p < size) p = origDim;  // fall back to full pool
        if (pool.length != p) pool = Arrays.copyOf(pool, p);
        if (p == origDim) { pool = new int[origDim]; for (int i = 0; i < origDim; i++) pool[i] = i; }
        for (int i = pool.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
        }
        int[] out = Arrays.copyOf(pool, Math.min(size, pool.length));
        Arrays.sort(out);
        return out;
    }

    private static double[] sharpenAndBlend(double[] base, double power, double beta) {
        int n = base.length;
        double[] out = new double[n];
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double v = base[i];
            if (!Double.isFinite(v) || v < 0.0) v = 0.0;
            v = Math.max(1e-12, v);
            double p = Math.pow(v, power);
            if (!Double.isFinite(p) || p < 0.0) p = 0.0;
            out[i] = p; sum += p;
        }
        if (!Double.isFinite(sum) || sum <= 0.0) Arrays.fill(out, 1.0 / n);
        else for (int i = 0; i < n; i++) out[i] /= sum;
        if (beta < 1.0) {
            double u = 1.0 / n;
            double s2 = 0.0;
            for (int i = 0; i < n; i++) { out[i] = beta * out[i] + (1.0 - beta) * u; s2 += out[i]; }
            if (Double.isFinite(s2) && s2 > 0.0) for (int i = 0; i < n; i++) out[i] /= s2;
            else Arrays.fill(out, 1.0 / n);
        }
        return out;
    }

    @Override
    public double[] predictProba(Instance full) {
        double[] agg = new double[numClasses];
        // Rank by recent accuracy
        Integer[] order = new Integer[ensembleSize];
        for (int i = 0; i < ensembleSize; i++) order[i] = i;
        final double[] accs = new double[ensembleSize];
        for (int i = 0; i < ensembleSize; i++) accs[i] = ensemble[i].recentAccuracy();
        Arrays.sort(order, (a, b) -> Double.compare(accs[b], accs[a]));

        int K = Math.max(1, (int) Math.ceil(ensembleSize * topKFraction));
        if (K > ensembleSize) K = ensembleSize;

        double wSum = 0.0;
        int used = 0;
        for (int rank = 0; rank < K; rank++) {
            BaseLearner bl = ensemble[order[rank]];
            double[] v = bl.votes(full, space);
            if (v == null || v.length == 0) continue;
            double s = 0.0;
            for (double x : v) s += x;
            if (!(s > 0.0) || !Double.isFinite(s)) continue;
            double w = (double) (K - rank);
            int upTo = Math.min(v.length, numClasses);
            for (int c = 0; c < upTo; c++) agg[c] += w * (v[c] / s);
            wSum += w;
            used++;
        }
        if (used == 0 || !(wSum > 0.0)) {
            // Fallback: uniform vote across all learners
            for (int i = 0; i < ensembleSize; i++) {
                double[] v = ensemble[i].votes(full, space);
                if (v == null || v.length == 0) continue;
                double s = 0.0; for (double x : v) s += x;
                if (!(s > 0.0)) continue;
                int upTo = Math.min(v.length, numClasses);
                for (int c = 0; c < upTo; c++) agg[c] += v[c] / s;
                wSum += 1.0;
            }
            if (!(wSum > 0.0)) return new double[numClasses];
        }
        double aggSum = 0.0;
        for (int i = 0; i < numClasses; i++) { agg[i] /= wSum; aggSum += agg[i]; }
        if (!(aggSum > 0.0)) return agg;
        for (int i = 0; i < numClasses; i++) agg[i] /= aggSum;
        return agg;
    }

    @Override
    public int predict(Instance full) {
        double[] v = predictProba(full);
        if (v == null || v.length == 0) return 0;
        int best = 0;
        for (int i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    @Override
    public void train(Instance full, int classLabel) { train(full, classLabel, false, Set.of()); }

    @Override
    public void train(Instance full, int classLabel, boolean driftAlarm, Set<Integer> driftingFeatures) {
        instancesSeen++;
        Set<Integer> drifting = (driftingFeatures == null) ? Set.of() : driftingFeatures;
        // Online bagging weight
        for (int i = 0; i < ensembleSize; i++) {
            BaseLearner bl = ensemble[i];
            int k = poisson(lambda, rng);
            // Per-learner training error (unweighted prediction for window stats)
            int yhat = bl.predict(full, space);
            int err = (yhat == classLabel) ? 0 : 1;
            bl.updateWindow(err == 0);
            if (k > 0) bl.train(full, space, classLabel, k);

            // Background learner (if present) trains in parallel on the same instance.
            if (bl.background != null) {
                int kb = poisson(lambda, rng);
                if (kb > 0) bl.background.train(full, space, classLabel, kb);
            }

            // Intrinsic per-learner drift management (Component A)
            handleIntrinsicDrift(i, err);
        }

        // External drift alarm (Component A: KEEP/FULL pass)
        if (driftAlarm && !drifting.isEmpty()) {
            externalKeepOrFull(drifting);
        }
    }

    private void handleIntrinsicDrift(int idx, int err) {
        // A4: intrinsic per-tree drift management fully disabled — external layer is
        // the sole reset authority. Lets the ablation isolate which layer hurts.
        if (!intrinsicDriftEnabled) return;

        BaseLearner bl = ensemble[idx];
        // Warning channel: spawn background on first warning.
        if (bl.warning != null) {
            bl.warning.setInput(err);
            if (bl.warning.getChange() && bl.background == null) {
                warningCount++;
                bl.background = newLearner(Set.of());
            }
        }
        // Drift channel: promote background, else full reset.
        bl.drift.setInput(err);
        if (bl.drift.getChange()) {
            driftCount++;
            if (bl.background != null) {
                ensemble[idx] = bl.background;
                ensemble[idx].drift = new ADWIN(driftDelta);
                ensemble[idx].warning = useBackgroundLearner ? new ADWIN(warningDelta) : null;
                ensemble[idx].background = null;
                bkgPromotions++;
            } else {
                ensemble[idx] = newLearner(Set.of());
                intrinsicFullResetCount++;
            }
        }
    }

    private void externalKeepOrFull(Set<Integer> drifting) {
        Set<Integer> unstable = lowImportanceDriftingFeatures(drifting);
        if (unstable.isEmpty() || externalResetFraction <= 0.0) {
            extKeepCount += ensembleSize;
            return;
        }

        boolean[] candidate = new boolean[ensembleSize];
        int candidateCount = 0;
        for (int i = 0; i < ensembleSize; i++) {
            BaseLearner bl = ensemble[i];
            // A3: do not fight the intrinsic mechanism — a learner already adapting via
            // a pending background is left alone by the external pass.
            if (gateExternalOnPendingBackground && bl.background != null) {
                extGatedSkipCount++;
                continue;
            }
            int overlap = 0;
            for (int s : bl.subspace) if (unstable.contains(s)) overlap++;
            if (overlap > 0) {
                candidate[i] = true;
                candidateCount++;
            }
        }

        int resetLimit = (int) Math.ceil(ensembleSize * externalResetFraction);
        resetLimit = Math.max(1, Math.min(resetLimit, candidateCount));
        boolean[] reset = chooseLowestAccuracyCandidates(candidate, resetLimit);
        boolean[] driftMask = driftingMaskOf(unstable);

        for (int i = 0; i < ensembleSize; i++) {
            if (!reset[i]) {
                extKeepCount++;
                continue;
            }
            if (externalActionMode == ExternalActionMode.SURGICAL) {
                // A2: keep the trained tree, swap only the drifting features.
                if (surgicalReplaceLearner(i, driftMask)) extSurgicalCount++;
                else extNoReplacementCount++;
            } else {
                ensemble[i] = newLearner(unstable);
                extFullCount++;
            }
        }
    }

    private boolean[] driftingMaskOf(Set<Integer> unstable) {
        boolean[] m = new boolean[origDim];
        for (int u : unstable) if (u >= 0 && u < origDim) m[u] = true;
        return m;
    }

    /**
     * A2: replace only the drifting features inside a learner's subspace with the
     * best-scoring, type-compatible non-drifting features, keeping the trained tree.
     * Mirrors {@link DriftAwareSRP} surgical semantics. Returns true if a swap happened.
     */
    private boolean surgicalReplaceLearner(int idx, boolean[] driftingMask) {
        BaseLearner bl = ensemble[idx];
        double[] scores = (importance == null) ? null : importance.getImportance();
        if (scores == null || scores.length != origDim) return false;
        int[] newSub = surgicalReplaceSubspace(bl.subspace, driftingMask, scores);
        if (Arrays.equals(newSub, bl.subspace)) return false;
        // Re-point the projection to the new features; reduced header stays structurally
        // identical (type-compatible swaps) so the tree's learned splits remain valid.
        bl.subspace = newSub;
        bl.reducedHeader = FilteredHeaderBuilder.build(space, newSub, "_daarf");
        return true;
    }

    private int[] surgicalReplaceSubspace(int[] currentSub, boolean[] driftingMask, double[] scores) {
        boolean[] inSub = new boolean[origDim];
        for (int s : currentSub) if (s >= 0 && s < origDim) inSub[s] = true;

        int candCount = 0;
        for (int i = 0; i < origDim; i++) if (!inSub[i] && !driftingMask[i]) candCount++;
        if (candCount == 0) return currentSub;
        Integer[] cand = new Integer[candCount];
        int c = 0;
        for (int i = 0; i < origDim; i++) if (!inSub[i] && !driftingMask[i]) cand[c++] = i;
        Arrays.sort(cand, (a, b) -> Double.compare(scoreAt(scores, b), scoreAt(scores, a)));

        // Drifting positions in the subspace, weakest current score first.
        int dpc = 0;
        for (int i = 0; i < currentSub.length; i++) {
            int s = currentSub[i];
            if (s >= 0 && s < origDim && driftingMask[s]) dpc++;
        }
        Integer[] order = new Integer[dpc];
        int o = 0;
        for (int i = 0; i < currentSub.length; i++) {
            int s = currentSub[i];
            if (s >= 0 && s < origDim && driftingMask[s]) order[o++] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(
                scoreAt(scores, currentSub[a]), scoreAt(scores, currentSub[b])));

        int[] result = currentSub.clone();
        boolean[] used = new boolean[candCount];
        for (Integer posObj : order) {
            int pos = posObj;
            int pick = -1;
            for (int k = 0; k < candCount; k++) {
                if (used[k]) continue;
                int f = cand[k];
                if (!sameAttrType(f, result[pos])) continue;
                if (isAcceptableSurgical(scoreAt(scores, f), scoreAt(scores, result[pos]))) {
                    pick = k;
                    break;
                }
            }
            if (pick < 0) continue;  // no acceptable type-compatible swap for this position
            result[pos] = cand[pick];
            used[pick] = true;
        }
        return result;
    }

    private boolean isAcceptableSurgical(double replacement, double current) {
        if (!Double.isFinite(replacement)) return false;
        if (!Double.isFinite(current)) return true;
        return replacement >= current * surgicalReplacementTolerance;
    }

    private boolean sameAttrType(int a, int b) {
        InstancesHeader h = space.getHeader();
        return h.attribute(space.attrIndexOf(a)).isNumeric()
                == h.attribute(space.attrIndexOf(b)).isNumeric();
    }

    private static double scoreAt(double[] scores, int idx) {
        if (idx < 0 || idx >= scores.length) return Double.NEGATIVE_INFINITY;
        double v = scores[idx];
        return Double.isFinite(v) ? v : Double.NEGATIVE_INFINITY;
    }

    private boolean[] chooseLowestAccuracyCandidates(boolean[] candidate, int limit) {
        boolean[] selected = new boolean[candidate.length];
        for (int picked = 0; picked < limit; picked++) {
            int best = -1;
            for (int i = 0; i < candidate.length; i++) {
                if (!candidate[i] || selected[i]) continue;
                if (best < 0 || ensemble[i].recentAccuracy() < ensemble[best].recentAccuracy()) {
                    best = i;
                }
            }
            if (best < 0) break;
            selected[best] = true;
        }
        return selected;
    }

    private Set<Integer> lowImportanceDriftingFeatures(Set<Integer> drifting) {
        if (drifting == null || drifting.isEmpty()) return Set.of();
        if (importance == null) return drifting;
        double[] scores = importance.getImportance();
        if (scores == null || scores.length != origDim) return drifting;
        double threshold = finiteQuantile(scores, unstableImportanceQuantile);
        if (!Double.isFinite(threshold)) return drifting;
        Set<Integer> out = new HashSet<>();
        for (int f : drifting) {
            if (f < 0 || f >= scores.length) continue;
            double score = scores[f];
            if (Double.isFinite(score) && score <= threshold) out.add(f);
        }
        return out;
    }

    private static double finiteQuantile(double[] values, double quantile) {
        int count = 0;
        for (double v : values) if (Double.isFinite(v)) count++;
        if (count == 0) return Double.NaN;
        double[] finite = new double[count];
        int j = 0;
        for (double v : values) if (Double.isFinite(v)) finite[j++] = v;
        Arrays.sort(finite);
        int idx = (int) Math.floor(quantile * (finite.length - 1));
        if (idx < 0) idx = 0;
        if (idx >= finite.length) idx = finite.length - 1;
        return finite[idx];
    }

    private static int poisson(double lambda, Random rng) {
        // Knuth's algorithm — fine for small λ (≤ 10) used in MOA bagging.
        double L = Math.exp(-lambda);
        int k = 0;
        double p = 1.0;
        do { k++; p *= rng.nextDouble(); } while (p > L);
        return k - 1;
    }

    @Override
    public int[] getCurrentSelection() {
        // Union of per-learner subspaces (sorted, deduped) — exposed for diagnostics
        HashSet<Integer> u = new HashSet<>();
        for (BaseLearner bl : ensemble) for (int s : bl.subspace) u.add(s);
        int[] out = new int[u.size()];
        int j = 0;
        for (Integer v : u) out[j++] = v;
        Arrays.sort(out);
        return out;
    }

    public int[][] getAllSubspaces() {
        int[][] out = new int[ensembleSize][];
        for (int i = 0; i < ensembleSize; i++) out[i] = ensemble[i].subspace.clone();
        return out;
    }

    public double[] getRecentAccuracies() {
        double[] out = new double[ensembleSize];
        for (int i = 0; i < ensembleSize; i++) out[i] = ensemble[i].recentAccuracy();
        return out;
    }

    @Override
    public void reset() {
        buildEnsemble();
        instancesSeen = extKeepCount = extFullCount = bkgPromotions = warningCount = driftCount = 0;
        extSurgicalCount = extNoReplacementCount = extGatedSkipCount = intrinsicFullResetCount = 0;
    }

    @Override
    public String name() {
        return "DA-ARF(N=" + ensembleSize + ", m=" + subspaceSize
                + ", topK=" + topKFraction + ", bkg=" + useBackgroundLearner + ")";
    }

    /** Per-base-learner state. */
    private static final class BaseLearner {
        int[] subspace;
        InstancesHeader reducedHeader;
        ARFHoeffdingTree tree;
        BaseLearner background;
        ADWIN warning;
        ADWIN drift;

        // Sliding-window accuracy via fixed-size ring of {0,1} hits
        final int[] hits;
        int wIdx;
        int wCount;
        int wHits;

        BaseLearner(int[] subspace, InstancesHeader reduced, ARFHoeffdingTree tree,
                    int accWindow, ADWIN warning, ADWIN drift) {
            this.subspace = subspace;
            this.reducedHeader = reduced;
            this.tree = tree;
            this.warning = warning;
            this.drift = drift;
            this.hits = new int[accWindow];
        }

        void updateWindow(boolean correct) {
            int newVal = correct ? 1 : 0;
            if (wCount == hits.length) {
                wHits -= hits[wIdx];
            } else {
                wCount++;
            }
            hits[wIdx] = newVal;
            wHits += newVal;
            wIdx = (wIdx + 1) % hits.length;
        }

        double recentAccuracy() {
            return wCount == 0 ? 0.0 : (double) wHits / wCount;
        }

        int predict(Instance full, FeatureSpace space) {
            double[] v = votes(full, space);
            if (v == null || v.length == 0) return 0;
            int best = 0;
            for (int i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
            return best;
        }

        double[] votes(Instance full, FeatureSpace space) {
            Instance proj = FilteredHeaderBuilder.filteredInstance(full, space, subspace, reducedHeader);
            return tree.getVotesForInstance(proj);
        }

        void train(Instance full, FeatureSpace space, int classLabel, int weightK) {
            Instance proj = FilteredHeaderBuilder.filteredInstance(full, space, subspace, reducedHeader);
            proj.setClassValue(classLabel);
            proj.setWeight(weightK);
            tree.trainOnInstance(proj);
        }
    }
}
