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
import java.util.function.Consumer;

/**
 * Native (reflection-free) reimplementation of {@link DriftAwareSRP}.
 *
 * <p>The original {@code DriftAwareSRP} wraps MOA {@code StreamingRandomPatches} and pokes its
 * private per-learner subspace arrays via reflection (fragile, version-dependent, hard to
 * instrument/defend). This class instead owns the ensemble explicitly — the same architecture as
 * {@link DAARFWrapper}: a custom ensemble of MOA {@link ARFHoeffdingTree} base learners, each with
 * an explicit fixed feature subspace (a "patch"), online bagging, and a per-learner ADWIN drift
 * channel with a background learner. Nothing is read or written by reflection.
 *
 * <p>Drift-aware feature adaptation (component A/B) and top-K importance-corrected voting
 * (component C) reproduce the behaviour of {@link DriftAwareSRP}:
 * <ul>
 *   <li><b>A</b> — on an external drift alarm, {@link #handleDrift} performs a per-learner
 *       KEEP / SURGICAL / FULL pass by subspace overlap with the low-importance drifting features
 *       (threshold {@code tau}).</li>
 *   <li><b>B</b> — new subspaces are drawn importance-weighted (sharpen {@code importancePower},
 *       blend-to-uniform {@code samplingBeta}).</li>
 *   <li><b>C</b> — {@link #predictProba} blends the plain ensemble vote toward a top-K
 *       importance-weighted correction with weight {@code correctionAlpha} (capped at
 *       {@code maxBlendAlpha}).</li>
 * </ul>
 * With {@code importance == null} it degenerates to a plain SRP ensemble (matches DA-SRP-A).
 */
public class NativeDriftAwareSRP implements ModelWrapper {

    @Getter private final FeatureSelector selector;
    private final FeatureSpace space;
    private final int origDim;
    private final int numClasses;
    @Getter private final int ensembleSize;
    @Getter private final int subspaceSize;
    @Getter private final double lambda;
    @Getter private final int accWindow;
    @Getter private final double tau;

    private final Random rng;
    @Getter private FeatureImportance importance;

    // --- Component B (sampling) / C (voting) hyperparameters (defaults match DriftAwareSRP) ---
    @Getter private double importancePower = 2.0;
    @Getter private double samplingBeta = 0.7;
    @Getter private double topKFraction = 0.3;
    @Getter private double correctionAlpha = 0.15;
    @Getter private double maxBlendAlpha = 0.5;
    @Getter private double unlocalizedFallbackFraction = 0.20;
    @Getter private double unstableImportanceQuantile = 0.50;
    @Getter private double surgicalReplacementTolerance = 0.95;
    @Getter private int treeGracePeriod = 50;
    @Getter private double treeSplitConfidence = 0.01;

    // --- tree drift-channel deltas (per-learner ADWIN), like DAARFWrapper ---
    @Getter private final boolean useBackgroundLearner;
    @Getter private final double warningDelta;
    @Getter private final double driftDelta;

    private BaseLearner[] ensemble;
    private long instanceCounter;

    // --- diagnostics (same names as DriftAwareSRP) ---
    @Getter private long handleDriftCalls;
    @Getter private long autoHandleDriftCalls;
    @Getter private long totalKept;
    @Getter private long totalSurgical;
    @Getter private long totalFull;
    @Getter private long totalNoReplacement;
    @Getter private long totalSwapsPerformed;
    @Getter private long weightedPredictions;
    @Getter private long unweightedFallbacks;
    @Getter private long bkgPromotions;
    @Getter private long driftCount;
    @Getter private DriftActionSummary lastSummary;

    private Consumer<DriftAwareSRP.DriftEvent> driftListener;

    public NativeDriftAwareSRP(FeatureSelector selector, InstancesHeader fullHeader, int numClasses,
                               int ensembleSize, int subspaceSize, double lambda, int accWindow,
                               double tau, boolean useBackgroundLearner,
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
        if (!(tau > 0.0 && tau <= 1.0)) throw new IllegalArgumentException("tau must be in (0,1]");
        if (!(warningDelta > 0.0 && warningDelta < 1.0))
            throw new IllegalArgumentException("warningDelta must be in (0,1)");
        if (!(driftDelta > 0.0 && driftDelta < 1.0))
            throw new IllegalArgumentException("driftDelta must be in (0,1)");

        this.selector = selector;
        this.space = new FeatureSpace(fullHeader);
        this.origDim = space.numFeatures();
        if (subspaceSize > origDim) throw new IllegalArgumentException("subspaceSize > origDim");
        if (importance != null && importance.getNumFeatures() != origDim)
            throw new IllegalArgumentException("FeatureImportance dim mismatch");
        this.numClasses = numClasses;
        this.ensembleSize = ensembleSize;
        this.subspaceSize = subspaceSize;
        this.lambda = lambda;
        this.accWindow = accWindow;
        this.tau = tau;
        this.useBackgroundLearner = useBackgroundLearner;
        this.warningDelta = warningDelta;
        this.driftDelta = driftDelta;
        this.rng = new Random(seed);
        this.importance = importance;
        buildEnsemble();
    }

    /**
     * MOA StreamingRandomPatches uses a 60%-of-features patch by default (subspaceSize=60,
     * mode=Percentage). Mirror that so the native ensemble matches the reflection-based DA-SRP —
     * a narrow sqrt(M) patch collapses on imbalanced/real data (same lesson as DA-ARF).
     */
    public static int defaultSubspaceSize(int origDim) {
        return Math.max(2, Math.min(origDim, (int) Math.ceil(0.6 * origDim)));
    }

    public static NativeDriftAwareSRP defaults(FeatureSelector selector, InstancesHeader header,
                                               int numClasses, long seed, FeatureImportance importance) {
        int origDim = header.numAttributes() - 1;
        return new NativeDriftAwareSRP(selector, header, numClasses,
                /*ensemble=*/10, defaultSubspaceSize(origDim), /*lambda=*/6.0, /*accWindow=*/1000,
                /*tau=*/0.5, /*useBkg=*/true, /*warnDelta=*/1e-4, /*driftDelta=*/1e-5, seed, importance);
    }

    // --- setters (mirror DriftAwareSRP contract) ------------------------------------------------

    public void setFeatureImportance(FeatureImportance imp) {
        if (imp != null && imp.getNumFeatures() != origDim)
            throw new IllegalArgumentException("FeatureImportance dim mismatch");
        this.importance = imp;
    }

    public void setImportancePower(double v) {
        if (!(v > 0.0) || !Double.isFinite(v)) throw new IllegalArgumentException("importancePower > 0");
        this.importancePower = v;
    }
    public void setSamplingBeta(double v) {
        if (!(v >= 0.0 && v <= 1.0)) throw new IllegalArgumentException("samplingBeta in [0,1]");
        this.samplingBeta = v;
    }
    public void setTopKFraction(double v) {
        if (!(v > 0.0 && v <= 1.0)) throw new IllegalArgumentException("topKFraction in (0,1]");
        this.topKFraction = v;
    }
    public void setCorrectionAlpha(double v) {
        if (!(v >= 0.0 && v <= 1.0) || !Double.isFinite(v))
            throw new IllegalArgumentException("correctionAlpha in [0,1]");
        this.correctionAlpha = v;
    }
    public void setMaxBlendAlpha(double v) {
        if (!(v >= 0.0 && v <= 1.0) || !Double.isFinite(v))
            throw new IllegalArgumentException("maxBlendAlpha in [0,1]");
        this.maxBlendAlpha = v;
    }
    public void setUnlocalizedFallbackFraction(double v) {
        if (!(v >= 0.0 && v <= 1.0) || !Double.isFinite(v))
            throw new IllegalArgumentException("unlocalizedFallbackFraction in [0,1]");
        this.unlocalizedFallbackFraction = v;
    }
    public void setUnstableImportanceQuantile(double v) {
        if (!(v >= 0.0 && v <= 1.0) || !Double.isFinite(v))
            throw new IllegalArgumentException("unstableImportanceQuantile in [0,1]");
        this.unstableImportanceQuantile = v;
    }
    public void setSurgicalReplacementTolerance(double v) {
        if (!(v >= 0.0 && v <= 1.0) || !Double.isFinite(v))
            throw new IllegalArgumentException("surgicalReplacementTolerance in [0,1]");
        this.surgicalReplacementTolerance = v;
    }
    public void setTreeParams(int gracePeriod, double splitConfidence) {
        if (gracePeriod < 1) throw new IllegalArgumentException("gracePeriod >= 1");
        if (!(splitConfidence > 0.0 && splitConfidence < 1.0))
            throw new IllegalArgumentException("splitConfidence in (0,1)");
        boolean changed = gracePeriod != treeGracePeriod || splitConfidence != treeSplitConfidence;
        this.treeGracePeriod = gracePeriod;
        this.treeSplitConfidence = splitConfidence;
        if (changed) buildEnsemble();
    }
    public void setDriftListener(Consumer<DriftAwareSRP.DriftEvent> listener) {
        this.driftListener = listener;
    }

    public int[][] getAllSubspaces() {
        int[][] out = new int[ensembleSize][];
        for (int i = 0; i < ensembleSize; i++) out[i] = ensemble[i].subspace.clone();
        return out;
    }

    // --- ensemble construction ------------------------------------------------------------------

    private void buildEnsemble() {
        ensemble = new BaseLearner[ensembleSize];
        for (int i = 0; i < ensembleSize; i++) ensemble[i] = newLearner(Set.of());
    }

    private BaseLearner newLearner(Set<Integer> avoid) {
        int[] sub = sampleSubspace(avoid);
        InstancesHeader reduced = FilteredHeaderBuilder.build(space, sub, "_dasrp");
        ARFHoeffdingTree tree = newTree(sub.length, reduced);
        return new BaseLearner(sub, reduced, tree, accWindow,
                useBackgroundLearner ? new ADWIN(warningDelta) : null, new ADWIN(driftDelta));
    }

    private ARFHoeffdingTree newTree(int dim, InstancesHeader header) {
        ARFHoeffdingTree t = new ARFHoeffdingTree();
        t.subspaceSizeOption.setValue(dim);   // patch: use all projected features, no internal subspacing
        t.gracePeriodOption.setValue(treeGracePeriod);
        t.splitConfidenceOption.setValue(treeSplitConfidence);
        t.maxByteSizeOption.setValue(2000000);
        t.prepareForUse();
        t.setModelContext(header);
        return t;
    }

    private int[] sampleSubspace(Set<Integer> avoid) {
        int effective = Math.min(subspaceSize, origDim - avoid.size());
        if (effective < 1) effective = Math.min(subspaceSize, origDim);
        Set<Integer> exclude = (origDim - avoid.size() < effective) ? Set.of() : avoid;
        if (importance != null) {
            double[] w = sharpenAndBlend(importance.getImportance(), importancePower, samplingBeta);
            int[] s = WeightedSubspaceSampler.sample(w, effective, rng, exclude);
            Arrays.sort(s);
            return s;
        }
        return uniformSubspace(effective, exclude);
    }

    private int[] uniformSubspace(int size, Set<Integer> avoid) {
        int[] pool = new int[origDim];
        int p = 0;
        for (int i = 0; i < origDim; i++) if (!avoid.contains(i)) pool[p++] = i;
        if (p < size) { pool = new int[origDim]; for (int i = 0; i < origDim; i++) pool[i] = i; p = origDim; }
        else pool = Arrays.copyOf(pool, p);
        for (int i = pool.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
        }
        int[] out = Arrays.copyOf(pool, Math.min(size, pool.length));
        Arrays.sort(out);
        return out;
    }

    // --- training ------------------------------------------------------------------------------

    @Override public void train(Instance full, int classLabel) { train(full, classLabel, false, Set.of()); }

    @Override
    public void train(Instance full, int classLabel, boolean driftAlarm, Set<Integer> driftingFeatures) {
        instanceCounter++;
        Set<Integer> drifting = (driftingFeatures == null) ? Set.of() : driftingFeatures;
        for (int i = 0; i < ensembleSize; i++) {
            BaseLearner bl = ensemble[i];
            int k = poisson(lambda, rng);
            int yhat = bl.predict(full, space);
            bl.updateWindow(yhat == classLabel);
            if (k > 0) bl.train(full, space, classLabel, k);
            if (bl.background != null) {
                int kb = poisson(lambda, rng);
                if (kb > 0) bl.background.train(full, space, classLabel, kb);
            }
            handleIntrinsicDrift(i, yhat == classLabel ? 0 : 1);
        }
        // Component A: external, feature-driven KEEP/SURGICAL/FULL pass.
        if (driftAlarm) {
            double[] scores = resolveScores();
            if (scores != null) {
                DriftActionSummary s = handleDrift(drifting, scores);
                autoHandleDriftCalls++;
                if (driftListener != null) {
                    driftListener.accept(new DriftAwareSRP.DriftEvent(instanceCounter, true, drifting, s,
                            importance == null ? null : importance.getImportance(), null));
                }
            }
        }
    }

    private void handleIntrinsicDrift(int idx, int err) {
        BaseLearner bl = ensemble[idx];
        if (bl.warning != null) {
            bl.warning.setInput(err);
            if (bl.warning.getChange() && bl.background == null) bl.background = newLearner(Set.of());
        }
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
            }
        }
    }

    private double[] resolveScores() {
        if (importance == null) return null;
        double[] imp = importance.getImportance();
        if (imp != null && imp.length == origDim && hasAnyPositive(imp)) return imp;
        double[] mi = importance.getMIScores();
        if (mi != null && mi.length == origDim && hasAnyPositive(mi)) return mi;
        return null;
    }

    // --- drift-aware feature adaptation (component A/B) -----------------------------------------

    public DriftActionSummary handleDrift(Set<Integer> driftingOriginal, double[] scoresOriginal) {
        if (scoresOriginal == null || scoresOriginal.length != origDim) {
            lastSummary = new DriftActionSummary(ensembleSize);
            return lastSummary;
        }
        if (driftingOriginal == null || driftingOriginal.isEmpty()) {
            if (unlocalizedFallbackFraction > 0.0) return handleUnlocalized(scoresOriginal);
            lastSummary = new DriftActionSummary(ensembleSize);
            return lastSummary;
        }
        Set<Integer> unstable = lowImportanceDrifting(driftingOriginal, scoresOriginal);
        boolean[] mask = new boolean[origDim];
        for (int idx : unstable) if (idx >= 0 && idx < origDim) mask[idx] = true;

        DriftActionSummary summary = new DriftActionSummary(ensembleSize);
        for (int li = 0; li < ensembleSize; li++) {
            int[] sub = ensemble[li].subspace;
            if (sub.length == 0) { summary.record(li, DriftActionSummary.Action.KEEP, 0, 0, 0); continue; }
            int overlap = 0;
            for (int s : sub) if (s >= 0 && s < origDim && mask[s]) overlap++;
            double frac = overlap / (double) sub.length;
            if (overlap == 0) {
                summary.record(li, DriftActionSummary.Action.KEEP, 0, sub.length, 0);
            } else if (frac < tau) {
                int[] result = surgicalReplace(sub, scoresOriginal, mask);
                int swaps = countDifferences(result, sub);
                if (swaps == 0) {
                    summary.record(li, DriftActionSummary.Action.NO_REPLACEMENT, overlap, sub.length, 0);
                } else {
                    rebuildSubspace(li, result);       // keep the tree, re-point the projection
                    summary.record(li, DriftActionSummary.Action.SURGICAL, overlap, sub.length, swaps);
                }
            } else {
                Set<Integer> avoid = (origDim - unstable.size() < sub.length) ? Set.of() : unstable;
                int[] newSub = sampleSubspace(avoid);
                rebuildSubspace(li, newSub);
                resetLearner(li, newSub);              // full: fresh tree on the new subspace
                summary.record(li, DriftActionSummary.Action.FULL, overlap, sub.length, sub.length);
            }
        }
        handleDriftCalls++;
        totalKept += summary.getKeptCount();
        totalSurgical += summary.getSurgicalCount();
        totalFull += summary.getFullCount();
        totalNoReplacement += summary.getNoReplacementCount();
        for (int s : summary.getSwapCounts()) totalSwapsPerformed += s;
        lastSummary = summary;
        return summary;
    }

    private DriftActionSummary handleUnlocalized(double[] scores) {
        DriftActionSummary summary = new DriftActionSummary(ensembleSize);
        double[] subScore = new double[ensembleSize];
        Arrays.fill(subScore, Double.POSITIVE_INFINITY);
        for (int li = 0; li < ensembleSize; li++) {
            int[] sub = ensemble[li].subspace;
            if (sub.length == 0) { summary.record(li, DriftActionSummary.Action.KEEP, 0, 0, 0); continue; }
            double s = 0; int c = 0;
            for (int f : sub) if (f >= 0 && f < origDim && Double.isFinite(scores[f])) { s += scores[f]; c++; }
            subScore[li] = (c == 0) ? Double.POSITIVE_INFINITY : s / c;
        }
        int target = Math.max(1, Math.min(ensembleSize, (int) Math.ceil(ensembleSize * unlocalizedFallbackFraction)));
        boolean[] refresh = lowestScored(subScore, target);
        for (int li = 0; li < ensembleSize; li++) {
            int[] sub = ensemble[li].subspace;
            if (sub.length == 0 || !refresh[li]) {
                summary.record(li, DriftActionSummary.Action.KEEP, 0, sub.length, 0);
                continue;
            }
            int[] newSub = sampleSubspace(Set.of());
            rebuildSubspace(li, newSub);
            resetLearner(li, newSub);
            summary.record(li, DriftActionSummary.Action.FULL, 0, sub.length, sub.length);
        }
        handleDriftCalls++;
        totalKept += summary.getKeptCount();
        totalFull += summary.getFullCount();
        lastSummary = summary;
        return summary;
    }

    /** Keep the trained tree; only re-point projection to the new (same-length) subspace. */
    private void rebuildSubspace(int li, int[] newSub) {
        BaseLearner bl = ensemble[li];
        bl.subspace = newSub;
        bl.reducedHeader = FilteredHeaderBuilder.build(space, newSub, "_dasrp");
    }

    private void resetLearner(int li, int[] sub) {
        BaseLearner bl = ensemble[li];
        bl.tree = newTree(sub.length, bl.reducedHeader);
        bl.drift = new ADWIN(driftDelta);
        bl.warning = useBackgroundLearner ? new ADWIN(warningDelta) : null;
        bl.background = null;
        bl.resetWindow();
    }

    private int[] surgicalReplace(int[] currentSub, double[] scores, boolean[] driftingMask) {
        boolean[] inSub = new boolean[origDim];
        for (int s : currentSub) if (s >= 0 && s < origDim) inSub[s] = true;
        int candCount = 0;
        for (int i = 0; i < origDim; i++) if (!inSub[i] && !driftingMask[i]) candCount++;
        if (candCount == 0) return currentSub.clone();
        int[] cand = new int[candCount];
        int j = 0;
        for (int i = 0; i < origDim; i++) if (!inSub[i] && !driftingMask[i]) cand[j++] = i;
        sortIndicesByScoreDesc(cand, scores);

        int[] driftPos = new int[currentSub.length];
        int dpc = 0;
        for (int i = 0; i < currentSub.length; i++) {
            int s = currentSub[i];
            if (s >= 0 && s < origDim && driftingMask[s]) driftPos[dpc++] = i;
        }
        int[] sortedPos = Arrays.copyOf(driftPos, dpc);
        sortPositionsByCurrentScoreAsc(sortedPos, currentSub, scores);

        int[] result = currentSub.clone();
        int ci = 0;
        for (int p = 0; p < sortedPos.length && ci < cand.length; p++) {
            int pos = sortedPos[p];
            int repl = cand[ci];
            if (acceptable(scores[repl], scores[result[pos]])) { result[pos] = repl; ci++; }
            else break;
        }
        return result;
    }

    private boolean acceptable(double repl, double cur) {
        if (!Double.isFinite(repl)) return false;
        if (!Double.isFinite(cur)) return true;
        return repl >= cur * surgicalReplacementTolerance;
    }

    private Set<Integer> lowImportanceDrifting(Set<Integer> drifting, double[] scores) {
        if (drifting == null || drifting.isEmpty()) return Set.of();
        double threshold = finiteQuantile(scores, unstableImportanceQuantile);
        if (!Double.isFinite(threshold)) return drifting;
        Set<Integer> out = new HashSet<>();
        for (int f : drifting) {
            if (f < 0 || f >= scores.length) continue;
            if (Double.isFinite(scores[f]) && scores[f] <= threshold) out.add(f);
        }
        return out;
    }

    // --- prediction (component C: blend toward top-K importance-weighted correction) ------------

    @Override
    public double[] predictProba(Instance full) {
        double[] base = ensembleVote(full, -1);  // plain unweighted ensemble vote
        if (importance == null || correctionAlpha <= 0.0) { unweightedFallbacks++; return base; }
        double[] impOrig = importance.getImportance();
        if (impOrig == null || !isAllFinite(impOrig) || !hasAnyPositive(impOrig)) {
            unweightedFallbacks++; return base;
        }
        // Per-learner subspace importance, rank, take top-K rank-weighted vote.
        double[] raw = new double[ensembleSize];
        int positive = 0;
        for (int li = 0; li < ensembleSize; li++) {
            int[] sub = ensemble[li].subspace;
            double w = 0; int c = 0;
            for (int s : sub) if (s >= 0 && s < origDim && Double.isFinite(impOrig[s]) && impOrig[s] >= 0) { w += impOrig[s]; c++; }
            raw[li] = (c == 0) ? 0.0 : w / c;
            if (raw[li] > 0.0) positive++;
        }
        if (positive == 0) { unweightedFallbacks++; return base; }
        Integer[] order = new Integer[ensembleSize];
        for (int i = 0; i < ensembleSize; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(raw[b], raw[a]));
        int K = Math.max(1, Math.min(ensembleSize, (int) Math.ceil(ensembleSize * topKFraction)));

        double[] agg = new double[numClasses];
        double wSum = 0; int used = 0;
        for (int rank = 0; rank < K; rank++) {
            int li = order[rank];
            if (!(raw[li] > 0.0)) break;
            double[] v = ensemble[li].votes(full, space);
            if (v == null || v.length == 0 || !isAllFinite(v)) continue;
            double vs = 0; for (double x : v) vs += x;
            if (!(vs > 0.0)) continue;
            double w = (double) (K - rank);
            int upTo = Math.min(v.length, numClasses);
            for (int ci = 0; ci < upTo; ci++) agg[ci] += w * (v[ci] / vs);
            wSum += w; used++;
        }
        if (used == 0 || !(wSum > 0.0)) { unweightedFallbacks++; return base; }
        double aggSum = 0;
        for (int i = 0; i < numClasses; i++) { agg[i] /= wSum; aggSum += agg[i]; }
        if (!(aggSum > 0.0)) { unweightedFallbacks++; return base; }
        for (int i = 0; i < numClasses; i++) agg[i] /= aggSum;

        double baseSum = 0; for (double v : base) baseSum += v;
        if (!(baseSum > 0.0)) { unweightedFallbacks++; return base; }
        double[] baseNorm = new double[numClasses];
        for (int i = 0; i < numClasses; i++) baseNorm[i] = base[i] / baseSum;

        double alpha = Math.max(0.0, Math.min(correctionAlpha, maxBlendAlpha));
        double[] blended = new double[numClasses];
        double bsum = 0;
        for (int i = 0; i < numClasses; i++) { blended[i] = (1 - alpha) * baseNorm[i] + alpha * agg[i]; bsum += blended[i]; }
        if (!(bsum > 0.0) || !isAllFinite(blended)) { unweightedFallbacks++; return base; }
        for (int i = 0; i < numClasses; i++) blended[i] /= bsum;
        weightedPredictions++;
        return blended;
    }

    /** Plain ensemble vote. If topN>0, only the topN learners by recent accuracy are aggregated. */
    private double[] ensembleVote(Instance full, int topN) {
        double[] agg = new double[numClasses];
        double wSum = 0;
        int[] idx;
        if (topN > 0 && topN < ensembleSize) {
            Integer[] order = new Integer[ensembleSize];
            for (int i = 0; i < ensembleSize; i++) order[i] = i;
            final double[] accs = new double[ensembleSize];
            for (int i = 0; i < ensembleSize; i++) accs[i] = ensemble[i].recentAccuracy();
            Arrays.sort(order, (a, b) -> Double.compare(accs[b], accs[a]));
            idx = new int[topN];
            for (int i = 0; i < topN; i++) idx[i] = order[i];
        } else {
            idx = new int[ensembleSize];
            for (int i = 0; i < ensembleSize; i++) idx[i] = i;
        }
        for (int li : idx) {
            double[] v = ensemble[li].votes(full, space);
            if (v == null || v.length == 0 || !isAllFinite(v)) continue;
            double vs = 0; for (double x : v) vs += x;
            if (!(vs > 0.0)) continue;
            int upTo = Math.min(v.length, numClasses);
            for (int c = 0; c < upTo; c++) agg[c] += v[c] / vs;
            wSum += 1.0;
        }
        if (!(wSum > 0.0)) return new double[numClasses];
        double sum = 0;
        for (int i = 0; i < numClasses; i++) { agg[i] /= wSum; sum += agg[i]; }
        if (sum > 0.0) for (int i = 0; i < numClasses; i++) agg[i] /= sum;
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

    // --- ModelWrapper misc ---------------------------------------------------------------------

    @Override
    public int[] getCurrentSelection() {
        HashSet<Integer> u = new HashSet<>();
        for (BaseLearner bl : ensemble) for (int s : bl.subspace) u.add(s);
        int[] out = new int[u.size()];
        int j = 0;
        for (int v : u) out[j++] = v;
        Arrays.sort(out);
        return out;
    }

    /**
     * Sum over every live patch: each foreground learner plus any pending background learner
     * (a background tree is real, resident memory and must be charged to the model).
     */
    @Override
    public long modelByteSize() {
        long total = 0L;
        for (int i = 0; i < ensembleSize; i++) {
            BaseLearner bl = ensemble[i];
            if (bl == null) continue;
            long fg = ModelSize.of(bl.tree);
            if (fg < 0L) return ModelSize.UNAVAILABLE;
            total += fg;
            if (bl.background != null) {
                long bg = ModelSize.of(bl.background.tree);
                if (bg < 0L) return ModelSize.UNAVAILABLE;
                total += bg;
            }
        }
        return total;
    }

    @Override
    public void reset() {
        buildEnsemble();
        instanceCounter = handleDriftCalls = autoHandleDriftCalls = 0;
        totalKept = totalSurgical = totalFull = totalNoReplacement = totalSwapsPerformed = 0;
        weightedPredictions = unweightedFallbacks = bkgPromotions = driftCount = 0;
        lastSummary = null;
    }

    @Override
    public String name() {
        return "NativeDA-SRP(N=" + ensembleSize + ", m=" + subspaceSize
                + ", tau=" + tau + ", topK=" + topKFraction + ", alpha=" + correctionAlpha + ")";
    }

    // --- shared helpers ------------------------------------------------------------------------

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
            double u = 1.0 / n, s2 = 0.0;
            for (int i = 0; i < n; i++) { out[i] = beta * out[i] + (1.0 - beta) * u; s2 += out[i]; }
            if (Double.isFinite(s2) && s2 > 0.0) for (int i = 0; i < n; i++) out[i] /= s2;
            else Arrays.fill(out, 1.0 / n);
        }
        return out;
    }

    private static boolean[] lowestScored(double[] score, int k) {
        boolean[] sel = new boolean[score.length];
        for (int picked = 0; picked < k; picked++) {
            int best = -1;
            for (int i = 0; i < score.length; i++) {
                if (sel[i]) continue;
                if (best < 0 || score[i] < score[best]) best = i;
            }
            if (best < 0) break;
            sel[best] = true;
        }
        return sel;
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

    private static void sortIndicesByScoreDesc(int[] idx, double[] scores) {
        for (int i = 1; i < idx.length; i++) {
            int cur = idx[i]; double cs = scoreAt(scores, cur); int j = i - 1;
            while (j >= 0 && scoreAt(scores, idx[j]) < cs) { idx[j + 1] = idx[j]; j--; }
            idx[j + 1] = cur;
        }
    }

    private static void sortPositionsByCurrentScoreAsc(int[] positions, int[] currentSub, double[] scores) {
        for (int i = 1; i < positions.length; i++) {
            int curPos = positions[i]; double cs = scoreAt(scores, currentSub[curPos]); int j = i - 1;
            while (j >= 0 && scoreAt(scores, currentSub[positions[j]]) > cs) { positions[j + 1] = positions[j]; j--; }
            positions[j + 1] = curPos;
        }
    }

    private static double scoreAt(double[] s, int i) {
        if (i < 0 || i >= s.length) return Double.NEGATIVE_INFINITY;
        return Double.isFinite(s[i]) ? s[i] : Double.NEGATIVE_INFINITY;
    }

    private static int countDifferences(int[] a, int[] b) {
        int d = 0; for (int i = 0; i < a.length; i++) if (a[i] != b[i]) d++; return d;
    }
    private static boolean isAllFinite(double[] v) { for (double x : v) if (!Double.isFinite(x)) return false; return true; }
    private static boolean hasAnyPositive(double[] v) { for (double x : v) if (x > 0.0 && Double.isFinite(x)) return true; return false; }

    private static int poisson(double lambda, Random rng) {
        double L = Math.exp(-lambda);
        int k = 0; double p = 1.0;
        do { k++; p *= rng.nextDouble(); } while (p > L);
        return k - 1;
    }

    // --- per-learner state (SRP patch: projected tree + ADWIN + explicit subspace) --------------

    private static final class BaseLearner {
        int[] subspace;
        InstancesHeader reducedHeader;
        ARFHoeffdingTree tree;
        BaseLearner background;
        ADWIN warning;
        ADWIN drift;
        final int[] hits;
        int wIdx, wCount, wHits;

        BaseLearner(int[] subspace, InstancesHeader reduced, ARFHoeffdingTree tree,
                    int accWindow, ADWIN warning, ADWIN drift) {
            this.subspace = subspace; this.reducedHeader = reduced; this.tree = tree;
            this.warning = warning; this.drift = drift; this.hits = new int[accWindow];
        }

        void updateWindow(boolean correct) {
            int v = correct ? 1 : 0;
            if (wCount == hits.length) wHits -= hits[wIdx]; else wCount++;
            hits[wIdx] = v; wHits += v; wIdx = (wIdx + 1) % hits.length;
        }
        void resetWindow() { Arrays.fill(hits, 0); wIdx = wCount = wHits = 0; }
        double recentAccuracy() { return wCount == 0 ? 0.0 : (double) wHits / wCount; }

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
