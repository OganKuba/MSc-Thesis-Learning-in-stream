package thesis.pipeline;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.core.Example;
import moa.core.InstanceExample;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.options.OptionHandler;
import moa.streams.ConceptDriftStream;
import moa.streams.InstanceStream;
import moa.streams.generators.HyperplaneGenerator;
import moa.streams.generators.RandomRBFGeneratorDrift;
import moa.streams.generators.SEAGenerator;
import moa.streams.generators.STAGGERGenerator;
import moa.tasks.TaskMonitor;

import java.util.ArrayList;
import java.util.Random;

public final class SyntheticStreamFactory {

    private SyntheticStreamFactory() {}

    public static InstanceStream createSEA(int seed, int numInstances) {
        int p1 = Math.max(1, numInstances / 4);
        int p2 = Math.max(p1 + 1, numInstances / 2);
        int p3 = Math.max(p2 + 1, (3 * numInstances) / 4);

        SEAGenerator g1 = newSEA(seed,     1);
        SEAGenerator g2 = newSEA(seed + 1, 2);
        SEAGenerator g3 = newSEA(seed + 2, 3);
        SEAGenerator g4 = newSEA(seed + 3, 4);

        ConceptDriftStream s34   = newDrift(g3, g4, p3, 1, seed);
        ConceptDriftStream s234  = newDrift(g2, s34, p2, 1, seed + 10);
        ConceptDriftStream s1234 = newDrift(g1, s234, p1, 1, seed + 20);
        s1234.prepareForUse();
        return limit(s1234, numInstances);
    }

    private static SEAGenerator newSEA(int seed, int function) {
        SEAGenerator g = new SEAGenerator();
        g.instanceRandomSeedOption.setValue(seed);
        g.functionOption.setValue(function);
        g.balanceClassesOption.setValue(false);
        g.noisePercentageOption.setValue(10);
        g.prepareForUse();
        return g;
    }

    public static InstanceStream createHyperplane(int seed, double sigma, int numInstances) {
        HyperplaneGenerator g = new HyperplaneGenerator();
        g.instanceRandomSeedOption.setValue(seed);
        g.numAttsOption.setValue(15);
        g.numDriftAttsOption.setValue(5);
        g.magChangeOption.setValue(sigma);
        g.noisePercentageOption.setValue(5);
        g.sigmaPercentageOption.setValue(10);
        g.prepareForUse();
        return limit(g, numInstances);
    }

    public static InstanceStream createRandomRBF(int seed, double speed, int numInstances) {
        RandomRBFGeneratorDrift g = new RandomRBFGeneratorDrift();
        g.modelRandomSeedOption.setValue(seed);
        g.instanceRandomSeedOption.setValue(seed + 1);
        g.numAttsOption.setValue(15);
        g.numCentroidsOption.setValue(20);
        g.numDriftCentroidsOption.setValue(5);
        g.speedChangeOption.setValue(speed);
        g.numClassesOption.setValue(2);
        g.prepareForUse();
        return limit(g, numInstances);
    }

    public static InstanceStream createSTAGGER(int seed, int numInstances) {
        int p1 = Math.max(1, numInstances / 5);
        int p2 = Math.max(p1 + 1, (2 * numInstances) / 5);
        int p3 = Math.max(p2 + 1, (3 * numInstances) / 5);

        STAGGERGenerator s1  = newStagger(seed,     1);
        STAGGERGenerator s2  = newStagger(seed + 1, 2);
        STAGGERGenerator s3  = newStagger(seed + 2, 3);
        STAGGERGenerator s1b = newStagger(seed + 3, 1);

        ConceptDriftStream d3 = newDrift(s3, s1b, p3, 1, seed + 30);
        ConceptDriftStream d2 = newDrift(s2, d3,  p2, 1, seed + 40);
        ConceptDriftStream d1 = newDrift(s1, d2,  p1, 1, seed + 50);
        d1.prepareForUse();
        return limit(d1, numInstances);
    }

    private static STAGGERGenerator newStagger(int seed, int function) {
        STAGGERGenerator s = new STAGGERGenerator();
        s.instanceRandomSeedOption.setValue(seed);
        s.functionOption.setValue(function);
        s.prepareForUse();
        return s;
    }

    public static InstanceStream createCustomFeatureDrift(int seed,
                                                          int numDriftFeatures,
                                                          double sigma,
                                                          int numInstances) {
        if (numDriftFeatures < 1 || numDriftFeatures > 20) {
            throw new IllegalArgumentException("numDriftFeatures must be in [1,20]");
        }
        HyperplaneGenerator g = new HyperplaneGenerator();
        g.instanceRandomSeedOption.setValue(seed);
        g.numAttsOption.setValue(20);
        g.numDriftAttsOption.setValue(numDriftFeatures);
        g.magChangeOption.setValue(sigma);
        g.noisePercentageOption.setValue(5);
        g.sigmaPercentageOption.setValue(10);
        g.prepareForUse();
        return limit(g, numInstances);
    }

    public static InstanceStream addNoiseFeatures(InstanceStream stream, int nNoise, int seed) {
        if (nNoise < 0 || nNoise > 1000) {
            throw new IllegalArgumentException("nNoise must be in [0, 1000]");
        }
        if (nNoise == 0) return stream;
        return new NoiseAugmentedStream(stream, nNoise, seed);
    }

    public static InstanceStream addNoiseFeatures(InstanceStream stream, int nNoise) {
        return addNoiseFeatures(stream, nNoise, 42);
    }

    private static InstanceStream limit(InstanceStream s, int numInstances) {
        if (numInstances <= 0) return s;
        return new LimitedStream(s, numInstances);
    }

    private static ConceptDriftStream newDrift(InstanceStream a, InstanceStream b,
                                               int position, int width, int seed) {
        ConceptDriftStream d = new ConceptDriftStream();
        d.streamOption.setCurrentObject(a);
        d.driftstreamOption.setCurrentObject(b);
        d.positionOption.setValue(position);
        d.widthOption.setValue(Math.max(1, width));
        d.randomSeedOption.setValue(seed);
        return d;
    }

    private static final class LimitedStream extends AbstractOptionHandler
            implements InstanceStream {

        private final InstanceStream base;
        private final long limit;
        private long emitted;

        LimitedStream(InstanceStream base, long limit) {
            this.base = base;
            this.limit = limit;
            this.emitted = 0;
        }

        @Override public InstancesHeader getHeader()        { return base.getHeader(); }
        @Override public boolean isRestartable()             { return base.isRestartable(); }

        @Override
        public long estimatedRemainingInstances() {
            long rem = limit - emitted;
            long baseRem = base.estimatedRemainingInstances();
            if (baseRem < 0) return rem;
            return Math.min(rem, baseRem);
        }

        @Override
        public boolean hasMoreInstances() {
            return emitted < limit && base.hasMoreInstances();
        }

        @Override
        public Example<Instance> nextInstance() {
            emitted++;
            return base.nextInstance();
        }

        @Override
        public void restart() {
            base.restart();
            emitted = 0;
        }

        @Override public void getDescription(StringBuilder sb, int indent) { sb.append("LimitedStream"); }

        @Override
        protected void prepareForUseImpl(TaskMonitor m, ObjectRepository r) {
            if (base instanceof OptionHandler) {
                ((OptionHandler) base).prepareForUse();
            }
        }
    }

    private static final class NoiseAugmentedStream extends AbstractOptionHandler
            implements InstanceStream {

        private final InstanceStream base;
        private final int nNoise;
        private final int seed;
        private Random rng;
        private InstancesHeader augmentedHeader;
        private int newClassIndex;

        NoiseAugmentedStream(InstanceStream base, int nNoise, int seed) {
            this.base = base;
            this.nNoise = nNoise;
            this.seed = seed;
            this.rng = new Random(seed);
            buildHeader();
        }

        private void buildHeader() {
            InstancesHeader src = base.getHeader();
            ArrayList<Attribute> attrs = new ArrayList<>();
            int origClass = src.classIndex();

            for (int i = 0; i < src.numAttributes(); i++) {
                if (i == origClass) continue;
                attrs.add(src.attribute(i));
            }
            for (int i = 0; i < nNoise; i++) {
                attrs.add(new Attribute("noise_" + i));
            }
            attrs.add(src.classAttribute());
            newClassIndex = attrs.size() - 1;

            Instances ins = new Instances(src.getRelationName() + "_noise" + nNoise,
                    attrs, 0);
            ins.setClassIndex(newClassIndex);
            augmentedHeader = new InstancesHeader(ins);
        }

        @Override public InstancesHeader getHeader()         { return augmentedHeader; }
        @Override public long estimatedRemainingInstances()  { return base.estimatedRemainingInstances(); }
        @Override public boolean hasMoreInstances()          { return base.hasMoreInstances(); }
        @Override public boolean isRestartable()             { return base.isRestartable(); }

        @Override
        public void restart() {
            base.restart();
            this.rng = new Random(seed);
        }

        @Override
        public Example<Instance> nextInstance() {
            Instance src = base.nextInstance().getData();
            int origClass = src.classIndex();
            double[] values = new double[augmentedHeader.numAttributes()];
            int idx = 0;
            for (int i = 0; i < src.numAttributes(); i++) {
                if (i == origClass) continue;
                values[idx++] = src.value(i);
            }
            for (int i = 0; i < nNoise; i++) {
                values[idx++] = rng.nextDouble();
            }
            values[newClassIndex] = src.classValue();

            Instance out = new DenseInstance(1.0, values);
            out.setDataset(augmentedHeader);
            return new InstanceExample(out);
        }

        @Override public void getDescription(StringBuilder sb, int indent) { sb.append("NoiseAugmentedStream"); }

        @Override
        protected void prepareForUseImpl(TaskMonitor m, ObjectRepository r) {
            if (base instanceof OptionHandler) {
                ((OptionHandler) base).prepareForUse();
            }
        }
    }
}