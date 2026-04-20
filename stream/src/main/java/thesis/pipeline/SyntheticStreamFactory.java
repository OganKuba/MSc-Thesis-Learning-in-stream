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
        SEAGenerator g1 = newSEA(seed,     1, 7.0);
        SEAGenerator g2 = newSEA(seed + 1, 2, 8.0);
        SEAGenerator g3 = newSEA(seed + 2, 3, 9.0);
        SEAGenerator g4 = newSEA(seed + 3, 4, 9.5);

        ConceptDriftStream s34 = newDrift(g3, g4, 75_000, 1, seed);
        ConceptDriftStream s234 = newDrift(g2, s34, 50_000, 1, seed + 10);
        ConceptDriftStream s1234 = newDrift(g1, s234, 25_000, 1, seed + 20);
        s1234.prepareForUse();
        return s1234;
    }

    private static SEAGenerator newSEA(int seed, int function, double threshold) {
        SEAGenerator g = new SEAGenerator();
        g.instanceRandomSeedOption.setValue(seed);
        g.functionOption.setValue(function);
        try {
            g.getOptions().getOption('t').setValueViaCLIString(String.valueOf(threshold));
        } catch (Exception ignored) { }
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
        return g;
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
        return g;
    }


    public static InstanceStream createSTAGGER(int seed, int numInstances) {
        STAGGERGenerator s1 = newStagger(seed,     1);
        STAGGERGenerator s2 = newStagger(seed + 1, 2);
        STAGGERGenerator s3 = newStagger(seed + 2, 3);
        STAGGERGenerator s1b = newStagger(seed + 3, 1);

        ConceptDriftStream d3 = newDrift(s3, s1b, 60_000, 1, seed + 30);
        ConceptDriftStream d2 = newDrift(s2, d3,  40_000, 1, seed + 40);
        ConceptDriftStream d1 = newDrift(s1, d2,  20_000, 1, seed + 50);
        d1.prepareForUse();
        return d1;
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
        return g;
    }


    public static InstanceStream addNoiseFeatures(InstanceStream stream, int nNoise) {
        if (nNoise <= 0) return stream;
        return new NoiseAugmentedStream(stream, nNoise);
    }


    private static ConceptDriftStream newDrift(InstanceStream a, InstanceStream b,
                                               int position, int width, int seed) {
        ConceptDriftStream d = new ConceptDriftStream();
        d.streamOption.setCurrentObject(a);
        d.driftstreamOption.setCurrentObject(b);
        d.positionOption.setValue(position);
        d.widthOption.setValue(width);
        d.randomSeedOption.setValue(seed);
        return d;
    }


    private static final class NoiseAugmentedStream extends AbstractOptionHandler
            implements InstanceStream {

        private final InstanceStream base;
        private final int nNoise;
        private final Random rng;
        private InstancesHeader augmentedHeader;
        private int newClassIndex;

        NoiseAugmentedStream(InstanceStream base, int nNoise) {
            this.base = base;
            this.nNoise = nNoise;
            this.rng = new Random(42);
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
        @Override public void restart()                      { base.restart(); }

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
        @Override protected void prepareForUseImpl(TaskMonitor m, ObjectRepository r) { /* no-op */ }
    }
}