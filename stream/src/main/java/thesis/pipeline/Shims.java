package thesis.pipeline;

import moa.streams.InstanceStream;
import thesis.detection.TwoLevelDriftDetector;
import thesis.models.ModelWrapper;
import thesis.selection.FeatureSelector;

final class DatasetFactory {
    static InstanceStream create(String name, int seed) {
        // TODO: map "yahoo_finance", "nyc_taxi", "synthetic_sea", … → your loader
        throw new UnsupportedOperationException("wire dataset: " + name);
    }
}
final class ModelFactory {
    static ModelWrapper create(String name, int seed) {
        // TODO: HT, ARF, SRP, DA_SRP_AB, DA_SRP_ABC → your ModelWrapper subclasses
        throw new UnsupportedOperationException("wire model: " + name);
    }
}
final class SelectorFactory {
    static FeatureSelector create(String name, int seed) {
        // TODO: S1, S2, S4 → your FeatureSelector implementations
        throw new UnsupportedOperationException("wire selector: " + name);
    }
}
final class DetectorFactory {
    static TwoLevelDriftDetector create(String name, int seed) {
        // TODO: "ADWIN", "TWO_LEVEL" → your detector configurations
        throw new UnsupportedOperationException("wire detector: " + name);
    }
}