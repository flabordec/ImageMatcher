
public enum MatcherType
{
    FlannBased,
    BruteForceHamming,
}
public record FeaturesMatchingSettings(MatcherType MatcherType, int NumberOfFeaturesToExtract, double MaxFeatureDistance, int MinGoodMatchesThreshold);
public record HashMatchingSettings(int SimilarityThreshold);

public record ImageMatchingSettings(FeaturesMatchingSettings FeaturesSettings, HashMatchingSettings HashSettings)
{
    public static ImageMatchingSettings Default => new(
        new FeaturesMatchingSettings(MatcherType.BruteForceHamming, 1000, 25.0, 120),
        new HashMatchingSettings(22)
    );
}
