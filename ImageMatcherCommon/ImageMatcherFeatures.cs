using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using NLog;
using OpenCvSharp;
using OpenCvSharp.Flann;

namespace MaguSoft.ImageMatcherCommon;

public interface IImageMatcherFactory
{
    public IImageMatcher CreateMatcher(float maxFeatureDistance, int minGoodMatchesThreshold);
}

public class FastFlannBasedImageMatcherFactory : IImageMatcherFactory
{
    public IImageMatcher CreateMatcher(float maxFeatureDistance, int minGoodMatchesThreshold)
    {
        return new FlannBasedImageMatcher(maxFeatureDistance, minGoodMatchesThreshold, 6, 24, 1);
    }
}
public class SlowFlannBasedImageMatcherFactory : IImageMatcherFactory
{
    public IImageMatcher CreateMatcher(float maxFeatureDistance, int minGoodMatchesThreshold)
    {
        return new FlannBasedImageMatcher(maxFeatureDistance, minGoodMatchesThreshold, 12, 20, 2);
    }
}

public class BruteForceHammingImageMatcherFactory : IImageMatcherFactory
{
    public IImageMatcher CreateMatcher(float maxFeatureDistance, int minGoodMatchesThreshold)
    {
        return new BruteForceHammingImageMatcher(maxFeatureDistance, minGoodMatchesThreshold);
    }
}


public interface IImagesMatcher
{
    Task<List<ImageGroup>> FindSimilarImagesAsync(List<string> files);
}

public class ImagesMatcherFeatures : IImagesMatcher
{
    private static readonly ILogger _logger = LogManager.GetCurrentClassLogger();

    public int NumberOfFeaturesToExtract { get; }
    public float MaxFeatureDistance { get; }
    public int MinGoodMatchesThreshold { get; }

    private readonly IImageMatcherFactory _matcherFactory;

    public ImagesMatcherFeatures(FeaturesMatchingSettings settings)
    {
        _matcherFactory = settings.MatcherType switch
        {
            MatcherType.FlannBased => new FastFlannBasedImageMatcherFactory(),
            MatcherType.BruteForceHamming => new BruteForceHammingImageMatcherFactory(),
            _ => throw new ArgumentOutOfRangeException(nameof(settings.MatcherType), settings.MatcherType, null)
        };
        NumberOfFeaturesToExtract = settings.NumberOfFeaturesToExtract;
        MaxFeatureDistance = (float)settings.MaxFeatureDistance;
        MinGoodMatchesThreshold = settings.MinGoodMatchesThreshold;

        if (MinGoodMatchesThreshold > NumberOfFeaturesToExtract)
        {
            throw new ArgumentException("The number of matches cannot be greater than the number of features to extract.");
        }
    }

    private ImageGroup FindSimilarImages(
        Span<ImageFeature> featuresList,
        Stopwatch stopwatch,
        ref int processedCount)
    {
        var imgA = featuresList[0];
        var similarGroup = new List<SimilarityResult>();

        using var matcher = _matcherFactory.CreateMatcher(MaxFeatureDistance, MinGoodMatchesThreshold);

        for (int j = 1; j < featuresList.Length; j++)
        {
            var imgB = featuresList[j];

            // We verify the descriptors are not empty when adding to the list
            Debug.Assert(!imgA.Descriptors.Empty());
            Debug.Assert(!imgB.Descriptors.Empty());

            if (matcher.ImagesMatch(imgA, imgB, out int goodMatchesCount))
            {
                similarGroup.Add(new FeaturesSimilarityResult(imgB.FilePath, goodMatchesCount));
            }
        }

        int processedCountUpdated = Interlocked.Increment(ref processedCount);
        long milliseconds = stopwatch.ElapsedMilliseconds;
        float throughput = processedCountUpdated / (milliseconds / 1000.0f);
        _logger.Info(
            "Processing image {0}, throughput: {1:F2} images/sec",
            processedCountUpdated,
            throughput);

        return new ImageGroup(imgA.FilePath, similarGroup);
    }

    private ImageFeature? ExtractFeatures(string file, ORB orb)
    {
        try
        {
            // Read in grayscale to ignore color edits and speed up processing
            using var img = Cv2.ImRead(file, ImreadModes.Grayscale);
            if (img.Empty())
            {
                _logger.Warn("Failed to read image: {0}", file);
                return null;
            }

            var descriptors = new Mat();
            orb.DetectAndCompute(img, null, out var keypoints, descriptors);

            // Only save if the image actually had recognizable features
            if (!descriptors.Empty() && descriptors.Rows > 0)
            {
                // Console.WriteLine($"Extracted features from '{file}'...");
                return new ImageFeature(file, descriptors.Clone());
            }
            else
            {
                _logger.Warn("No recognizable features found in image: {0}", file);
                return null;
            }
        }
        catch (Exception ex)
        {
            _logger.Error(ex, "Failed to process {0}", file);
            return null;
        }
    }

    private async Task<List<ImageGroup>> FindSimilarImagesByFeaturesAsync(List<string> files)
    {
        _logger.Info(
            "Features to extract: {0}, max feature distance: {1}, min good matches: {2}. " +
            "Algorithm to use: {3}",
            NumberOfFeaturesToExtract,
            MaxFeatureDistance,
            MinGoodMatchesThreshold,
            _matcherFactory.GetType().Name);

        using var orb = ORB.Create(nFeatures: NumberOfFeaturesToExtract);

        _logger.Info("Indexing and extracting features from {0} images...", files.Count);

        // 1. Extract Features from all images
        var featuresList = (
            from f in files.AsParallel()
            let features = ExtractFeatures(f, orb)
            where features != null
            select features
            ).ToArray();

        _logger.Info("Comparing images for similarities...");

        var tasks = new List<Task<ImageGroup>>();
        int processedCount = 0;
        // 2. Compare features (O(N^2) comparison)
        Stopwatch stopwatch = Stopwatch.StartNew();
        for (int i = 0; i < featuresList.Length; i++)
        {
            Memory<ImageFeature> memorySegment = featuresList.AsMemory(i);
            var task = Task.Run(() =>
            {
                Span<ImageFeature> span = memorySegment.Span;
                return FindSimilarImages(span, stopwatch, ref processedCount);
            });
            tasks.Add(task);
        }

        var similarGroupsComplete = await Task.WhenAll(tasks);
        var similarGroupsFiltered = new List<ImageGroup>();
        HashSet<string> processedFiles = new HashSet<string>();
        foreach (var group in similarGroupsComplete)
        {
            if (processedFiles.Contains(group.MainImagePath))
            {
                continue;
            }
            if (group.SimilarityResults.Count > 0)
            {
                foreach (var result in group.SimilarityResults)
                {
                    processedFiles.Add(result.ImagePath);
                }
                similarGroupsFiltered.Add(group);
            }
        }

        // 3. Cleanup Unmanaged Resources
        foreach (var feature in featuresList)
        {
            feature.Dispose();
        }
        return similarGroupsFiltered;
    }

    public Task<List<ImageGroup>> FindSimilarImagesAsync(List<string> files)
        => FindSimilarImagesByFeaturesAsync(files);
}