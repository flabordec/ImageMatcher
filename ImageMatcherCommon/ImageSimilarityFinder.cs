using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NLog;
using OpenCvSharp;
using OpenCvSharp.Flann;

namespace MaguSoft.ImageMatcherCommon;

public class ImageSimilarityFinder
{
    private static readonly ILogger _logger = LogManager.GetCurrentClassLogger();

    public int NumberOfFeaturesToExtract { get; }
    public float MaxFeatureDistance { get; }
    public int MinGoodMatchesThreshold { get; }

    private readonly IImageMatcherFactory _matcherFactory;

    public ImageSimilarityFinder(IImageMatcherFactory matcherFactory, int numberOfFeaturesToExtract, float maxFeatureDistance, int minGoodMatchesThreshold)
    {
        _matcherFactory = matcherFactory;
        NumberOfFeaturesToExtract = numberOfFeaturesToExtract;
        MaxFeatureDistance = maxFeatureDistance;
        MinGoodMatchesThreshold = minGoodMatchesThreshold;
        if (MinGoodMatchesThreshold > NumberOfFeaturesToExtract)
        {
            throw new ArgumentException("The number of matches cannot be greater than the number of features to extract.");
        }
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
                similarGroup.Add(new SimilarityResult(imgB.FilePath, goodMatchesCount));
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

    public async Task<List<ImageGroup>> RunImageSearch(
        string targetDirectory,
        bool recursive)
    {
        _logger.Info(
            "Starting image search in {0}. " +
            "Features to extract: {1}, max feature distance: {2}, min good matches: {3}. " +
            "Algorithm to use: {4}",
            targetDirectory,
            NumberOfFeaturesToExtract,
            MaxFeatureDistance,
            MinGoodMatchesThreshold,
            _matcherFactory.GetType().Name);

        SearchOption searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;

        if (!Directory.Exists(targetDirectory))
        {
            throw new ArgumentException("Directory does not exist.");
        }

        var extensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };
        var files = Directory.GetFiles(targetDirectory, "*.*", searchOption)
                             .Where(f => extensions.Contains(Path.GetExtension(f).ToLower()))
                             .ToList();

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

        _logger.Info("Processing complete.");

        return similarGroupsFiltered;
    }
}

