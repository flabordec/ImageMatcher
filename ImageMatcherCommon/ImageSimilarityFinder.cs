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

    public float MaxFeatureDistance { get; }
    public int MinGoodMatchesThreshold { get; }

    private readonly IImageMatcherFactory _matcherFactory;

    public ImageSimilarityFinder(IImageMatcherFactory matcherFactory, float maxFeatureDistance, int minGoodMatchesThreshold)
    {
        _matcherFactory = matcherFactory;
        MaxFeatureDistance = maxFeatureDistance;
        MinGoodMatchesThreshold = minGoodMatchesThreshold;
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
        List<ImageFeature> featuresList,
        int i,
        Stopwatch stopwatch,
        ref int processedCount)
    {
        var imgA = featuresList[i];
        var similarGroup = new List<SimilarityResult>();

        using var matcher = _matcherFactory.CreateMatcher(MaxFeatureDistance, MinGoodMatchesThreshold);

        for (int j = i + 1; j < featuresList.Count; j++)
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

        int pc = Interlocked.Increment(ref processedCount);
        long milliseconds = stopwatch.ElapsedMilliseconds;
        float throughput = pc / (milliseconds / 1000.0f);
        _logger.Info(
            "Processing image {0}/{1} ({2}%), throughput: {3:F2} images/sec",
            pc,
            featuresList.Count,
            pc * 100 / featuresList.Count,
            throughput);

        return new ImageGroup(imgA.FilePath, similarGroup);
    }

    public async Task<List<ImageGroup>> RunImageSearch(
        string targetDirectory,
        bool recursive)
    {
        SearchOption searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;

        if (!Directory.Exists(targetDirectory))
        {
            throw new ArgumentException("Directory does not exist.");
        }

        var extensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };
        var files = Directory.GetFiles(targetDirectory, "*.*", searchOption)
                             .Where(f => extensions.Contains(Path.GetExtension(f).ToLower()))
                             .ToList();

        var featuresList = new List<ImageFeature>();

        // Initialize ORB detector. We pull up to 1000 top features per image.
        using var orb = ORB.Create(nFeatures: 1000);

        _logger.Info("Indexing and extracting features from {0} images...", files.Count);

        // 1. Extract Features from all images
        featuresList = (
            from f in files.AsParallel()
            let features = ExtractFeatures(f, orb)
            where features != null
            select features
            ).ToList();

        _logger.Info("Comparing images for similarities...");

        var tasks = new List<Task<ImageGroup>>();
        int processedCount = 0;
        // 2. Compare features (O(N^2) comparison)
        Stopwatch stopwatch = Stopwatch.StartNew();
        for (int i = 0; i < featuresList.Count; i++)
        {
            int index = i;
            tasks.Add(
                Task.Run(() =>
                    FindSimilarImages(
                        featuresList,
                        index,
                        stopwatch,
                        ref processedCount)));
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

