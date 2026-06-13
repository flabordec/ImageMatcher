using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NLog;
using OpenCvSharp;
using OpenCvSharp.Flann;

namespace MaguSoft.ImageMatcherCommon;

public record ImageFeature(string FilePath, Mat Descriptors) : IDisposable
{
    public void Dispose()
    {
        Descriptors?.Dispose();
    }
}

public record SimilarityResult(string ImagePath, int GoodMatchesFlann, int GoodMatchesBf);

public record ImageGroup(string MainImagePath, List<SimilarityResult> SimilarityResults);

public class ImageSimilarityFinder
{
    private static readonly ILogger _logger = LogManager.GetCurrentClassLogger();

    public float MaxFeatureDistance { get; set; } = 25f;
    public int MinGoodMatchesThreshold { get; set; } = 120;

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

    private int ImagesMatch(
        DescriptorMatcher matcher,
        ImageFeature imgA,
        ImageFeature imgB)
    {
        // Match descriptors
        var matches = matcher.Match(imgA.Descriptors, imgB.Descriptors);

        // Filter for only high-quality matches (robust against lighting changes)
        int goodMatchesCount = matches.Count(m => m.Distance < MaxFeatureDistance);

        // If enough features match, they are similar (robust against cropping/resizing)
        return goodMatchesCount;
    }

    private ImageGroup FindSimilarImages(
        FlannBasedMatcher flannMatcher,
        BFMatcher bfMatcher,
        List<ImageFeature> featuresList,
        int i,
        ref int processedCount)
    {
        var imgA = featuresList[i];
        var similarGroup = new List<SimilarityResult>();

        for (int j = i + 1; j < featuresList.Count; j++)
        {
            var imgB = featuresList[j];

            // We verify the descriptors are not empty when adding to the list
            Debug.Assert(!imgA.Descriptors.Empty());
            Debug.Assert(!imgB.Descriptors.Empty());

            int goodMatchesFlann = ImagesMatch(flannMatcher, imgA, imgB);
            if (goodMatchesFlann >= MinGoodMatchesThreshold)
            {
                int goodMatchesBf = ImagesMatch(bfMatcher, imgA, imgB);
                if (goodMatchesBf >= MinGoodMatchesThreshold)
                {
                    similarGroup.Add(new SimilarityResult(imgB.FilePath, goodMatchesFlann, goodMatchesBf));
                }
            }
        }

        int pc = Interlocked.Increment(ref processedCount);
        _logger.Info(
            "Processing image {0}/{1} ({2}%)",
            pc,
            featuresList.Count,
            pc * 100 / featuresList.Count);

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
        // Brute Force Matcher using Hamming distance (best for ORB binary descriptors)
        using var bfMatcher = new BFMatcher(NormTypes.Hamming, crossCheck: true);

        // Flann-based Matcher for efficient matching of ORB descriptors
        // - table_number: Number of hash tables to use
        // - key_size: Length of the key in bits -- kind of like zip code
        // - multi_probe_level (2): Number of levels to probe for neighboring buckets
        using var indexParams = new LshIndexParams(6, 24, 1);
        // - checks: How many times the tree(s) should be recursively traversed. 
        // Higher = more accurate matches, but slower.
        using var searchParams = new SearchParams(50);
        using var flannMatcher = new FlannBasedMatcher(indexParams, searchParams);

        var tasks = new List<Task<ImageGroup>>();
        int processedCount = 0;
        // 2. Compare features (O(N^2) comparison)
        for (int i = 0; i < featuresList.Count; i++)
        {
            int index = i;
            tasks.Add(
                Task.Run(() =>
                    FindSimilarImages(
                        flannMatcher,
                        bfMatcher,
                        featuresList,
                        index,
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

