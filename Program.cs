using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Flann;

namespace ImageSimilarityFinder;

record ImageFeature(string FilePath, Mat Descriptors) : IDisposable
{
    public void Dispose()
    {
        Descriptors?.Dispose();
    }
}


class Program
{
    // Wrapper class to safely store and dispose of the Unmanaged OpenCV Mat objects


    private static ImageFeature? ExtractFeatures(string file, ORB orb)
    {
        try
        {
            // Read in grayscale to ignore color edits and speed up processing
            using var img = Cv2.ImRead(file, ImreadModes.Grayscale);
            if (img.Empty())
            {
                Console.WriteLine($"[Warning] Failed to read image: '{file}'");
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
                Console.WriteLine($"[Warning] No recognizable features found in image: '{file}'");
                return null;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Error] Failed to process '{file}': {ex.Message}");
            return null;
        }
    }

    private static bool ImagesMatch(
        DescriptorMatcher matcher,
        ImageFeature imgA,
        ImageFeature imgB,
        float maxFeatureDistance,
        int minGoodMatchesThreshold)
    {
        // Match descriptors
        var matches = matcher.Match(imgA.Descriptors, imgB.Descriptors);

        // Filter for only high-quality matches (robust against lighting changes)
        int goodMatchesCount = matches.Count(m => m.Distance < maxFeatureDistance);

        // If enough features match, they are similar (robust against cropping/resizing)
        return goodMatchesCount >= minGoodMatchesThreshold;
    }

    static void Main(string[] args)
    {
        // --- CONFIGURATION ---
        string targetDirectory = @"S:\pictures\sorted\Edens Zero";

        // How many matching keypoints are required to consider images "similar".
        // Increase this to reduce false positives; decrease to catch heavily cropped images.
        int minGoodMatchesThreshold = 120;

        // Maximum distance (error) allowed between two features to consider them a match
        float maxFeatureDistance = 25f;
        // ---------------------

        if (!Directory.Exists(targetDirectory))
        {
            Console.WriteLine("Directory does not exist.");
            return;
        }

        var extensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };
        var files = Directory.GetFiles(targetDirectory, "*.*", SearchOption.AllDirectories)
                             .Where(f => extensions.Contains(Path.GetExtension(f).ToLower()))
                             .ToList();

        var featuresList = new List<ImageFeature>();

        // Initialize ORB detector. We pull up to 1000 top features per image.
        using var orb = ORB.Create(nFeatures: 1000);

        Console.WriteLine($"Indexing and extracting features from {files.Count} images...");

        // 1. Extract Features from all images
        featuresList = (
            from f in files.AsParallel()
            let features = ExtractFeatures(f, orb)
            where features != null
            select features
            ).ToList();

        Console.WriteLine("\nComparing images for similarities...");
        var processedFiles = new HashSet<string>();

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

        var similarGroups = new List<List<string>>();
        // 2. Compare features (O(N^2) comparison)
        for (int i = 0; i < featuresList.Count; i++)
        {
            Console.WriteLine($"\rProcessing image {i + 1}/{featuresList.Count} ({(i + 1) * 100 / featuresList.Count}%)");

            var imgA = featuresList[i];
            if (processedFiles.Contains(imgA.FilePath))
            {
                continue;
            }

            var similarGroup = new List<string>();

            for (int j = i + 1; j < featuresList.Count; j++)
            {
                var imgB = featuresList[j];
                if (processedFiles.Contains(imgB.FilePath))
                {
                    continue;
                }

                // We verify the descriptors are not empty when adding to the list
                Debug.Assert(!imgA.Descriptors.Empty());
                Debug.Assert(!imgB.Descriptors.Empty());

                if (ImagesMatch(flannMatcher, imgA, imgB, maxFeatureDistance, minGoodMatchesThreshold)
                    && ImagesMatch(bfMatcher, imgA, imgB, maxFeatureDistance, minGoodMatchesThreshold))
                {
                    similarGroup.Add(imgB.FilePath);
                    processedFiles.Add(imgB.FilePath);
                }
            }

            // Print the group if similarities were found
            if (similarGroup.Any())
            {
                similarGroup.Add(imgA.FilePath);
                similarGroups.Add(similarGroup);
            }
            processedFiles.Add(imgA.FilePath);
        }

        foreach (var group in similarGroups)
        {
            Console.WriteLine($"\n[Match Group Found]");
            foreach (var file in group)
            {
                Console.WriteLine($" -> {file}");
            }
        }

        // 3. Cleanup Unmanaged Resources
        foreach (var feature in featuresList)
        {
            feature.Dispose();
        }

        Console.WriteLine("\nProcessing complete.");
    }
}
