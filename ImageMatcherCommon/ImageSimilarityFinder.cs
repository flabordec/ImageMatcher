using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using MaguSoft.ImageMatcherCommon.ImageMatchers;
using NLog;
using OpenCvSharp;
using OpenCvSharp.Flann;

namespace MaguSoft.ImageMatcherCommon;

public class ImageSimilarityFinder
{
    private static readonly ILogger _logger = LogManager.GetCurrentClassLogger();

    private readonly IImagesMatcher _hashMatcher;
    private readonly IImagesMatcher _featuresMatcher;

    public ImageSimilarityFinder(ImageMatchingSettings settings)
    {
        _hashMatcher = new ImagesMatcherHash(settings.HashSettings);
        _featuresMatcher = new ImagesMatcherFeatures(settings.FeaturesSettings);
    }

    public async Task<List<ImageGroup>> RunImageSearch(
        string targetDirectory,
        bool recursive)
    {
        Stopwatch stopwatch = Stopwatch.StartNew();

        _logger.Info("Starting image search in {0}.", targetDirectory);

        SearchOption searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;

        if (!Directory.Exists(targetDirectory))
        {
            throw new ArgumentException("Directory does not exist.");
        }

        var extensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };
        var files = Directory
            .GetFiles(targetDirectory, "*.*", searchOption)
            .Where(f => extensions.Contains(Path.GetExtension(f)))
            .ToList();

        int imagesProcessed = 0;
        int totalFiles = files.Count;
        Progress<ProgressEvent> progress = new();
        progress.ProgressChanged += (sender, e) => 
        {
            int step = totalFiles / 100;
            if (step == 0) step = 1;
            if (e is ImageProcessed) 
            {
                int actualImagesProcessed = Interlocked.Increment(ref imagesProcessed);
                if (actualImagesProcessed % step == 0)
                {
                    _logger.Info(
                        "Image processed {0}/{1}, {2:0.00}%", 
                        actualImagesProcessed, 
                        totalFiles, 
                        (double)actualImagesProcessed / totalFiles * 100);
                }
            }
        };

        List<ImageGroup> similarGroups = new();
        var similarGroupsNotValidated = await _hashMatcher.FindSimilarImagesAsync(files, progress);
        foreach (var group in similarGroupsNotValidated)
        {
            var similarImagesInGroup = group.SimilarityResults.Select(r => r.ImagePath).ToList();
            imagesProcessed = 0;
            totalFiles = similarImagesInGroup.Count;

            var similarGroupValidated = await _featuresMatcher.FindSimilarImagesAsync(similarImagesInGroup, progress);
            similarGroups.AddRange(similarGroupValidated);
        }

        stopwatch.Stop();
        _logger.Info("Processed {0} images in {1} ms.", files.Count, stopwatch.ElapsedMilliseconds);

        return similarGroups;
    }

}

