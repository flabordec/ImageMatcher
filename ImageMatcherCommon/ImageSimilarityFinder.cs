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
        _logger.Info("Starting image search in {0}.", targetDirectory);

        SearchOption searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;

        if (!Directory.Exists(targetDirectory))
        {
            throw new ArgumentException("Directory does not exist.");
        }

        var extensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };
        var files = Directory.GetFiles(targetDirectory, "*.*", searchOption)
                             .Where(f => extensions.Contains(Path.GetExtension(f).ToLower()))
                             .ToList();

        List<ImageGroup> similarGroups = new();
        var similarGroupsNotValidated = await _hashMatcher.FindSimilarImagesAsync(files);
        foreach (var group in similarGroupsNotValidated)
        {
            var similarGroupValidated = await _featuresMatcher.FindSimilarImagesAsync(group.SimilarityResults.Select(r => r.ImagePath).ToList());
            similarGroups.AddRange(similarGroupValidated);
        }

        _logger.Info("Processing complete.");

        return similarGroups;
    }

}

