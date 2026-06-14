using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NLog;
using OpenCvSharp;
using OpenCvSharp.Flann;

namespace MaguSoft.ImageMatcherCommon;

public interface IImageMatcher : IDisposable
{
    public bool ImagesMatch(ImageFeature imgA, ImageFeature imgB, out int goodMatchesCount);
}

public abstract class ImageMatcherBase : IImageMatcher
{
    public float MaxFeatureDistance { get; }
    public int MinGoodMatchesThreshold { get; }

    private bool _disposed;

    public ImageMatcherBase(float maxFeatureDistance, int minGoodMatchesThreshold)
    {
        MaxFeatureDistance = maxFeatureDistance;
        MinGoodMatchesThreshold = minGoodMatchesThreshold;
    }

    public abstract bool ImagesMatch(ImageFeature imgA, ImageFeature imgB, out int goodMatchesCount);

    protected int InnerGetGoodMatchesCount(
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

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            _disposed = true;
        }
    }

    public void Dispose()
    {
        // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}

public class FlannBasedImageMatcher : ImageMatcherBase
{
    private readonly LshIndexParams _indexParams;
    private readonly SearchParams _searchParams;
    private readonly FlannBasedMatcher _flannMatcher;

    public FlannBasedImageMatcher(float maxFeatureDistance, int minGoodMatchesThreshold) : base(maxFeatureDistance, minGoodMatchesThreshold)
    {
        // Flann-based Matcher for efficient matching of ORB descriptors
        // - table_number: Number of hash tables to use
        // - key_size: Length of the key in bits -- kind of like zip code
        // - multi_probe_level (2): Number of levels to probe for neighboring buckets
        _indexParams = new LshIndexParams(6, 24, 1);
        // - checks: How many times the tree(s) should be recursively traversed. 
        // Higher = more accurate matches, but slower.
        _searchParams = new SearchParams(50);
        _flannMatcher = new FlannBasedMatcher(_indexParams, _searchParams);
    }

    public override bool ImagesMatch(ImageFeature imgA, ImageFeature imgB, out int goodMatchesCount)
    {
        goodMatchesCount = InnerGetGoodMatchesCount(_flannMatcher, imgA, imgB);
        return goodMatchesCount >= MinGoodMatchesThreshold;
    }

    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);
        if (disposing)
        {
            _indexParams?.Dispose();
            _searchParams?.Dispose();
            _flannMatcher?.Dispose();
        }
    }
}

public class BruteForceHammingImageMatcher : ImageMatcherBase
{
    private readonly BFMatcher _bfMatcher;

    public BruteForceHammingImageMatcher(float maxFeatureDistance, int minGoodMatchesThreshold) : base(maxFeatureDistance, minGoodMatchesThreshold)
    {
        // Brute Force Matcher using Hamming distance (best for ORB binary descriptors)
        _bfMatcher = new BFMatcher(NormTypes.Hamming, crossCheck: true);
    }

    public override bool ImagesMatch(ImageFeature imgA, ImageFeature imgB, out int goodMatchesCount)
    {
        goodMatchesCount = InnerGetGoodMatchesCount(_bfMatcher, imgA, imgB);
        return goodMatchesCount >= MinGoodMatchesThreshold;
    }

    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);
        if (disposing)
        {
            _bfMatcher?.Dispose();
        }
    }
}

public class BothImageMatcher : ImageMatcherBase
{
    private readonly FlannBasedImageMatcher _flannMatcher;
    private readonly BruteForceHammingImageMatcher _bfMatcher;

    public BothImageMatcher(float maxFeatureDistance, int minGoodMatchesThreshold) : base(maxFeatureDistance, minGoodMatchesThreshold)
    {
        _flannMatcher = new FlannBasedImageMatcher(maxFeatureDistance, minGoodMatchesThreshold);
        _bfMatcher = new BruteForceHammingImageMatcher(maxFeatureDistance, minGoodMatchesThreshold);
    }

    public override bool ImagesMatch(ImageFeature imgA, ImageFeature imgB, out int goodMatchesCount)
    {
        return
            _flannMatcher.ImagesMatch(imgA, imgB, out goodMatchesCount) &&
            _bfMatcher.ImagesMatch(imgA, imgB, out goodMatchesCount);
    }

    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);
        if (disposing)
        {
            _flannMatcher?.Dispose();
            _bfMatcher?.Dispose();
        }
    }
}