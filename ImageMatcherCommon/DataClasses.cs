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

public abstract record SimilarityResult(string ImagePath)
{
    public abstract string DisplayText { get; }
}
public record FeaturesSimilarityResult(string ImagePath, int GoodMatchesCount) : SimilarityResult(ImagePath)
{
    public override string DisplayText => $"Image: {ImagePath}, Good Matches: {GoodMatchesCount}";
}
public record HashSimilarityResult(string ImagePath, int Distance) : SimilarityResult(ImagePath)
{
    public override string DisplayText => $"Image: {ImagePath}, Distance: {Distance}";
}

public record ImageGroup(string MainImagePath, List<SimilarityResult> SimilarityResults);
