using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NLog;
using OpenCvSharp;
using OpenCvSharp.Flann;

namespace MaguSoft.ImageMatcherCommon;

public interface IImageMatcherFactory
{
    public IImageMatcher CreateMatcher(float maxFeatureDistance, int minGoodMatchesThreshold);
}

public class FlannBasedImageMatcherFactory : IImageMatcherFactory
{
    public IImageMatcher CreateMatcher(float maxFeatureDistance, int minGoodMatchesThreshold)
    {
        return new FlannBasedImageMatcher(maxFeatureDistance, minGoodMatchesThreshold);
    }
}

public class BruteForceHammingImageMatcherFactory : IImageMatcherFactory
{
    public IImageMatcher CreateMatcher(float maxFeatureDistance, int minGoodMatchesThreshold)
    {
        return new BruteForceHammingImageMatcher(maxFeatureDistance, minGoodMatchesThreshold);
    }
}

public class BothImageMatcherFactory : IImageMatcherFactory
{
    public IImageMatcher CreateMatcher(float maxFeatureDistance, int minGoodMatchesThreshold)
    {
        return new BothImageMatcher(maxFeatureDistance, minGoodMatchesThreshold);
    }
}
