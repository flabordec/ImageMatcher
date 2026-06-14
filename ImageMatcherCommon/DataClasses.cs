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

public record SimilarityResult(string ImagePath, int GoodMatchesCount);

public record ImageGroup(string MainImagePath, List<SimilarityResult> SimilarityResults);
