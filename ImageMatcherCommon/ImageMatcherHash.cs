using System.Collections.Concurrent;
using System.Numerics;
using NLog;
using SkiaSharp;

namespace MaguSoft.ImageMatcherCommon;

public class ImagesMatcherHash : IImagesMatcher
{
    private static readonly ILogger _logger = LogManager.GetCurrentClassLogger();

    private readonly int _similarityThreshold;

    public ImagesMatcherHash(HashMatchingSettings settings)
    {
        _similarityThreshold = settings.SimilarityThreshold;
    }

    private async Task<List<ImageGroup>> FindSimilarImagesByHashAsync(List<string> files)
    {
        _logger.Info("Similarity threshold: {0}", _similarityThreshold);


        var imageHashes = new ConcurrentBag<(string Path, ulong Hash)>();
        int processedCount = 0;

        Parallel.ForEach(files, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, file =>
        {
            ulong? hash = ComputeDHash(file);
            if (hash.HasValue)
            {
                imageHashes.Add((file, hash.Value));
            }

            int currentCount = Interlocked.Increment(ref processedCount);
            if (currentCount % 100 == 0)
            {
                _logger.Info($"Processed {currentCount} / {files.Count} images...");
            }
        });

        var hashList = imageHashes.ToList();
        _logger.Info($"\nSuccessfully hashed {hashList.Count} images. Grouping similarities...");

        return GroupSimilarImages(hashList);
    }

    /// <summary>
    /// Computes the Difference Hash (dHash) using SkiaSharp.
    /// </summary>
    private ulong? ComputeDHash(string filePath)
    {
        try
        {
            // Decode the image directly into memory
            using var original = SKBitmap.Decode(filePath);
            if (original == null) return null;

            // Resize to 9x8. Nearest-neighbor routing, is extremely fast
            // and perfect for perceptual hashing.
            var info = new SKImageInfo(9, 8);
            using var resized = original.Resize(info, new SKSamplingOptions(SKFilterMode.Nearest));
            if (resized == null) return null;

            ulong hash = 0;
            int bitIndex = 0;

            // Compare adjacent pixels
            for (int y = 0; y < 8; y++)
            {
                for (int x = 0; x < 8; x++)
                {
                    var leftColor = resized.GetPixel(x, y);
                    var rightColor = resized.GetPixel(x + 1, y);

                    // Convert RGB to a fast Grayscale luminosity value
                    int leftLuma = (leftColor.Red * 299 + leftColor.Green * 587 + leftColor.Blue * 114);
                    int rightLuma = (rightColor.Red * 299 + rightColor.Green * 587 + rightColor.Blue * 114);

                    // If the left pixel is brighter than the right, set the bit to 1
                    if (leftLuma > rightLuma)
                    {
                        hash |= (1UL << bitIndex);
                    }
                    bitIndex++;
                }
            }
            return hash;
        }
        catch
        {
            // Ignore files that are locked, corrupt, or unsupported
            return null;
        }
    }

    /// <summary>
    /// Groups images based on the Hamming distance of their hashes.
    /// </summary>
    private List<ImageGroup> GroupSimilarImages(List<(string Path, ulong Hash)> hashes)
    {
        var clusters = new List<ImageGroup>();
        var visited = new HashSet<int>();

        for (int i = 0; i < hashes.Count; i++)
        {
            if (visited.Contains(i)) continue;

            var currentCluster = new List<SimilarityResult>();
            visited.Add(i);

            for (int j = i + 1; j < hashes.Count; j++)
            {
                if (visited.Contains(j)) continue;

                // BitOperations.PopCount translates to a native hardware instruction (POPCNT)
                // making this comparison almost instantaneous.
                ulong xorResult = hashes[i].Hash ^ hashes[j].Hash;
                int distance = BitOperations.PopCount(xorResult);

                if (distance <= _similarityThreshold)
                {
                    currentCluster.Add(new HashSimilarityResult(hashes[j].Path, distance));
                    visited.Add(j);
                }
            }

            if (currentCluster.Count > 1)
            {
                clusters.Add(new ImageGroup(hashes[i].Path, currentCluster));
            }
        }

        return clusters;
    }

    public Task<List<ImageGroup>> FindSimilarImagesAsync(List<string> files)
        => FindSimilarImagesByHashAsync(files);
}