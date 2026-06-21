using System.Collections.Concurrent;
using System.Numerics;
using NLog;
using Blake3;
using System.Collections.Immutable;

namespace MaguSoft.ImageMatcherCommon;

public class ImagesMatcherBinary : IImagesMatcher
{
    private static readonly ILogger _logger = LogManager.GetCurrentClassLogger();

    public ImagesMatcherBinary()
    {
    }

    private async Task<List<ImageGroup>> FindSimilarImagesByHashAsync(List<string> files)
    {
        var imageHashes = new ConcurrentBag<(string Path, Hash Hash)>();
        int processedCount = 0;

        Parallel.ForEach(
            files,
            new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
            file =>
            {
                byte[] data = File.ReadAllBytes(file);
                var hash = Hasher.Hash(data);
                imageHashes.Add((file, hash));

                int currentCount = Interlocked.Increment(ref processedCount);
                if (currentCount % 100 == 0)
                {
                    _logger.Info($"Processed {currentCount} images...");
                }
            });

        ImmutableDictionary<Hash, List<string>> hashGroups = imageHashes
            .GroupBy(x => x.Hash)
            .ToImmutableDictionary(g => g.Key, g => g.Select(x => x.Path).ToList());

        List<ImageGroup> imageGroups = new();
        foreach (var group in hashGroups)
        {
            var mainImage = group.Value.First();
            var similarImages = group.Value.Skip(1)
                .Select(p => new BinarySimilarityResult(p))
                .Cast<SimilarityResult>()
                .ToList();
            imageGroups.Add(new ImageGroup(mainImage, similarImages));
        }
        return imageGroups;
    }

    public Task<List<ImageGroup>> FindSimilarImagesAsync(List<string> files)
        => FindSimilarImagesByHashAsync(files);
}