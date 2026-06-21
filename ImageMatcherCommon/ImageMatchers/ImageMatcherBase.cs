namespace MaguSoft.ImageMatcherCommon.ImageMatchers;


public record HashCalcualted(string FilePath) : ProgressEvent()
{
    public override string GetDisplayString() => $"Calculated hash for file: {FilePath}";
}

public record ImageProcessed(string FilePath) : ProgressEvent()
{
    public override string GetDisplayString() => $"Processed file: {FilePath}";
}
public abstract record ProgressEvent()
{
    public abstract string GetDisplayString();
}

public interface IImagesMatcher
{
    Task<List<ImageGroup>> FindSimilarImagesAsync(List<string> files, IProgress<ProgressEvent> progress);
}