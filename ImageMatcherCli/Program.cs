using System;
using System.Collections.Generic;
using System.CommandLine;
using System.Diagnostics;
using System.IO;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Flann;
using MaguSoft.ImageMatcherCommon;

namespace MaguSoft.ImageMatcherCli;

enum MatcherType
{
    FlannBased,
    BruteForceHamming,
    Both,
}

class Program
{
    // Wrapper class to safely store and dispose of the Unmanaged OpenCV Mat objects

    private static void ConfigureLogging()
    {
        var config = new NLog.Config.LoggingConfiguration();

        var logfile = new NLog.Targets.FileTarget("logfile") { FileName = "file.txt" };
        var logconsole = new NLog.Targets.ConsoleTarget("logconsole");

        config.AddRule(NLog.LogLevel.Info, NLog.LogLevel.Fatal, logconsole);
        config.AddRule(NLog.LogLevel.Debug, NLog.LogLevel.Fatal, logfile);

        NLog.LogManager.Configuration = config;
    }

    static async Task<int> Main(string[] args)
    {
        ConfigureLogging();
        var directoryArgument = new Argument<string>("directory")
        {
            Description = "The directory to search.",
        };

        var recursiveOption = new Option<bool>("--recursive")
        {
            Description = "Whether the search is recursive or not.",
            DefaultValueFactory = r => false,
            Aliases = { "-r" }
        };

        var numberOfFeaturesToExtractOption = new Option<int>("--number-of-features-to-extract")
        {
            Description = "The number of features to extract from each image.",
            DefaultValueFactory = r => 1000
        };

        var minKeypointMatchesOption = new Option<int>("--min-keypoint-matches")
        {
            Description = "The minimum number of keypoint matches to consider two images similar.",
            DefaultValueFactory = r => 120
        };

        var maxFeatureDistanceOption = new Option<float>("--max-feature-distance")
        {
            Description = "Maximum distance (error) to consider two features a match.",
            DefaultValueFactory = r => 25f
        };

        var matcherTypeOption = new Option<MatcherType>("--matcher-type")
        {
            Description = "The type of matcher to use. FlannBased is the fastest, BruteForceHamming is more accurate.",
            DefaultValueFactory = r => MatcherType.Both
        };

        // 3. Assemble the Root Command
        var rootCommand = new RootCommand("Image similarity search and keypoint matching tool.");
        rootCommand.Arguments.Add(directoryArgument);
        rootCommand.Options.Add(recursiveOption);
        rootCommand.Options.Add(minKeypointMatchesOption);
        rootCommand.Options.Add(maxFeatureDistanceOption);
        rootCommand.Options.Add(matcherTypeOption);
        rootCommand.Options.Add(numberOfFeaturesToExtractOption);
        rootCommand.SetAction(
            pr => RunImageSearch(
                pr.GetRequiredValue(directoryArgument),
                pr.GetValue(recursiveOption),
                pr.GetValue(numberOfFeaturesToExtractOption),
                pr.GetValue(minKeypointMatchesOption),
                pr.GetValue(maxFeatureDistanceOption),
                pr.GetValue(matcherTypeOption)));

        try
        {
            return await rootCommand.Parse(args).InvokeAsync();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static async Task RunImageSearch(
        string targetDirectory,
        bool recursive,
        int numberOfFeaturesToExtract,
        int minGoodMatchesThreshold,
        float maxFeatureDistance,
        MatcherType matcherType)
    {
        IImageMatcherFactory matcherFactory;
        switch (matcherType)
        {
            case MatcherType.FlannBased:
                matcherFactory = new FastFlannBasedImageMatcherFactory();
                break;
            case MatcherType.BruteForceHamming:
                matcherFactory = new BruteForceHammingImageMatcherFactory();
                break;
            case MatcherType.Both:
                matcherFactory = new BothImageMatcherFactory();
                break;
            default:
                throw new ArgumentException($"Invalid matcher type: {matcherType}.");
        }

        var similarityFinder = new ImageSimilarityFinder(matcherFactory, numberOfFeaturesToExtract, maxFeatureDistance, minGoodMatchesThreshold);

        var similarGroupsFiltered = await similarityFinder.RunImageSearch(targetDirectory, recursive);

        foreach (var group in similarGroupsFiltered)
        {
            Console.WriteLine($"\n[Match Group Found]");
            Console.WriteLine($" -> '{group.MainImagePath}'");
            foreach (var result in group.SimilarityResults)
            {
                Console.WriteLine($" -> '{result.ImagePath}' (good matches: {result.GoodMatchesCount})");
            }
        }
        Console.WriteLine("Search complete");
    }
}


