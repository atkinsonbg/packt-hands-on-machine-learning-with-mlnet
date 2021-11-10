using System;
using System.IO;

using Chapter03.App.ML.Base;
using Chapter03.App.ML.Objects;

using Microsoft.ML;

namespace Chapter03.App.ML
{
    public class Predictor : BaseML
    {
        public void Predict(string inputDataFile)
        {
            if (!File.Exists(ModelPath))
            {
                Console.WriteLine($"Failed to find model at {ModelPath}");

                return;
            }

            if (!File.Exists(inputDataFile))
            {
                Console.WriteLine($"Failed to find input data at {inputDataFile}");

                return;
            }

            ITransformer mlModel;

            using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                mlModel = MlContext.Model.Load(stream, out _);
            }

            if (mlModel == null)
            {
                Console.WriteLine("Failed to load model");

                return;
            }

            var predictionEngine = MlContext.Model.CreatePredictionEngine<FileInput, FilePrediction>(mlModel);

            var prediction = predictionEngine.Predict(new FileInput
            {
                Strings = GetStrings(File.ReadAllBytes(inputDataFile))
            });

            Console.WriteLine(
                                $"Based on the file ({inputDataFile}) the file is classified as {(prediction.IsMalicious ? "malicious" : "benign")}" +
                                $" at a confidence level of {prediction.Probability:P0}");
        }
    }
}