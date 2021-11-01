using Chapter03.App.Ml.Base;
using Chapter03.App.Ml.Objects;
using Microsoft.ML;
using System.Text.Json;

namespace Chapter03.App.Ml
{
    public class Predictor : BaseML
    {
        public void Predict(string inputDataFile)
        {
            if (!File.Exists(inputDataFile))
            {
                Console.WriteLine($"Failed to find input data at path {inputDataFile}");
                return;
            }

            if (!File.Exists(ModelPath))
            {
                Console.WriteLine($"Failed to find model at path {ModelPath}");
                return;
            }

            ITransformer mlModel;

            using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                mlModel = mlContext.Model.Load(stream, out _);
            }

            if (mlModel == null)
            {
                Console.WriteLine("Failed to load ML model");
                return;
            }

            var predictionEngine = mlContext.Model.CreatePredictionEngine<EmploymentHistory, EmploymentHistoryPrediction>(mlModel);

            var jsonString = File.ReadAllText(inputDataFile);
            var employmentHistory = JsonSerializer.Deserialize<EmploymentHistory>(jsonString);

            var prediction = predictionEngine.Predict(employmentHistory);

            string predictedMonths = $"{prediction.DurationInMonths:#.##}";

            Console.WriteLine(
                $"Based on input data, employee is predicted to work {predictedMonths} months."
            );
        }


    }
}