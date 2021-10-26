using Chapter02.App.Ml.Base;
using Chapter02.App.Ml.Objects;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;

namespace Chapter02.App.Ml
{
    public class Trainer : BaseML
    {
        public void Train(string trainingFileName)
        {
            if (!File.Exists(trainingFileName))
            {
                Console.WriteLine($"Could not find the training file {trainingFileName}");
                return;
            }

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<RestaurantFeedback>(trainingFileName);

            DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);

            TextFeaturizingEstimator dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(
                            outputColumnName: "Features",
                            inputColumnName: nameof(RestaurantFeedback.Text));

            SdcaLogisticRegressionBinaryTrainer sdcsRegressionTrainer =
                mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: nameof(RestaurantFeedback.Label),
                    featureColumnName: "Features"
                );

            EstimatorChain<BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>> trainingPipeline =
                    dataProcessPipeline.Append(sdcsRegressionTrainer);

            ITransformer trainedModel = trainingPipeline.Fit(dataSplit.TrainSet);

            mlContext.Model.Save(trainedModel, dataSplit.TrainSet.Schema, ModelPath);

            IDataView testSetTransform = trainedModel.Transform(dataSplit.TestSet);

            CalibratedBinaryClassificationMetrics modelMetrics =
                mlContext.BinaryClassification.Evaluate(
                    data: testSetTransform,
                    labelColumnName: nameof(RestaurantFeedback.Label),
                    scoreColumnName: nameof(RestaurantPrediction.Score)
                );

            Console.WriteLine(
                $"Area Under Curve: {modelMetrics.AreaUnderRocCurve:P2}{Environment.NewLine}" +
                $"Area Under Precision Recall Curve: {modelMetrics.AreaUnderPrecisionRecallCurve:P2}{Environment.NewLine}" +
                $"Accuracy: {modelMetrics.Accuracy:P2}{Environment.NewLine}" +
                $"F1 Score: {modelMetrics.F1Score:P2}{Environment.NewLine}" +
                $"Postive Recall: {modelMetrics.PositiveRecall:#.##}{Environment.NewLine}" +
                $"Negative Recall: {modelMetrics.NegativeRecall:#.##}{Environment.NewLine}"
            );
        }
    }
}