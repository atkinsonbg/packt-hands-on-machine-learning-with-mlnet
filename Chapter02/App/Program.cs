using Chapter02.App.Ml;

namespace Chapter02.App
{
    class Program
    {
        public static void Main(string[] args)
        {
            if (args.Length != 2)
            {
                Console.WriteLine(
                    $"Invalid arguments passed in, exiting {Environment.NewLine}" +
                    $"Usage:{Environment.NewLine}" +
                    $"predict <line of text to predict against{Environment.NewLine}" +
                    $"train <path to training data file>"
                );
                return;
            }

            switch (args[0])
            {
                case "predict":
                    new Predictor().Predict(args[1]);
                    break;
                case "train":
                    new Trainer().Train(args[1]);
                    break;
                default:
                    Console.WriteLine($"{args[0]} is an invalid option");
                    break;
            }
        }
    }
}
