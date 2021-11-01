using Chapter03.App.Common;
using Chapter03.App.Ml.Objects;
using Microsoft.ML;

namespace Chapter03.App.Ml.Base
{
    public class BaseML
    {
        protected static string ModelPath =>
            Path.Combine(AppContext.BaseDirectory, Constants.MODEL_FILENAME);

        protected readonly MLContext mlContext;

        protected BaseML()
        {
            mlContext = new MLContext(0);
        }
    }
}