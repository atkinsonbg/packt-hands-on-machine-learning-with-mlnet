using Chapter02.App.Common;
using Chapter02.App.Ml.Objects;
using Microsoft.ML;

namespace Chapter02.App.Ml.Base
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