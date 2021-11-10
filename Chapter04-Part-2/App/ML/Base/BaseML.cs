using System;
using App.Common;
using Microsoft.ML;

namespace App.ML.Base
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

