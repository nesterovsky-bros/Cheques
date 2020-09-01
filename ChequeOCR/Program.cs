using System;
using System.Linq;
using System.Collections.Generic;
using OpenCvSharp;
using Tesseract;
using System.Text.Json.Serialization;
using System.IO;
using System.Text.Json;
using System.Text.Encodings.Web;
using System.Threading;
using System.ComponentModel;
using System.Text;
using System.Runtime.ExceptionServices;
using System.Diagnostics;

namespace ChequeOCR
{
  class Program
  {
    static void Main(string[] args)
    {
      Console.OutputEncoding = Encoding.UTF8;
      Run();
    }

    public static void Run()
    {
      using var scope = new Scope();

      var ocrEngine = new TesseractEngine(
        @"C:\projects\git\Cheques\Cheques\ChequeOCR\traindata\",
        "eng+heb").In(scope);

      using var input = Cv2.ImRead(@"C:\temp\images\cheque1.jpg", ImreadModes.Grayscale);
      //using var input = Cv2.ImRead(@"C:\temp\images\cheque3.jpg", ImreadModes.Grayscale);
//      using var input = Cv2.ImRead(@"C:\temp\images\cheque1-rotated15.jpg", ImreadModes.Grayscale);
      
      //using var input = Cv2.ImRead(@"C:\projects\git\Cheques\Cheques\ChequeOCR\Lenna.png", ImreadModes.Grayscale);
      //using var input = Cv2.ImRead(@"C:\temp\images\cheque1.jpg");
      //using var filtered = input.PyrMeanShiftFiltering(10, 32);

      //input.Dispose();

      //using var src = filtered.CvtColor(ColorConversionCodes.RGB2GRAY);

      //filtered.Dispose();

      var src = input;

      using var scaled = Scale(src, 1600);
      //using var bilateral = scaled.BilateralFilter(7, 50, 50);
      var background = GetBackground(scaled, .9);

      TextImageHistStats stats = GetTextImageHistStats(scaled);

      Console.WriteLine("Background: " + background);

      //using 
      var masked =
        //RemoveWatermark(scaled, background, new Size(3, 3));
        //NormalizeTextImage(scaled, stats, new Size(7, 7));
      NormalizeTextImage2(scaled, new Size(5, 5));


      //var otsu = NormalizeTextImage2(scaled, new Size(5, 5));

      //using(new Window("Otsu", WindowMode.Normal | WindowMode.KeepRatio, otsu))
      //{
      //  Cv2.WaitKey();
      //}

      //using var clahe = Cv2.CreateCLAHE();
      //var normalized = new Mat();

      //clahe.Apply(masked, normalized);
      //masked = normalized;

      using var smooth = Smooth(masked, background, .1);

      var stopwatch = new Stopwatch();

      stopwatch.Start();

      using var sharpened = Sharpen(smooth, new Size(13, 13));
      //using var sharpened = Sharpen(scaled, new Size(10, 10));

      stopwatch.Stop();

      Console.WriteLine("Sharpen: " + stopwatch.ElapsedMilliseconds);

      //using var blured = masked.GaussianBlur(new Size(25, 25), 0);

      var featureInfo = Features(ocrEngine, masked, new Size(18, 18), background);
      var features = sharpened.Clone();

      features.DrawContours(featureInfo.Contours, -1, 64, 2);
      //features.DrawContours(points3, -1, Scalar.WhiteSmoke, 2);

      //foreach(var node in featureInfo.Nodes)
      //{
      //  if (node?.Contours != null)
      //  {
      //    features.DrawContours(
      //      node.Contours, 
      //      -1, 
      //      64, 
      //      2,
      //      LineTypes.Link4,
      //      null,
      //      int.MaxValue,
      //      new Point(node.Left, node.Top));
      //  }
      //}

      //sharpened.SaveImage(@"C:\temp\images\cheque1\cheque1-sharpened.jpg");

      //var page = OCR(ocrEngine, sharpened, 225);
      var nodes = featureInfo.Nodes.
        Where(node => node != null).
        OrderBy(node => node.Top).
        ThenBy(node => node.Left).
        ToList();

      using(var ocrFile =
        File.CreateText(@"C:\temp\images\cheque1\cheque1-sharpened-ocr.json"))
      {
        var json = JsonSerializer.Serialize(
          nodes,
          new JsonSerializerOptions
          {
            IgnoreNullValues = true,
            WriteIndented = true,
            Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
          });

        ocrFile.WriteLine(json);
      }

      //using var binaryMask = input.
      //  Threshold(100, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu).
      //  Dilate(Mat.Ones(18, 18, MatType.CV_8SC1));


      using(new Window("Image", WindowMode.Normal | WindowMode.KeepRatio, scaled))
      using(new Window("Masked", WindowMode.Normal | WindowMode.KeepRatio, masked))
      //using(new Window("BinaryMask", WindowMode.Normal | WindowMode.KeepRatio, binaryMask))
      using(new Window("Sharpened", WindowMode.Normal | WindowMode.KeepRatio, sharpened))
      using(new Window("Features", WindowMode.Normal | WindowMode.KeepRatio, features))
      {
        Cv2.WaitKey();
      }
    }
    public static unsafe Mat Sharpen(Mat input, Size size)
    {
      return input;
      using var e1 = new Mat();
      using var e2 = new Mat();
      using var e3 = new Mat();

      {
        using var data = new Mat();
        using OutputArray dataOutputArray = data;

        input.ConvertTo(dataOutputArray, MatType.CV_32FC1);

        using var data2 = data.Mul(data);
        using var data3 = data2.Mul(data);
        using InputArray dataInputArray = data;
        using InputArray data2InputArray = data2;
        using InputArray data3InputArray = data3;
        using OutputArray e1OutputArray = e1;
        using OutputArray e2OutputArray = e2;
        using OutputArray e3OutputArray = e3;

        Cv2.BoxFilter(dataInputArray, e1OutputArray, MatType.CV_32FC1, size);
        Cv2.BoxFilter(data2InputArray, e2OutputArray, MatType.CV_32FC1, size);
        Cv2.BoxFilter(data3InputArray, e3OutputArray, MatType.CV_32FC1, size);
      }

      var output = new Mat(input.Size(), MatType.CV_8UC1);
      var index = input.GetUnsafeGenericIndexer<byte>();
      var e1Index = e1.GetUnsafeGenericIndexer<float>();
      var e2Index = e2.GetUnsafeGenericIndexer<float>();
      var e3Index = e3.GetUnsafeGenericIndexer<float>();

      output.ForEachAsByte((value, position) =>
      {
        var y = position[0];
        var x = position[1];

        var data = index[y, x];
        var mean = e1Index[y, x];
        var mean2 = mean * mean;
        var stdev2 = e2Index[y, x] - mean2;

        if (stdev2 == 0)
        {
          *value = data;
        }
        else
        {
          var stdev = Math.Sqrt(stdev2);
          var m3 = 
            (e3Index[y, x] - (3 * stdev2 + mean2) * mean) / (stdev2 * stdev);
          var m3_2 = m3 / 2;
          var t = Math.Sqrt(1 + m3_2 * m3_2) - m3_2;

          *value = data <= mean ?
            (byte)Math.Max(mean - stdev * t, 0) :
            (byte)Math.Min(mean + stdev / t, 255);
        }
      });

      return output;
    }

    public static Mat Sharpen2(Mat input, Size size)
    {
      using var data = new Mat();

      {
        using OutputArray dataOutputArray = data;

        input.ConvertTo(dataOutputArray, MatType.CV_32FC1);
      }

      using var mean = new Mat();

      {
        using InputArray dataInputArray = data;
        using OutputArray meanOutputArray = mean;

        Cv2.BoxFilter(dataInputArray, meanOutputArray, MatType.CV_32FC1, size);
      }

      using InputArray meanInputArray = mean;

      Mat stdev2;

      {
        using var varience = new Mat();

        {
          using InputArray dataInputArray = data;
          using OutputArray varienceOutputArray = varience;

          Cv2.SqrBoxFilter(dataInputArray, varienceOutputArray, MatType.CV_32FC1, size);
        }

        using var mean2 = mean.Mul(meanInputArray);
        using var varience_mean2 = varience - mean2;

        stdev2 = varience_mean2;
      }

      using(stdev2)
      {
        using var le = data.LessThanOrEqual(mean);
        using InputArray leInputArray = le;
        using var p = new Mat();

        {
          using OutputArray pOutputArray = p;

          Cv2.BoxFilter(leInputArray, pOutputArray, MatType.CV_32FC1, size);
        }

        using var l2_ = 255.0 / p;
        using var l2 = l2_ - 1.0;

        using InputArray l2InputArray = l2;
        using var r2__0 = stdev2.Mul(l2InputArray);
        using Mat r2_0 = r2__0;
        using Mat r___0 = r2_0.Sqrt();

        r2_0.Dispose();

        using var r__0 = - r___0;
        using Mat r_0 = r__0;

        r___0.Dispose();

        using var r2__1 = stdev2 / l2;
        using Mat r2_1 = r2__1;
        using Mat r_1 = r2_1.Sqrt();

        stdev2.Dispose();
        r2_1.Dispose();

        r_0.CopyTo(r_1, leInputArray);

        r_0.Dispose();

        using var result_ = mean + r_1;
        using Mat result = result_;

        mean.Dispose();

        var output = new Mat();
        using OutputArray outputOutputArray = output;

        result.ConvertTo(outputOutputArray, MatType.CV_8UC1);

        return output;
      }
    }

    public unsafe static Mat Sharpen1(Mat input, Size size)
    {
      var width = input.Width;
      var height = input.Height;
      var output = new Mat(input.Size(), MatType.CV_8UC1);
      var index = input.GetUnsafeGenericIndexer<byte>();

//      var r = input[152 - (size.Height / 2), 152 + ((size.Height + 1) / 2), 559 - (size.Width / 2), 559 + ((size.Width + 1) / 2)].GetRectangularArray<byte>(out var b);

      output.ForEachAsByte((value, position) =>
      {
        var y = position[0];
        var x = position[1];

        var s1 = 0;
        var s2 = 0;
        var count = 0;

        var minx = Math.Max(x - (size.Width >> 1), 0);
        var maxx = Math.Min(x + ((size.Width + 1) >> 1), width);
        var miny = Math.Max(y - (size.Height >> 1), 0);
        var maxy = Math.Min(y + ((size.Height + 1) >> 1), height);

        for(var iny = miny; iny < maxy; ++iny)
        {
          for(var inx = minx; inx < maxx; ++inx)
          {
            var invalue = index[iny, inx];

            s1 += invalue;
            s2 += invalue * invalue;
            ++count;
          }
        }

        var mean = (double)s1 / count;
        var stdev = Math.Sqrt((double)s2 * count - (double)s1 * s1) / count;

        var p = 0;

        for(var iny = miny; iny < maxy; ++iny)
        {
          for(var inx = minx; inx < maxx; ++inx)
          {
            if(index[iny, inx] <= mean)
            {
              ++p;
            }
          }
        }

        if((x == 559) && ((y == 152) || (y == 157)))
        {
          x = x;
        }

        *value = index[y, x] <= mean ?
          (byte)Math.Max(
            mean - stdev * Math.Sqrt((count - p) / (double)p), 0) :
          (byte)Math.Min(
            mean + stdev * Math.Sqrt((double)p / (count - p)), 255);
      });

      return output;
    }

    public struct TextImageHistStats
    {
      public int X0;
      public int M0;
      public int X1;
      public int M1;
      public double Stdev1;
    }

    public unsafe static TextImageHistStats GetTextImageHistStats(Mat input)
    {
      // Calculate histogram
      var hist = new int[256];

      input.ForEachAsByte((value, position) => Interlocked.Increment(ref hist[*value]));

      var size = input.Size();
      var count = size.Width * size.Height;
      var x1 = 0;
      var m1 = -1;
      var s1 = 0.0;
      var s2 = 0.0;

      for(var i = 0; i < hist.Length; ++i)
      {
        var value = hist[i];

        if (m1 < value)
        {
          m1 = value;
          x1 = i;
        }

        var p = (double)value / count;
        
        s1 += i * p;
        s2 += i * i * p;
      }

      var stdev = Math.Sqrt(s2 - s1 * s1);
      var x0 = 0;
      var m0 = -1;

      for(int i = 0, e = (int)(x1 - stdev * 3); i < e; ++i)
      {
        var value = hist[i];

        if (m0 < value)
        {
          m0 = value;
          x0 = i;
        }
      }

      for(int i = (int)(x1 + stdev * 3); i < hist.Length; ++i)
      {
        var value = hist[i];

        if(m0 < value)
        {
          m0 = value;
          x0 = i;
        }
      }

      return new TextImageHistStats
      {
        X0 = x0,
        M0 = m0,
        X1 = x1,
        M1 = m1,
        Stdev1 = stdev
      };
    }

    public unsafe static byte GetBackground(
      Mat input,
      double backgroundPercent)
    {
      // Calculate histogram
      var hist = new int[256];

      input.ForEachAsByte((value, position) => Interlocked.Increment(ref hist[*value]));

      var total = (int)(backgroundPercent * input.Width * input.Height);
      var s = 0;

      for(var i = 256; i-- > 0;)
      {
        s += hist[i];

        var next = total - hist[i];

        if(next <= 0)
        {
          return (byte)((i < 256) && (total < -next) ? i + 1 : i);
        }

        total = next;
      }

      return 1;
    }


    public static unsafe Mat NormalizeTextImage2(
      Mat input,
      Size size)
    {
      using var mean = input.Blur(size);
      using var otsu = mean.
        Threshold(100, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);
      using var ones = Mat.Ones(size.Width, size.Height, MatType.CV_8SC1);
      using var dilate = otsu.Dilate(ones);
      var mask = dilate;

      using var maskInv = ~mask;
      using InputArray maskInvInputArray = maskInv;
      using var backgroundMean = new Mat();
      using OutputArray backgroundMeanOutputArray = backgroundMean;
      using var backgroundStdev = new Mat();
      using OutputArray backgroundStdevOutputArray = backgroundStdev;

      input.MeanStdDev(
        backgroundMeanOutputArray, 
        backgroundStdevOutputArray, 
        maskInvInputArray);

      var meanValue = backgroundMean.Get<double>(0);
      var stdevValue = backgroundStdev.Get<double>(0);
      var background = (byte)Math.Max(meanValue - stdevValue * 3, 0);

      var output = new Mat(input.Size(), MatType.CV_8UC1, background);

      input.CopyTo(output, mask);

      output = output.Threshold(background, background, ThresholdTypes.Trunc);

      //var clahe = Cv2.CreateCLAHE();

      //clahe.Apply(output, output);


      //using(new Window("Output", WindowMode.Normal | WindowMode.KeepRatio, output))
      //{
      //  Cv2.WaitKey();
      //}

      return output;
    }

    public static unsafe Mat NormalizeTextImage(
      Mat input, 
      TextImageHistStats stats, 
      Size size,
      double noiseStdevFactor = 1)
    {
      var x0 = stats.X0;
      var x1 = stats.X1;
      using var inverse_ = x1 < x0 ? ~input : null;
      using var inverse = inverse_ != null ? (Mat)inverse_ : null;

      if(x1 < x0)
      {
        x0 = 255 - x0;
        x1 = 255 - x1;
      }

      var background =
        (byte)Math.Max((int)(x1 - stats.Stdev1 * noiseStdevFactor), 0);
      var src = inverse ?? input;
      using var mean = src.Blur(size);
      var meanIndex = mean.GetUnsafeGenericIndexer<byte>();
      var index = src.GetUnsafeGenericIndexer<byte>();
      var normalized = new Mat(input.Size(), MatType.CV_8UC1);

      normalized.ForEachAsByte((value, position) =>
      {
        var y = position[0];
        var x = position[1];
        var invalue = index[y, x];
        var mean = meanIndex[y, x];

        *value = mean < background ? invalue : background;
      });

      var k = 255.0 / (background - x0);
      var lookup = new byte[256];

      for(var i = 0; i < lookup.Length; ++i)
      {
        lookup[i] = i <= x0 ? (byte)0 : i >= x1 ? 
          (byte)255 : (byte)Math.Min((i - x0) * k, 255);
      }

      using var lut = InputArray.Create(lookup);
      
      return normalized.LUT(lut);
    }

    public static unsafe Mat RemoveWatermark(Mat input, byte background, Size size)
    {
      var output = new Mat(input.Size(), MatType.CV_8UC1);
      var index = input.GetUnsafeGenericIndexer<byte>();
      using var mean = input.Blur(size);
      var meanIndex = mean.GetUnsafeGenericIndexer<byte>();

      output.ForEachAsByte((value, position) =>
      {
        var y = position[0];
        var x = position[1];
        var invalue = index[y, x];
        var mean = meanIndex[y, x];

        *value = (mean < background) && (invalue < background) ?
          invalue : background;
      });

      return output;
    }

    public static unsafe Mat Smooth(
      Mat input,
      byte background,
      double threshold)
    {
      var width = input.Width;
      var height = input.Height;
      var output = new Mat(input.Size(), MatType.CV_8UC1);
      var index = input.GetUnsafeGenericIndexer<byte>();

      output.ForEachAsByte((value, position) =>
      {
        var y = position[0];
        var x = position[1];

        int d4 = index[y, x];
        int d1 = y > 0 ? index[y - 1, x] - d4 : d4;
        int d3 = x > 0 ? index[y, x - 1] - d4 : d4;
        int d5 = x + 1 < width ? index[y, x + 1] - d4 : d4;
        int d7 = y + 1 < height ? index[y + 1, x] - d4 : d4;

        if((((d1 > 0) == (d7 > 0)) &&
          (Math.Min(d1, d7) >= d4 * threshold)) ||
          (((d3 > 0) == (d5 > 0)) &&
          (Math.Min(d3, d5) >= d4 * threshold)))
        {
          var result = (d1 + d5 + d7 + d3) / 4 + d4;

          *value = result < background ? (byte)result : background;
        }
        else
        {
          *value = (byte)d4;
        }
      });

      return output;
    }

    public static Mat Scale(Mat input, int width)
    {
      var scale = (double)width / input.Width;

      return input.Resize(
        Size.Zero,
        scale,
        scale,
        InterpolationFlags.Area);
    }

    public class FeatureInfo
    {
      public Point[][] Contours;
      public HierarchyIndex[] Hierarchy;
      public OcrNode[] Nodes;
    }

    public unsafe static FeatureInfo Features(
      TesseractEngine engine, 
      Mat input, 
      Size size, 
      byte background)
    {
      using var scope = new Scope();

      var mask = input.
        Threshold(100, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu).
        Temp(scope).
        Dilate(Mat.Ones(size.Width, size.Height, MatType.CV_8SC1).Temp(scope)).
        In(scope);

      mask.FindContours(
        out var points, 
        out var hierarchy, 
        RetrievalModes.CComp, 
        ContourApproximationModes.ApproxSimple);

      var nodes = points.
        Select((contour, index) =>
        {
          if (hierarchy[index].Parent >= 0)
          {
            return null;
          }

          var rect = Cv2.BoundingRect(contour);

          if ((rect.Width < size.Width) && (rect.Height < size.Height))
          {
            return null;
          }

          using var roi = new Mat(rect.Size, MatType.CV_8UC1, 0);

          roi.DrawContours(
            points,
            index,
            Scalar.White,
            -1,
            LineTypes.Link4,
            null,
            int.MaxValue,
            -rect.TopLeft);

          using var inputView = input[rect];
          using var maskedInput = 
            new Mat(rect.Size, MatType.CV_8UC1, background);

          inputView.CopyTo(maskedInput, roi);

          using var fineMask = maskedInput.
            Threshold(100, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu).
            Dilate(new Mat());

          fineMask.FindContours(
            out var finePoints,
            out var fineHierarchy,
            RetrievalModes.CComp,
            ContourApproximationModes.ApproxSimple);

          using var fineMask2 = new Mat(rect.Size, MatType.CV_8UC1, 0);

          fineMask2.DrawContours(finePoints, -1, Scalar.White, -1);

          using var bluredFineMask2 = fineMask2.Blur(size);

          using var maxedFineMask2 = new Mat(rect.Size, MatType.CV_8UC1);
          var ix = bluredFineMask2.GetUnsafeGenericIndexer<byte>();

          maxedFineMask2.ForEachAsByte((value, position) =>
          {
            var y = position[0];
            var x = position[1];
            var minx = Math.Max(x - (size.Width >> 1), 0);
            var maxx = Math.Min(x + ((size.Width + 1) >> 1), rect.Width);
            var miny = Math.Max(y - (size.Height >> 1), 0);
            var maxy = Math.Min(y + ((size.Height + 1) >> 1), rect.Height);
            var max = 0;

            for(var iny = miny; iny < maxy; ++iny)
            {
              for(var inx = minx; inx < maxx; ++inx)
              {
                var invalue = ix[iny, inx];

                if(max < invalue)
                {
                  max = invalue;
                }
              }
            }

            var v = ix[y, x];

            *value = v >= max && max > 0 ? (byte)255 : (byte)0;
          });

          //using(new Window("fineMask2 " + index, WindowMode.AutoSize, maxedFineMask2))
          //{
          //  Cv2.WaitKey();
          //}

          using var smooth = Smooth(maskedInput, background, .1);
          using var sharpened = Sharpen(smooth, size);

          var lines = maxedFineMask2.HoughLinesP(
            1,
            Math.PI / 180,
            100,
            Math.Max(rect.Width, rect.Height) / 2,
            size.Width);

          //if (lines.Length == 0)
          //{
          //  return null;
          //}

          var angle = 0;

          //(int)(lines.
          //  Average(l => Math.Atan2(l.P2.Y - l.P1.Y, l.P2.X - l.P1.Y)) /
          //    Math.PI * 180);

          //angle -= angle % 4;

          var ocrView = sharpened;

          //if(angle != 0)
          //{
          //  using var rotation = Cv2.GetRotationMatrix2D(new Point2f(), angle, 1);

          //  var cos = Math.Abs(rotation.At<double>(0, 0));
          //  var sin = Math.Abs(rotation.At<double>(0, 1));
          //  var width = (int)(rect.Height * sin + rect.Width * cos);
          //  var height = (int)(rect.Height * cos + rect.Width * sin);

          //  rotation.At<float>(0, 2) = (width - rect.Width) / 2;
          //  rotation.At<float>(1, 2) = (height - rect.Height) / 2;

          //  ocrView = sharpened.WarpAffine(
          //    rotation,
          //    new Size(width, height),
          //    InterpolationFlags.Area,
          //    BorderTypes.Constant,
          //    new Scalar(background));
          //}

          using var ocrViewResource = ocrView;
          var node = OCR(engine, ocrView, 225);

          //if(index == 44)
          //{
          //  index = 44;
          //}

          if(!(node?.Children?.Count > 0))
          {
            return null;
          }

          node.Contours = finePoints;
          node.Angle = angle;
          node.Left = rect.Left;
          node.Top = rect.Top;
          node.Width = rect.Width;
          node.Height = rect.Height;

          return node;
        }).
        ToArray();

      return new FeatureInfo
      {
        Contours = points,
        Hierarchy = hierarchy,
        Nodes = nodes
      };
    }

    public static Mat Edge(Mat input)
    {
      var scale = 1;
      var delta = 0;
      var ddepth = MatType.CV_16S;

      /// Gradient X
      using var grad_x = input.Scharr(ddepth, 1, 0, scale, delta, BorderTypes.Default);
      //using var grad_x = input.Sobel(ddepth, 1, 0, 3, scale, delta, BorderTypes.Default);

      using var abs_grad_x = grad_x.ConvertScaleAbs();

      grad_x.Dispose();

      /// Gradient Y
      using var grad_y = input.Scharr(ddepth, 0, 1, scale, delta, BorderTypes.Default);
      //using var grad_y = input.Sobel(ddepth, 0, 1, 3, scale, delta, BorderTypes.Default);

      using var abs_grad_y = grad_y.ConvertScaleAbs();

      grad_y.Dispose();

      /// Total Gradient (approximate)

      using var grad = new Mat();
        
      Cv2.AddWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

      abs_grad_y.Dispose();
      abs_grad_y.Dispose();

      var output = new Mat(grad.Width, grad.Height, MatType.CV_8UC1);

      Cv2.Normalize(grad, output, 255, 0, NormTypes.MinMax);

      return output;
    }

    public unsafe static OcrNode OCR(
      TesseractEngine engine, 
      Mat input, 
      int resolution)
    {
      using var pix = Pix.Create(input.Width, input.Height, 8);

      pix.XRes = resolution;
      pix.YRes = resolution;

      var pixData = pix.GetData();
      var data = (uint*)pixData.Data;
      var wordsPerLine = pixData.WordsPerLine;

      input.ForEachAsByte((value, position) =>
      {
        var y = position[0];
        var x = position[1];
        var color = *value;

        PixData.SetDataByte(data + y * wordsPerLine, x, color);
      });


      using var page = engine.Process(pix);

      var ocrPage = page.GetTsvText(1).
        Split("\n", StringSplitOptions.RemoveEmptyEntries).
        Select(line => line.Split("\t")).
        Select(columns => new OcrNode
        {
          Level = int.Parse(columns[0]),
          PageNum = int.Parse(columns[1]),
          BlockNum = int.Parse(columns[2]),
          ParNum = int.Parse(columns[3]),
          LineNum = int.Parse(columns[4]),
          WordNum = int.Parse(columns[5]),
          Left = int.Parse(columns[6]),
          Top = int.Parse(columns[7]),
          Width = int.Parse(columns[8]),
          Height = int.Parse(columns[9]),
          Confidence = int.Parse(columns[10]),
          Text = columns.Length > 11 ? columns[11] : null
        }).
        Aggregate((prev, next) =>
        {
          if(prev == null)
          {
            if(next.Level != 1)
            {
              throw new InvalidOperationException("nodes");
            }
          }
          else
          {
            if((next.Level <= 0) || (next.Level > prev.Level + 1))
            {
              throw new InvalidOperationException("nodes");
            }

            while(next.Level != prev.Level + 1)
            {
              prev = prev.Parent;
            }

            next.Parent = prev;

            if(prev.Children == null)
            {
              prev.Children = new List<OcrNode>();
            }

            next.Parent.Children.Add(next);
          }

          return next;
        });

      while(ocrPage.Parent != null)
      {
        ocrPage = ocrPage.Parent;
      }

      ocrPage.Text = page.GetText();

      return ocrPage;
    }

    public class OcrNode
    {
      public override string ToString()
      {
        return Text;
      }

      [JsonIgnore]
      public OcrNode Parent { get; set; }
      public List<OcrNode> Children { get; set; }

      [DefaultValue(0)]
      public int Angle { get; set; }

      [JsonIgnore]
      public Point[][] Contours { get; set; }

      /**
       * 1 - page
       * 2 - block
       * 3 - paragraph
       * 4 - line
       * 5 - word
       */
      public int Level { get; set; }

      public int PageNum { get; set; }
      public int BlockNum { get; set; }
      public int ParNum { get; set; }
      public int LineNum { get; set; }
      public int WordNum { get; set; }
      public int Left { get; set; }
      public int Top { get; set; }
      public int Width { get; set; }
      public int Height { get; set; }
      public int Confidence { get; set; }
      public string Text { get; set; }
    }
  }

  public class Scope: IDisposable
  {
    public void Dispose()
    {
      try
      {
        Dispose(tempResources);
      }
      finally
      {
        Dispose(resources);
      }
    }

    public T Add<T>(T resource)
      where T : IDisposable
    {
      resources.Push(resource);
      Dispose(tempResources);

      return resource;
    }

    public T AddTemp<T>(T resource)
      where T : IDisposable
    {
      tempResources.Push(resource);

      return resource;
    }

    private static void Dispose(Stack<IDisposable> resources)
    {
      ExceptionDispatchInfo error = null;

      while(resources.TryPop(out var resource))
      {
        try
        {
          resource?.Dispose();
        }
        catch(Exception e)
        {
          if(error == null)
          {
            error = ExceptionDispatchInfo.Capture(e);
          }
        }
      }

      error?.Throw();
    }

    private readonly Stack<IDisposable> resources = new Stack<IDisposable>();
    private readonly Stack<IDisposable> tempResources = new Stack<IDisposable>();
  }

  public static class ScopeExtension
  {
    public static T In<T>(this T resource, Scope scope)
      where T : IDisposable
    {
      return scope.Add(resource);
    }

    public static T Temp<T>(this T resource, Scope scope)
      where T : IDisposable
    {
      return scope.AddTemp(resource);
    }
  }
}
