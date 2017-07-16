// ------------------------------------------------------------ 
// Blog post code sample! Read the post on http://en.morzel.net
// ------------------------------------------------------------

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Diagnostics;
using System.Drawing;

namespace EmguApp
{
    class Program
    {
        // Determines boundary of brightness while turning grayscale image to binary (black-white) image
        private const int Threshold = 5;

        // Erosion to remove noise (reduce white pixel zones)
        private const int ErodeIterations = 3;

        // Dilation to enhance erosion survivors (enlarge white pixel zones)
        private const int DilateIterations = 3;

        // Window names used in CvInvoke.Imshow calls
        private const string BackgroundFrameWindowName = "Background Frame";
        private const string RawFrameWindowName = "Raw Frame";
        private const string GrayscaleDiffFrameWindowName = "Grayscale Difference Frame";
        private const string BinaryDiffFrameWindowName = "Binary Difference Frame";
        private const string DenoisedDiffFrameWindowName = "Denoised Difference Frame";
        private const string FinalFrameWindowName = "Final Frame";

        // Containers for images demonstrating different phases of frame processing 
        private static Mat rawFrame = new Mat(); // Frame as obtained from video
        private static Mat backgroundFrame = new Mat(); // Frame used as base for change detection
        private static Mat diffFrame = new Mat(); // Image showing differences between background and raw frame
        private static Mat grayscaleDiffFrame = new Mat(); // Image showing differences in 8-bit color depth
        private static Mat binaryDiffFrame = new Mat(); // Image showing changed areas in white and unchanged in black
        private static Mat denoisedDiffFrame = new Mat(); // Image with irrelevant changes removed with opening operation
        private static Mat finalFrame = new Mat(); // Video frame with detected object marked

        private static MCvScalar drawingColor = new Bgr(Color.Red).MCvScalar;

        static void Main(string[] args)
        {
            // Put a path to video file below:
            string videoFile = @"";
            // You can download sample video used in the blog post from here:
            // http://morzel.net/download/emgu_cv_drone_test_video.mp4 (4.04 MB, MPEG4 H264 640x480 25fps)

            using (var capture = new VideoCapture(videoFile)) // Loading video from file
            {
                if (capture.IsOpened)
                {
                    Console.WriteLine($"{videoFile} is opened");
                    Console.WriteLine("Press ESCAPE key in any image window to close the program.");
                    Console.WriteLine("Press other key in any image window to move to next frame.");

                    // Obtaining and showing first frame of loaded video (used as the base for difference detection)
                    backgroundFrame = capture.QueryFrame();
                    CvInvoke.Imshow(BackgroundFrameWindowName, backgroundFrame);

                    // Handling video frames (image processing and contour detection)
                    VideoProcessingLoop(capture, backgroundFrame);
                }
                else
                {
                    Console.WriteLine($"Unable to open {videoFile}");
                }
            }
        }

       private static void VideoProcessingLoop(VideoCapture capture, Mat backgroundFrame)
        {
            var stopwatch = new Stopwatch(); // Used for measuring video processing performance

            int frameNumber = 1;
            while (true) // Loop video
            {
                rawFrame = capture.QueryFrame(); // Getting next frame (null is returned if no further frame exists)

                if (rawFrame != null) 
                {
                    frameNumber++;

                    stopwatch.Restart();
                    ProcessFrame(backgroundFrame, Threshold, ErodeIterations, DilateIterations);
                    stopwatch.Stop();

                    WriteFrameInfo(stopwatch.ElapsedMilliseconds, frameNumber);
                    ShowWindowsWithImageProcessingStages();

                    int key = CvInvoke.WaitKey(0); // Wait indefinitely until key is pressed

                    // Close program if Esc key was pressed (any other key moves to next frame)
                    if (key == 27)
                        Environment.Exit(0);
                }
                else
                {
                    capture.SetCaptureProperty(CapProp.PosFrames, 0); // Move to first frame
                    frameNumber = 0;
                }
            }
        }

        private static void ProcessFrame(Mat backgroundFrame, int threshold, int erodeIterations, int dilateIterations)
        {
            // Find difference between background (first) frame and current frame
            CvInvoke.AbsDiff(backgroundFrame, rawFrame, diffFrame);

            // Apply binary threshold to grayscale image (white pixel will mark difference)
            CvInvoke.CvtColor(diffFrame, grayscaleDiffFrame, ColorConversion.Bgr2Gray);
            CvInvoke.Threshold(grayscaleDiffFrame, binaryDiffFrame, threshold, 255, ThresholdType.Binary);

            // Remove noise with opening operation (erosion followed by dilation)
            CvInvoke.Erode(binaryDiffFrame, denoisedDiffFrame, null, new Point(-1, -1), erodeIterations, BorderType.Default, new MCvScalar(1));
            CvInvoke.Dilate(denoisedDiffFrame, denoisedDiffFrame, null, new Point(-1, -1), dilateIterations, BorderType.Default, new MCvScalar(1));

            rawFrame.CopyTo(finalFrame);
            DetectObject(denoisedDiffFrame, finalFrame);
        }

        private static void DetectObject(Mat detectionFrame, Mat displayFrame)
        {
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                // Build list of contours
                CvInvoke.FindContours(detectionFrame, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);

                // Selecting largest contour
                if (contours.Size > 0)
                {
                    double maxArea = 0;
                    int chosen = 0;
                    for (int i = 0; i < contours.Size; i++)
                    {
                        VectorOfPoint contour = contours[i];

                        double area = CvInvoke.ContourArea(contour);
                        if (area > maxArea)
                        {
                            maxArea = area;
                            chosen = i;
                        }
                    }

                    // Draw on a frame
                    MarkDetectedObject(displayFrame, contours[chosen], maxArea);
                }
            }
        }

        private static void WriteFrameInfo(long elapsedMs, int frameNumber)
        {
            var info = new string[] {
                $"Frame Number: {frameNumber}",
                $"Processing Time: {elapsedMs} ms"
            };

            WriteMultilineText(finalFrame, info, new Point(5, 10));
        }

        private static void ShowWindowsWithImageProcessingStages()
        {
            CvInvoke.Imshow(RawFrameWindowName, rawFrame);
            CvInvoke.Imshow(GrayscaleDiffFrameWindowName, grayscaleDiffFrame);
            CvInvoke.Imshow(BinaryDiffFrameWindowName, binaryDiffFrame);
            CvInvoke.Imshow(DenoisedDiffFrameWindowName, denoisedDiffFrame);
            CvInvoke.Imshow(FinalFrameWindowName, finalFrame);
        }

        private static void MarkDetectedObject(Mat frame, VectorOfPoint contour, double area)
        {
            // Getting minimal rectangle which contains the contour
            Rectangle box = CvInvoke.BoundingRectangle(contour);

            // Drawing contour and box around it
            CvInvoke.Polylines(frame, contour.ToArray(), true, drawingColor);
            CvInvoke.Rectangle(frame, box, drawingColor);

            // Write information next to marked object
            Point center = new Point(box.X + box.Width / 2, box.Y + box.Height / 2);

            var info = new string[] {
                $"Area: {area}",
                $"Position: {center.X}, {center.Y}"
            };

            WriteMultilineText(frame, info, new Point(box.Right + 5, center.Y));
        }

        private static void WriteMultilineText(Mat frame, string[] lines, Point origin)
        {
            for (int i = 0; i < lines.Length; i++)
            {
                int y = i * 10 + origin.Y; // Moving down on each line
                CvInvoke.PutText(frame, lines[i], new Point(origin.X, y), FontFace.HersheyPlain, 0.8, drawingColor);
            }
        }
    }
}
