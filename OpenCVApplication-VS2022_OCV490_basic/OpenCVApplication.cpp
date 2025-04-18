// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>
#include <cmath>

wchar_t* projectPath;
Mat img;
Mat img2;
Point centerOfMass;
double phi_angle;



void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
	waitKey();

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}



void grayAdditive()
{
	char fname[MAX_PATH];
	uchar final;
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				int additive = val - 80;
				if (additive > 255)
				{
					uchar final = 255;
				}
				else if (additive < 0)
				{
					final = 0;
				}
				else
				{
					final = additive;
				}

				dst.at<uchar>(i, j) = final;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("new image", dst);
		waitKey();
	}
}

void grayMultiplicative()
{
	char fname[MAX_PATH];
	uchar final;
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				float multiplicative = val * 1.5;
				if (multiplicative > 255)
				{
					final = 255;
				}
				else if (multiplicative < 0)
				{
					final = 0;
				}
				else
				{
					final = multiplicative;
				}

				dst.at<uchar>(i, j) = final;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("new image", dst);
		waitKey();
	}

}


void the4Squares()
{
	Mat img(256, 256, CV_8UC3);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (i < 128 && j < 128)
			{
				img.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			else if (i < 128 && j >= 128)
			{
				img.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
			}
			else if (i >= 128 && j < 128)
			{
				img.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
			}
			else
			{
				img.at<Vec3b>(i, j) = Vec3b(0, 255, 255);
			}
		}
	}

	imshow("4 Colored Squares", img);
	waitKey(0);
}



void inverseMatrix()
{
	float vals[9] = { 1, 2, 3, 4 ,5,6,7,8,9 };
	Mat M(3, 3, CV_32FC1, vals);
	Mat inverse_matrix;
	double determinant = invert(M, inverse_matrix, cv::DECOMP_LU);

	if (determinant != 0) {
		std::cout << "Inverse Matrix:\n" << inverse_matrix << "\n";
	}
	else {
		std::cout << "The matrix is singular (not invertible).\n";
	}
	std::cout << inverse_matrix << std::endl;

}

void ex1LB2()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		Mat r = Mat(height, width, CV_8UC1);
		Mat g = Mat(height, width, CV_8UC1);
		Mat b = Mat(height, width, CV_8UC1);




		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar red = v3[2];
				uchar green = v3[1];
				uchar blue = v3[0];
				r.at<uchar>(i, j) = red;
				g.at<uchar>(i, j) = green;
				b.at<uchar>(i, j) = blue;
			}
		imshow("red", r);
		imshow("green", g);
		imshow("blue", b);
		waitKey(0);
	}
}

void ex2LB2()
{

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		Mat r = Mat(height, width, CV_8UC1);
		Mat g = Mat(height, width, CV_8UC1);
		Mat b = Mat(height, width, CV_8UC1);
		Mat grey = Mat(height, width, CV_8UC1);




		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar red = v3[2];
				uchar green = v3[1];
				uchar blue = v3[0];
				r.at<uchar>(i, j) = red;
				g.at<uchar>(i, j) = green;
				b.at<uchar>(i, j) = blue;
				grey.at<uchar>(i, j) = (red + blue + green) / 3;
			}

		imshow("grey", grey);
		waitKey(0);
	}
}
void ex3LB2()
{
	uchar threshold = 100;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				if (val > threshold)
					dst.at<uchar>(i, j) = 255;
				else
					dst.at<uchar>(i, j) = 0;
			}
		imshow("input image", src);
		imshow("binary image", dst);
		waitKey();
	}


}
void onMouse(int event, int x, int y, int, void*) {
	if (event != EVENT_LBUTTONDOWN) return;

	int height = img.rows;
	int width = img.cols;
	int area = 0, sumX = 0, sumY = 0;
	int rowMin = height, rowMax = 0, colMin = width, colMax = 0;
	int perimeter = 0;
	double a1 = 0, a2 = 0, a3 = 0;

	Mat dst = Mat(height, width, CV_8UC3);

	Vec3b bgColor = img.at<Vec3b>(y, x);
	int colorThreshold = 30;

	Mat horizontalProjection = Mat(height, width, CV_8UC3);
	Mat verticalProjection = Mat(height, width, CV_8UC3);
	Vec3b projColor(0, 0, 255);
	std::vector<int> rowCounts(height, 0);
	std::vector<int> colCounts(width, 0);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Vec3b color = img.at<Vec3b>(i, j);
			int diff = abs(color[0] - bgColor[0]) + abs(color[1] - bgColor[1]) + abs(color[2] - bgColor[2]);

			if (diff < colorThreshold) {
				area++;
				sumX += j;
				sumY += i;
				rowCounts[i]++;
				colCounts[j]++;

				a1 += (j * j);
				a2 += (i * i);
				a3 += (i * j);

				if (i < rowMin) rowMin = i;
				if (i > rowMax) rowMax = i;
				if (j < colMin) colMin = j;
				if (j > colMax) colMax = j;

				if (i > 0 && i < height && j > 0 && j < width) {
					if (abs(img.at<Vec3b>(i - 1, j)[0] - bgColor[0]) > colorThreshold ||
						abs(img.at<Vec3b>(i + 1, j)[0] - bgColor[0]) > colorThreshold ||
						abs(img.at<Vec3b>(i, j - 1)[0] - bgColor[0]) > colorThreshold ||
						abs(img.at<Vec3b>(i, j + 1)[0] - bgColor[0]) > colorThreshold ||
						abs(img.at<Vec3b>(i + 1, j + 1)[0] - bgColor[0]) > colorThreshold ||
						abs(img.at<Vec3b>(i - 1, j + 1)[0] - bgColor[0]) > colorThreshold ||
						abs(img.at<Vec3b>(i + 1, j - 1)[0] - bgColor[0]) > colorThreshold ||
						abs(img.at<Vec3b>(i - 1, j - 1)[0] - bgColor[0]) > colorThreshold) {
						perimeter++;
						dst.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);
					}
				}

				
			}
		}
	}

	if (area == 0) return;

	int maxRowCount = *max_element(rowCounts.begin(), rowCounts.end());
	int maxColCount = *max_element(colCounts.begin(), colCounts.end());

	// Draw horizontal projection (vertical bars)
	for (int i = 0; i < height; i++) {
		int barWidth = (rowCounts[i] * width) / max(maxRowCount, 1);
		line(horizontalProjection, Point(0, i), Point(barWidth, i), projColor);
	}

	// Draw vertical projection (horizontal bars)
	for (int j = 0; j < width; j++) {
		int barHeight = height - (colCounts[j] * height) / max(maxColCount, 1);
		line(verticalProjection, Point(j, height), Point(j, barHeight), projColor);
	}

	centerOfMass = Point(sumX / area, sumY / area);
	circle(dst, centerOfMass, 5, Scalar(0, 255, 255), -1);
	double aspectRatio = (double)(colMax - colMin) / (rowMax - rowMin);
	perimeter = perimeter * CV_PI / 4;
	double thinness = (4 * CV_PI * area) / (perimeter * perimeter);

	a1 /= area; a2 /= area; a3 /= area;
	a1 -= pow(sumX / (double)area, 2);
	a2 -= pow(sumY / (double)area, 2);
	a3 -= (sumX / (double)area) * (sumY / (double)area);
	phi_angle = 0.5 * atan2(2 * a3, a1 - a2);
	double phi_degrees = (phi_angle + CV_PI) * 180.0 / CV_PI;

	double angle = phi_angle;
	double length = 100.0;

	double dx = length * cos(angle);
	double dy = length * sin(angle);

	Point pt1((int)(centerOfMass.x - dx), (int)(centerOfMass.y - dy));
	Point pt2((int)(centerOfMass.x + dx), (int)(centerOfMass.y + dy));
	line(dst, pt1, pt2, Scalar(255, 0, 0), 2);

	std::cout << "Area: " << area << "\n";
	std::cout << "Center of Mass: (" << centerOfMass.x << ", " << centerOfMass.y << ")" << "\n";
	std::cout << "Perimeter: " << perimeter << "\n";
	std::cout << "Thinness Ratio: " << thinness << "\n";
	std::cout << "Axis Elongation: " << phi_degrees << "\n";
	std::cout << "Aspect Ratio: " << aspectRatio << "\n";


	imshow("Image", img);
	imshow("New Image", dst);
	imshow("Horizontal", horizontalProjection);
	imshow("Vertical", verticalProjection);


}




void ex1aLb4()
{
	char fname[MAX_PATH];
	if (openFileDlg(fname)) {
		img = imread(fname, IMREAD_COLOR);
		if (img.empty()) {
			std::cout << "Error loading image!\n";
			return;
		}
		imshow("Image", img);
		setMouseCallback("Image", onMouse);
		waitKey(0);
	}

}
void filter_objects(Mat img, Mat labeled_img, double* orientations, int TH_area, double phi_LOW, double phi_HIGH) {
	int height = img.rows;
	int width = img.cols;
	Mat dst = img.clone(); 

	double phi_low_rad = phi_LOW * CV_PI / 180.0;
	double phi_high_rad = phi_HIGH * CV_PI / 180.0;

	// Find max label value
	double maxLabel;
	minMaxLoc(labeled_img, NULL, &maxLabel);
	int numLabels = (int)maxLabel;

	// For visualization - we'll draw filtered objects with their original colors
	Mat filteredDisplay = Mat::zeros(height, width, CV_8UC3);

	for (int label = 1; label <= numLabels; label++) {
		// Calculate object properties (same as in onMouse)
		int area = 0;
		double sumX = 0, sumY = 0;
		double a1 = 0, a2 = 0, a3 = 0;
		int rowMin = height, rowMax = 0, colMin = width, colMax = 0;

		// First pass: compute statistics
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (labeled_img.at<int>(i, j) == label) {
					area++;
					sumX += j;
					sumY += i;
					a1 += j * j;
					a2 += i * i;
					a3 += i * j;

					// Update bounding box
					if (i < rowMin) rowMin = i;
					if (i > rowMax) rowMax = i;
					if (j < colMin) colMin = j;
					if (j > colMax) colMax = j;
				}
			}
		}

		if (area == 0) continue;

		// Calculate orientation
		a1 = a1 / area - pow(sumX / area, 2);
		a2 = a2 / area - pow(sumY / area, 2);
		a3 = a3 / area - (sumX / area) * (sumY / area);
		double phi = 0.5 * atan2(2 * a3, a1 - a2);

		// Check filtering conditions
		if (area < TH_area && phi > phi_low_rad && phi < phi_high_rad) {
			// Draw the object (copy pixels from original image)
			for (int i = rowMin; i <= rowMax; i++) {
				for (int j = colMin; j <= colMax; j++) {
					if (labeled_img.at<int>(i, j) == label) {
						filteredDisplay.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);
					}
				}
			}

			// Draw center and axis (like in onMouse)
			Point center(sumX / area, sumY / area);
			circle(filteredDisplay, center, 3, Scalar(0, 255, 255), -1);

			double length = 50.0;
			Point pt1(center.x - length * cos(phi), center.y - length * sin(phi));
			Point pt2(center.x + length * cos(phi), center.y + length * sin(phi));
			line(filteredDisplay, pt1, pt2, Scalar(255, 0, 0), 2);
		}
	}

	imshow("Filtered Objects", filteredDisplay);
}

void ex2Lb4() {
	char fname[MAX_PATH];
	if (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_COLOR);
		if (src.empty()) {
			std::cout << "Error loading image!\n";
			return;
		}

		// Convert to grayscale and threshold to binary
		Mat gray, binary;
		cvtColor(src, gray, COLOR_BGR2GRAY);
		threshold(gray, binary, 128, 255, THRESH_BINARY);

		// Label connected components
		Mat labels;
		int nLabels = connectedComponents(binary, labels, 8, CV_32S);

		// Get user parameters
		int TH_area;
		double phi_LOW, phi_HIGH;
		printf("Enter maximum area threshold: ");
		scanf("%d", &TH_area);
		printf("Enter minimum orientation angle (degrees): ");
		scanf("%lf", &phi_LOW);
		printf("Enter maximum orientation angle (degrees): ");
		scanf("%lf", &phi_HIGH);

		// Compute object orientations manually
		std::vector<double> orientations(nLabels, 0);
		for (int label = 1; label < nLabels; label++) {
			int area = 0;
			double sumX = 0, sumY = 0, a1 = 0, a2 = 0, a3 = 0;

			for (int i = 0; i < labels.rows; i++) {
				for (int j = 0; j < labels.cols; j++) {
					if (labels.at<int>(i, j) == label) {
						area++;
						sumX += j;
						sumY += i;
						a1 += j * j;
						a2 += i * i;
						a3 += i * j;
					}
				}
			}

			if (area > 0) {
				a1 /= area;
				a2 /= area;
				a3 /= area;
				a1 -= pow(sumX / area, 2);
				a2 -= pow(sumY / area, 2);
				a3 -= (sumX / area) * (sumY / area);
				orientations[label] = 0.5 * atan2(2 * a3, a1 - a2) * 180.0 / CV_PI; 
			}
		}

		filter_objects(src, labels, orientations.data(), TH_area, phi_LOW, phi_HIGH);
	}
}

void BFS(Mat src, Mat& colored)
{
	int height = src.rows;
	int width = src.cols;

	colored = Mat(height, width, CV_8UC3, Scalar(255, 255, 255));

	Mat labels = Mat::zeros(height, width, CV_32SC1);

	int current_label = 0;

	int dx8[] = { -1, -1, -1, 0, 1, 1, 1, 0 };
	int dy8[] = { -1, 0, 1, 1, 1, 0, -1, -1 };

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(50, 255);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
				current_label++;
				Vec3b color(dis(gen), dis(gen), dis(gen));

				std::queue<Point> q;
				labels.at<int>(i, j) = current_label;
				colored.at<Vec3b>(i, j) = color;
				q.push(Point(j, i));

				while (!q.empty()) {
					Point p = q.front();
					q.pop();

					for (int k = 0; k < 8; k++) {
						int nx = p.y + dx8[k];
						int ny = p.x + dy8[k];

						if (nx >= 0 && nx < height && ny >= 0 && ny < width) {
							if (src.at<uchar>(nx, ny) == 0 && labels.at<int>(nx, ny) == 0) {
								labels.at<int>(nx, ny) = current_label;
								colored.at<Vec3b>(nx, ny) = color;
								q.push(Point(ny, nx));

							}
						}
					}
				}
			}
		}
	}
}


void twoPass(Mat src, Mat & final, Mat& first_pass)
{
	int height = src.rows;
	int width = src.cols;

	final = Mat(height, width, CV_8UC3, Scalar(255, 255, 255));
	first_pass = Mat(height, width, CV_32SC1);

	Mat labels = Mat::zeros(height, width, CV_32SC1);

	std::vector<std::vector<int>>edges(1000);

	int current_label = 0;

	int dx8[] = { -1, -1, -1, 0, 1, 1, 1, 0 };
	int dy8[] = { -1, 0, 1, 1, 1, 0, -1, -1 };



	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
				std::vector <int> L;

				for (int k = 0; k < 8; k++) {
					int nj = j + dx8[k];
					int ni = i + dy8[k];

					if (ni >= 0 && ni < height && nj >= 0 && nj < width) {

						if (labels.at<int>(ni, nj) > 0)
						{
							L.push_back(labels.at<int>(ni, nj));
						}
					}
				}

				if (L.empty())
				{
					current_label++;
					labels.at<int>(i, j) = current_label;
				}
				else {
					int min_label = *std::min_element(L.begin(), L.end());
					labels.at<int>(i, j) = min_label;
					for (int label : L)
					{
						if (label != min_label)
						{
							edges[min_label].push_back(label);
							edges[label].push_back(min_label);
						}


					}
				}
			}
		}
	}
	labels.copyTo(first_pass);
	std::vector<int> new_labels(current_label + 1, 0);
	int new_label = 0;
	std::queue<int>Q;

	for (int i = 1; i <= current_label; i++)
	{
		if (new_labels[i] == 0)
		{
			new_label++;
			new_labels[i] = new_label;
			Q.push(i);

			while (!Q.empty())
			{
				int x = Q.front();
				Q.pop();

				for (int y : edges[x])
				{
					if (new_labels[y] == 0)
					{
						new_labels[y] = new_label;
						Q.push(y);
					}
				}
			}
		}
	}
	RNG rng(getTickCount());
	std::map<int, Vec3b>color_map;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int label = labels.at<int>(i, j);
			if (label > 0) {
				int final_label = new_labels[label];

				if (color_map.find(final_label) == color_map.end()) {
					color_map[final_label] = Vec3b(
						rng.uniform(50, 255),
						rng.uniform(50, 255),
						rng.uniform(50, 255));
				}

				final.at<Vec3b>(i, j) = color_map[final_label];
	}
}
}
}



void ex1Lb5() {
	char fname[MAX_PATH];
	if (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		if (src.empty()) {
			printf("Error loading image\n");
			return;
		}

		Mat colored;
		BFS(src, colored);

		imshow("Original", src);
		imshow("Colored Components", colored);
		waitKey(0);
	}
}

void ex3Lb5() {
	char fname[MAX_PATH];
	if (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		if (src.empty()) {
			printf("Error loading image\n");
			return;
		}

		Mat final,first_pass;
		twoPass(src, final,first_pass);

		imshow("Original", src);
		Mat first_pass_display;
		double minVal, maxVal;
		minMaxLoc(first_pass, &minVal, &maxVal);
		first_pass.convertTo(first_pass_display, CV_8UC1, 255.0 / maxVal, 0);

		imshow("First Pass Labels", first_pass_display);

		imshow("Final Labeled Components", final);		
		waitKey(0);
	}
}

void borderTracing(Mat src, Mat & final) {
	int height = src.rows;
	int width = src.cols;

	final = Mat(height, width, CV_8UC1, Scalar(255)); 

	int dx8[] = { -1, 0, 1, 0 };
	int dy8[] = { 1,  0,  1,  1, 1, 0, -1, -1 };

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 0) {  
				std::cout << "Starting border tracing at (" << j << ", " << i << ")\n";

				std::vector<Point> contour;
				int x = j, y = i;
				int startX = x, startY = y;
				int dir = 7;  // Start direction
				bool firstIteration = true;
				bool loopCompleted = false;

				while (!loopCompleted) {
					contour.push_back(Point(x, y));
					final.at<uchar>(y, x) = 0;  

					bool foundNext = false;
					int searchDir;
					if (dir % 2 == 0) {
						searchDir = (dir + 7) % 8;  // if dir is even
					}
					else {
						searchDir = (dir + 6) % 8;  // if dir is odd
					}

					for (int k = 0; k < 8; k++) {
						int newDir = (searchDir + k) % 8;  
						int nx = x + dx8[newDir];
						int ny = y + dy8[newDir];

						if (nx >= 0 && nx < width && ny >= 0 && ny < height && src.at<uchar>(ny, nx) == 0) {
							x = nx;
							y = ny;
							dir = newDir;  
							foundNext = true;

							if (firstIteration) {
								firstIteration = false;
							}
							else if (x == startX && y == startY) {
								loopCompleted = true; 
							}

							break; 
						}
					}

					if (!foundNext) {
						std::cout << "No next border pixel found. Stopping at (" << x << ", " << y << ")\n";
						break;
					}
				}
				return; 
			}
		}
	}
}


void borderChain(Mat src, Mat & final, std::vector<int> &chainCode) {
	int height = src.rows;
	int width = src.cols;

	final = Mat(height, width, CV_8UC1, Scalar(255));

	int dx8[] = { -1, -1, -1,  0, 1, 1, 1, 0 };
	int dy8[] = { -1,  0,  1,  1, 1, 0, -1, -1 };

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 0) {
				std::cout << "Starting border tracing at (" << j << ", " << i << ")\n";

				std::vector<Point> contour;
				int x = j, y = i;
				int startX = x, startY = y;
				int dir = 7;  
				bool firstIteration = true;
				bool loopCompleted = false;
				chainCode.clear();


				while (!loopCompleted) {
					contour.push_back(Point(x, y));
					final.at<uchar>(y, x) = 0;

					bool foundNext = false;
					int searchDir = (dir % 2 == 0) ? (dir + 7) % 8 : (dir + 6) % 8;

					for (int k = 0; k < 8; k++) {
						int newDir = (searchDir + k) % 8;
						int nx = x + dx8[newDir];
						int ny = y + dy8[newDir];

						if (nx >= 0 && nx < width && ny >= 0 && ny < height && src.at<uchar>(ny, nx) == 0) {
							x = nx;
							y = ny;
							dir = newDir;
							chainCode.push_back(newDir);
							foundNext = true;

							if (firstIteration) {
								firstIteration = false;
							}
							else if (x == startX && y == startY) {
								loopCompleted = true;
							}

							break;
						}
					}

					if (!foundNext) {
						std::cout << "No next border pixel found. Stopping at (" << x << ", " << y << ")\n";
						break;
					}
				}
				std::cout << "Chain code:";
				for (int i = 0; i < chainCode.size(); i++) {
					std::cout << chainCode[i] << " ";
				}
				std::cout << std::endl;
				return;
			}
		}
	}
	

}

void ex1Lb6() {


	char fname[MAX_PATH];
	if (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_COLOR);
		if (src.empty()) {
			printf("Error loading image\n");
			return;
		}
		Mat dst;
		borderTracing(src, dst);
		std::cout << "Displaying result image..." << std::endl;
		if (dst.empty()) {
			std::cout << "Error: dst is empty!" << std::endl;
		}
		imshow("Original", src);
		imshow("Traced", dst);
		waitKey(0);
	}

}

void ex2Lb6()
{
	char fname[MAX_PATH];
	if (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		if (src.empty()) {
			printf("Error loading image\n");
			return;
		}
		Mat dst;
		std::vector <int>chainCode;
	   borderChain(src, dst,chainCode);
		imshow("Original", src);
		imshow("Countour", dst);
		waitKey(0);
	}
}


void dilation()
{
	char fname[MAX_PATH];
	int di4[] = { -1, 0, 1, 0 };
	int dj4[] = { 0, 1, 0, -1 };

	if (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1, Scalar(255));

		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				uchar pixel = src.at<uchar>(i, j);
				if (pixel == 0)
				{
					dst.at<uchar>(i, j) = 0;

					for (int k = 0; k < 4; k++)
					{
						int nx = j + di4[k];
						int ny = i + dj4[k];

						if (ny >= 0 && ny < height && nx >= 0 && nx < width)
						{
							dst.at<uchar>(ny, nx) = 0;
						}


					}

				}
			}
		}
		imshow("Original", src);
		imshow("Dilation", dst);
		waitKey(0);

	}
}



void erosion()
{
	char fname[MAX_PATH];
	int di4[] = { -1, 0, 1, 0 };
	int dj4[] = { 0, 1, 0, -1 };

	if (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1, Scalar(255));

		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				uchar pixel = src.at<uchar>(i, j);
				if (pixel == 0)
				{
					dst.at<uchar>(i, j) = 0;


					for (int k = 0; k < 4; k++)
					{
						int nx = j + di4[k];
						int ny = i + dj4[k];

						if (ny >= 0 && ny < height && nx >= 0 && nx < width && src.at<uchar>(ny,nx)==0)
						{
							dst.at<uchar>(ny, nx) = 255;
						}


					}

				}
			}
		}
		imshow("Original", src);
		imshow("Erosion", dst);
		waitKey(0);

	}
}

void erosion2(Mat& dst)
{
	char fname[MAX_PATH];
	int di4[] = { -1, 0, 1, 0 };
	int dj4[] = { 0, 1, 0, -1 };

	if (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;


		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				uchar pixel = src.at<uchar>(i, j);
				if (pixel == 0)
				{
					dst.at<uchar>(i, j) = 0;


					for (int k = 0; k < 4; k++)
					{
						int nx = j + di4[k];
						int ny = i + dj4[k];

						if (ny >= 0 && ny < height && nx >= 0 && nx < width && src.at<uchar>(ny, nx) == 0)
						{
							dst.at<uchar>(ny, nx) = 255;
						}


					}

				}
			}
		}
		

	}
}
Mat dilateImage(const Mat& src)
{
	int di4[] = { -1, 0, 1, 0 };
	int dj4[] = { 0, 1, 0, -1 };

	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height, width, CV_8UC1, Scalar(255));

	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			uchar pixel = src.at<uchar>(i, j);
			if (pixel == 0)
			{
				dst.at<uchar>(i, j) = 0;

				for (int k = 0; k < 4; k++)
				{
					int nx = j + di4[k];
					int ny = i + dj4[k];

					if (ny >= 0 && ny < height && nx >= 0 && nx < width)
					{
						dst.at<uchar>(ny, nx) = 0;
					}
				}
			}
		}
	}

	return dst;
}

Mat erodeImage(const Mat& src)
{
	int di4[] = { -1, 0, 1, 0 };
	int dj4[] = { 0, 1, 0, -1 };

	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height, width, CV_8UC1, Scalar(255));

	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			bool isObject = true;

			if (src.at<uchar>(i, j) != 0)
				continue;

			for (int k = 0; k < 4; k++)
			{
				int nx = j + di4[k];
				int ny = i + dj4[k];

				if (ny >= 0 && ny < height && nx >= 0 && nx < width)
				{
					if (src.at<uchar>(ny, nx) != 0)
					{
						isObject = false;
						break;
					}
				}
			}

			if (isObject)
				dst.at<uchar>(i, j) = 0;
		}
	}

	return dst;
}




void opening()
{
	char fname[MAX_PATH];

	if (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		Mat eroded = erodeImage(src);
		Mat opened = dilateImage(eroded);

		imshow("Original", src);
		imshow("Opening (Erosion + Dilation)", opened);
		waitKey(0);
	}
}

void closing()
{
	char fname[MAX_PATH];

	if (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		
		Mat dilate = dilateImage(src);
		Mat eroded = erodeImage(dilate);

		imshow("Original", src);
		imshow("Closing ( Dilation + Erosion)", eroded);
		waitKey(0);
	}

}
void dilationNTimes()
{
	char fname[MAX_PATH];
	int n;
	if (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1, Scalar(255));
		printf("Enter number of iterations: ");
		scanf("%d", &n);
		Mat src2 = src.clone();
		for (int i = 0; i < n; i++)
		{
			dst = dilateImage(src);
			src = dst.clone();
		}
		imshow("Original", src2);
		imshow("Dilation N times", dst);
		waitKey(0);
	}
}
void erosionNTimes() {
	char fname[MAX_PATH];
	int n;
	if (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1, Scalar(255));
		printf("Enter number of iterations: ");
		scanf("%d", &n);
		Mat src2 = src.clone();
		for (int i = 0; i < n; i++)
		{
			dst = erodeImage(src);
			src = dst.clone();
		}
		imshow("Original", src2);
		imshow("Erosion N times", dst);
		waitKey(0);
	}
}

void openingNTimes() {
	char fname[MAX_PATH];
	int n;
	if (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1, Scalar(255));
		printf("Enter number of iterations: ");
		scanf("%d", &n);
		Mat src2 = src.clone();
		for (int i = 0; i < n; i++)
		{
			dst = erodeImage(src);
			src = dst.clone();
			dst = dilateImage(src);
			src = dst.clone();
		}
		imshow("Original", src2);
		imshow("Opening N times", dst);
		waitKey(0);
	}

}

void closingNTimes()
{
	char fname[MAX_PATH];
	int n;
	if (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1, Scalar(255));
		printf("Enter number of iterations: ");
		scanf("%d", &n);
		Mat src2 = src.clone();
		for (int i = 0; i < n; i++)
		{
			dst = dilateImage(src);
			src = dst.clone();
			dst = erodeImage(src);
			src = dst.clone();
		}
		imshow("Original", src2);
		imshow("Closing N times", dst);
		waitKey(0);
	}
}


Mat erodeImage8(const Mat& src)
{


	int dj8[] = { -1, -1, -1,  0, 1, 1, 1, 0 };
	int di8[] = { -1,  0,  1,  1, 1, 0, -1, -1 };

	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height, width, CV_8UC1, Scalar(255));

	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			bool isObject = true;

			if (src.at<uchar>(i, j) != 0)
				continue;

			for (int k = 0; k < 8; k++)
			{
				int nx = j + di8[k];
				int ny = i + dj8[k];

				if (ny >= 0 && ny < height && nx >= 0 && nx < width)
				{
					if (src.at<uchar>(ny, nx) != 0)
					{
						isObject = false;
						break;
					}
				}
			}

			if (isObject)
				dst.at<uchar>(i, j) = 0;
		}
	}

	return dst;
}


void boundaryExtraction()
{
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1, Scalar(255));
		Mat eroded = erodeImage8(src);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == 0 && eroded.at<uchar>(i, j) == 255)
				{
					dst.at<uchar>(i, j) = 0;
				}
			}
		}
		imshow("Original", src);
		imshow("Boundary Extraction", dst);
		waitKey(0);
	}
}

void regionFilling()
{
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat complement_src = Mat(src.size(), src.type(), Scalar(255));
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1, Scalar(255));
		Mat dilated = dilateImage(src);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < height; j++)
			{
				if (src.at<uchar>(i, j) == 255)
				{
					complement_src.at<uchar>(i, j) = 0;
				}
				else
				{
					complement_src.at<uchar>(i, j) = 255;
				}
			}
		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (dilated.at<uchar>(i, j) == 0 && complement_src.at<uchar>(i, j) == 0)
				{
					dst.at<uchar>(i, j) = 0;
				}
				else
				{
					dst.at<uchar>(i, j) = 255;
				}
			}
		}
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == 0 || dst.at<uchar>(i, j) == 0)
				{
					dst.at<uchar>(i, j) = 0;
				}
				else
				{
					dst.at<uchar>(i, j) = 255;
				}
			}
		}
		imshow("Original", src);
		imshow("Region Filling", dst);
		waitKey(0);
	}
}

void meanAndStdDev()
{
	double mean = 0, std = 0;

	char fname[MAX_PATH];
	int PDF[256] = { 0 };

	if (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		if (src.empty()) {
			std::cout << "Error loading image!\n";
			return;
		}

		int hist[256] = { 0 };
		int cumulativeHist[256] = { 0 };
		int totalPixels = src.rows * src.cols;

		// Compute histogram
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				hist[src.at<uchar>(i, j)]++;
			}
		}

		// Mean
		for (int g = 0; g < 256; g++) {
			mean += g * hist[g];
		}
		mean /= totalPixels;

		// PDF
		for (int i = 0; i < 256; i++) {
			PDF[i] = static_cast<float>(hist[i]) / totalPixels;
		}

		// Standard Deviation
		for (int g = 0; g < 256; g++) {
			std += (g - mean) * (g - mean) * PDF[g];
		}
		std = sqrt(std);

		// Cumulative Histogram
		cumulativeHist[0] = hist[0];
		for (int g = 1; g < 256; g++) {
			cumulativeHist[g] = cumulativeHist[g - 1] + hist[g];
		}
		std::cout << "Mean: " << mean << std::endl;
		std::cout << "Standard Deviation: " << std << std::endl;


		showHistogram("Histogram", hist, 256, 300);
		showHistogram("Cumulative Histogram", cumulativeHist, 256, 300);
		waitKey(0);


	

		waitKey(0);
	}
}

void automaticThreshold()
{
	char fname[MAX_PATH];
	if (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		if (src.empty()) {
			std::cout << "Error loading image!\n";
			return;
		}

		int hist[256] = { 0 };
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++)
				hist[src.at<uchar>(i, j)]++;

		int totalPixels = src.rows * src.cols;
		double T = 0, Tprev = -1;

		for (int g = 0; g < 256; g++)
			T += g * hist[g];
		T /= totalPixels;

		while (abs(T - Tprev) >= 1.0) {
			Tprev = T;
			double m1 = 0, m2 = 0;
			int w1 = 0, w2 = 0;
			for (int g = 0; g <= T; g++) {
				m1 += g * hist[g];
				w1 += hist[g];
			}
			for (int g = T + 1; g < 256; g++) {
				m2 += g * hist[g];
				w2 += hist[g];
			}
			m1 = w1 ? m1 / w1 : 0;
			m2 = w2 ? m2 / w2 : 0;
			T = (m1 + m2) / 2.0;
		}

		int thresholdVal = (int)T;
		Mat dst;
		threshold(src, dst, thresholdVal, 255, THRESH_BINARY);

		std::cout << "Automatic Threshold: " << thresholdVal << "\n";
		imshow("Original", src);
		imshow("Thresholded", dst);
		waitKey(0);
	}
}

void stretchHistogram()
{
	char fname[MAX_PATH];
	int gmin, gmax;
	std::cout << "Enter gmin and gmax: ";
	std::cin >> gmin >> gmax;

	if (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		if (src.empty()) return;

		double minVal, maxVal;
		minMaxLoc(src, &minVal, &maxVal);
		Mat dst = (src - minVal) * ((gmax - gmin) / (maxVal - minVal)) + gmin;
		dst.convertTo(dst, CV_8UC1);

		int histSrc[256] = { 0 }, histDst[256] = { 0 };
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++) {
				histSrc[src.at<uchar>(i, j)]++;
				histDst[dst.at<uchar>(i, j)]++;
			}

		showHistogram("Source Histogram", histSrc, 256, 300);
		showHistogram("Stretched Histogram", histDst, 256, 300);
		imshow("Stretched Image", dst);
		waitKey(0);
	}
}

void shrinkHistogram()
{
	char fname[MAX_PATH];
	int gmin, gmax;
	std::cout << "Enter gmin and gmax: ";
	std::cin >> gmin >> gmax;

	if (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		if (src.empty()) return;

		double minVal, maxVal;
		minMaxLoc(src, &minVal, &maxVal);
		Mat dst = (src - minVal) * ((gmax - gmin) / (maxVal - minVal)) + gmin;
		dst.convertTo(dst, CV_8UC1);

		int histSrc[256] = { 0 }, histDst[256] = { 0 };
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++) {
				histSrc[src.at<uchar>(i, j)]++;
				histDst[dst.at<uchar>(i, j)]++;
			}

		showHistogram("Source Histogram", histSrc, 256, 300);
		showHistogram("Shrink Histogram", histDst, 256, 300);
		imshow("Shrink Image", dst);
		waitKey(0);
	}
}

void gammaCorrection()
{
	char fname[MAX_PATH];
	double gamma;
	std::cout << "Enter gamma: ";
	std::cin >> gamma;

	if (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		if (src.empty()) return;

		Mat dst = src.clone();
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++) {
				dst.at<uchar>(i, j) = saturate_cast<uchar>(pow(src.at<uchar>(i, j) / 255.0, gamma) * 255.0);
			}

		int histSrc[256] = { 0 }, histDst[256] = { 0 };
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++) {
				histSrc[src.at<uchar>(i, j)]++;
				histDst[dst.at<uchar>(i, j)]++;
			}

		showHistogram("Source Histogram", histSrc, 256, 300);
		showHistogram("Gamma Histogram", histDst, 256, 300);
		imshow("Gamma Corrected", dst);
		waitKey(0);
	}
}
void histogramEq()
{
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		if (src.empty()) {
			std::cout << "Error loading image!\n";
			return;
		}

		int hist[256] = { 0 };
		double PR[256] = { 0.0 };
		double PC[256] = { 0.0 };
		int map[256] = { 0 };

		int totalPixels = src.rows * src.cols;

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				hist[src.at<uchar>(i, j)]++;
			}
		}

		for (int i = 0; i < 256; i++) {
			PR[i] = (double)hist[i] / totalPixels;
		}

		PC[0] = PR[0];
		for (int i = 1; i < 256; i++) {
			PC[i] = PC[i - 1] + PR[i];
		}

		for (int i = 0; i < 256; i++) {
			map[i] = cvRound(PC[i] * 255); 
		}

		Mat dst = src.clone();
		for (int i = 0; i < dst.rows; i++) {
			for (int j = 0; j < dst.cols; j++) {
				dst.at<uchar>(i, j) = map[src.at<uchar>(i, j)];
			}
		}

		imshow("Original", src);
		imshow("Equalized", dst);

		int newHist[256] = { 0 };
		for (int i = 0; i < dst.rows; i++) {
			for (int j = 0; j < dst.cols; j++) {
				newHist[dst.at<uchar>(i, j)]++;
			}
		}

		showHistogram("Original Histogram", hist, 256, 300);
		showHistogram("Equalized Histogram", newHist, 256, 300);
		waitKey(0);
	}
}




/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/


int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative\n");
		printf(" 4 - Image negative (fast)\n");
		printf(" 5 - BGR->Gray\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 7 - BGR->HSV\n");
		printf(" 8 - Resize image\n");
		printf(" 9 - Canny edge detection\n");
		printf(" 10 - Edges in a video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");
		printf(" 13 - Change gray level by adding additive\n");
		printf(" 14 - Change gray level by adding mutiplicative factor\n");
		printf(" 15 - The 4 squares problem\n");
		printf(" 16 - Inverse\n");
		printf(" 17 - Ex1 Lb2\n");
		printf(" 18 - Ex2 Lb2\n");
		printf(" 19 - Ex1 Lb3\n");
		printf("20 - Ex1_a Lb4 \n");
		printf("21 - Ex2Lb4 \n");
		printf("22- Ex1Lb5 \n");
		printf("23 - Ex3Lb5 \n");
		printf("24- Ex1Lb6 \n");
		printf("25 - E2Lb6 \n");
		printf("26 - Dilation\n");
		printf("27 - Erosion\n");
		printf("28 - Opening\n");
		printf("29 - Closing\n");
		printf("30 - Dilation n times\n");
		printf("31 - Erosion n times\n");
		printf("32 - Opening n times\n");
		printf("33 - Closing n times\n");
		printf("34- Boundary Extraction\n");
		printf("35 - Region Filling\n");
		printf("36 - Mean and Standard deviation\n ");
		printf("37 - Automatic Thresholding\n");
		printf("38 -  Histogram Stretch\n");
		printf("39 - Histogram Shrink\n");
		printf("40 - Gamma correction\n");
		printf("41 - Histogram Equilization\n");

        


		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testNegativeImage();
			break;
		case 4:
			testNegativeImageFast();
			break;
		case 5:
			testColor2Gray();
			break;
		case 6:
			testImageOpenAndSave();
			break;
		case 7:
			testBGR2HSV();
			break;
		case 8:
			testResize();
			break;
		case 9:
			testCanny();
			break;
		case 10:
			testVideoSequence();
			break;
		case 11:
			testSnap();
			break;
		case 12:
			testMouseClick();
			break;
		case 13:
			grayAdditive();
			break;
		case 14:
			grayMultiplicative();
			break;
		case 15:
			the4Squares();
			break;
		case 16:
			inverseMatrix();
			break;
		case 17:
			ex1LB2();
			break;
		case 18:
			ex2LB2();
			break;
		case 19:
			ex3LB2();
			break;

		case 20:
			ex1aLb4();
			break;

		case 21:
			ex2Lb4();
			break;

		case 22:
			ex1Lb5();
			break;
		case 23:
			ex3Lb5();
			break;
		case 24:
			ex1Lb6();
			break;

		case 25:
			ex2Lb6();
			break;

		case 26:
			dilation();

		case 27:
			erosion();

		case 28:
			opening();

		case 29:
			closing();
			break;
		case 30:
			dilationNTimes();
			break;
		case 31:
			erosionNTimes();
			break;

		case 32:
			openingNTimes();
			break;
		case 33:
			closingNTimes();
			break;
		case 34:
			boundaryExtraction();
			break;
		case 35:
			regionFilling();
			break;
		case 36:
			meanAndStdDev();
			break;
		case 37:
			automaticThreshold();
			break;

		case 38:
			stretchHistogram();
			break;
		case 39:
			shrinkHistogram();
		case 40:
			gammaCorrection();

		case 41:
			histogramEq();


		}
	} while (op != 0);
	return 0;
}