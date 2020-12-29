#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

enum class colorSpace { HSV, HSL, ycrcb, LAB};

/******************************
** intensity transform funcs **
*******************************/
void linearTransform(const Mat&, Mat&, double alpha = 1.0, double beta = 0);
Mat calcHist(const Mat&);
Mat equalHist(Mat, colorSpace);
Mat equalHistRGB(Mat);
void normalize(Mat, Mat&);
void gammaCorrection(const Mat&, Mat&, double gamma = 1.0);
Mat changeSaturation(const Mat&, double alpha = 1.0, double beta = 0);

Mat applyCLAHE(const Mat& input, double clipLimit = 40.0, Size tile_grid_size = Size(8,8));
void showHistogram(Mat);

Mat avgBlur(const Mat&, Size, int border_type = BORDER_DEFAULT);
Mat medBlur(const Mat&, Size, int border_type = BORDER_DEFAULT);
void conv2D(Mat, Mat, Mat&, int border_type = BORDER_DEFAULT);
Mat gaussBlur(Mat, Size , float, int border_type = BORDER_DEFAULT);
Mat getCloseWeight(double, Size);
Mat bilateralFilter(const Mat&, Size, float, float);
Mat sharpen(Mat);

void img1Process(Mat, Mat&);
void img2Process(Mat, Mat&);
void img3Process(Mat, Mat&);
void img4Process(Mat, Mat&);
void img5Process(Mat, Mat&);
void img6Process(Mat, Mat&);

int main(int argc, const char** argv) {		
	// Read the source images
	Mat images[6];	
	for (int i = 1; i <= 6; i++) {
		string file_name("../images/p1im");
		file_name = file_name + to_string(i) + ".png";
		images[i - 1] = imread(file_name);
		if (!images[i-1].data) {
			cout << "Load image " << i << " failed! Please check and try again." << endl;
			return EXIT_FAILURE;
		}
	}
	/* 
	   we can comment other lines to 
	   select one of image processing
	   to execute.
	*/
	Mat results[6], img_show;
	img1Process(images[0], results[0]);
	img2Process(images[1], results[1]);
	img3Process(images[2], results[2]);
	img4Process(images[3], results[3]);
	img5Process(images[4], results[4]);
	img6Process(images[5], results[5]);

	/*
	   Modify the index to show the 
	   result of the corresponding 
	   processing result.	
	*/
	int index = 0;
	hconcat(images[index], results[index], img_show);
	imshow("Result", img_show);
	int key = waitKey(0);
	if (key == 27)
		destroyAllWindows();
	
	return EXIT_SUCCESS;
}

/* image 1 ~ 6 processings */
void img1Process(Mat img, Mat& output)
{	
	Mat _img;
	// smoothing
	_img = bilateralFilter(img, Size(3, 3), 20, 20);
	_img = medBlur(img, Size(3, 3), BORDER_REPLICATE);
	// increase saturation
	_img = changeSaturation(img, 1.2, 0);
	// intensity transformation
	gammaCorrection(_img, _img, 2.5);
	_img = equalHist(_img, colorSpace::ycrcb);
	gammaCorrection(_img, _img, 0.6);
	linearTransform(_img, _img, 1.0, 40);
	
	output = _img;
}

void img2Process(Mat img, Mat& output)
{
	Mat _img;
	_img = gaussBlur(img, Size(3, 3), 1.0, BORDER_REPLICATE);	
	linearTransform(_img, _img, 3.5, -70);	
	_img = sharpen(_img);	
	output = _img;
}

void img3Process(Mat img, Mat& output)
{
	Mat _img;
	// smoothing
	_img = gaussBlur(img, Size(3, 3), 1.0, BORDER_REPLICATE);
	// histogram normalize
	normalize(img, _img);
	// increase saturation
	_img = changeSaturation(_img, 1.0, 30);
	// adjust intensity 
	linearTransform(_img, _img, 1.0, -10);
	// sharpen the edge
	_img = sharpen(_img);
	_img = sharpen(_img);
	output = _img;
}

void img4Process(Mat input, Mat& output)
{
	Mat _img;
	// intensity adjustment in HSV
	cvtColor(input, _img, COLOR_BGR2HSV);
	vector<Mat> channels;
	split(_img, channels);
	linearTransform(channels[2], channels[2], 1.5, 50);
	merge(channels, _img);
	cvtColor(_img, _img, COLOR_HSV2BGR);
	// contrast adjustment
	gammaCorrection(_img, _img, 1.2);
	// sharpen
	_img = sharpen(_img);
	output = _img;
}

void img5Process(Mat img, Mat& output)
{
	Mat _img;
	_img = equalHistRGB(img);	
	linearTransform(_img, _img, 1.5, 20);
	_img = sharpen(_img);
	output = _img;
}

void img6Process(Mat img, Mat& output)
{
	Mat _img;
	// smoothing
	bilateralFilter(img, _img, 20, 50, 50);
	output = _img;
}

/******************************
** intensity transform funcs **
*******************************/

void linearTransform(const Mat& input, Mat& output, double alpha, double beta)
{
	Mat _img = input.clone();
	_img.convertTo(_img, CV_32F);
	output = (_img / 255.f) * alpha + beta/255.f;
	output *= 255.f;
	output.convertTo(output, CV_8U);
}

Mat calcHist(const Mat& img)
{
	Mat histogram = Mat::zeros(Size(256, 1), CV_32SC1);
	const int rows = img.rows;
	const int cols = img.cols;

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			int index = int(img.at<uchar>(r, c));
			histogram.at<int>(0, index) += 1;
		}
	}
	return histogram;
}

void showHistogram(Mat img)
{
	vector<Mat> img_channels;
	split(img, img_channels);

	// 256 bins (the number of possibles values)
	const int num_bins = 256;

	// Set the ranges for B,G,R
	float range[] = { 0, 256 };
	const float* hist_range = { range };

	Mat b_hist, g_hist, r_hist;

	cv::calcHist(&img_channels[0], 1, 0, Mat(), b_hist, 1, &num_bins, &hist_range);
	cv::calcHist(&img_channels[1], 1, 0, Mat(), g_hist, 1, &num_bins, &hist_range);
	cv::calcHist(&img_channels[2], 1, 0, Mat(), r_hist, 1, &num_bins, &hist_range);

	// Draw the histogram
	const int width = 512;
	const int height = 300;
	Mat hist_image(height, width, CV_8UC3, Scalar(20, 20, 20));

	// Normalize the histograms to height of image
	normalize(b_hist, b_hist, 0, height, NORM_MINMAX);
	normalize(g_hist, g_hist, 0, height, NORM_MINMAX);
	normalize(r_hist, r_hist, 0, height, NORM_MINMAX);

	// draw lines for each channel
	int bin_step = cvRound((float)width / (float)num_bins);
	for (int i = 1; i < num_bins; i++)
	{
		line(hist_image,
			Point(bin_step * (i - 1), height - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_step * (i), height - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0)
		);
		line(hist_image,
			Point(bin_step * (i - 1), height - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_step * (i), height - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0)
		);
		line(hist_image,
			Point(bin_step * (i - 1), height - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_step * (i), height - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255)
		);
	}
	imshow("Histogram", hist_image);
	waitKey(0);
}

Mat equalHist(Mat img, colorSpace space)
{
	int code, code_r, channel;
	switch (space) {
		case colorSpace::HSV:
			code = COLOR_BGR2HSV;
			code_r = COLOR_HSV2BGR;
			break;
		case colorSpace::HSL:
			code = COLOR_BGR2HLS;
			code_r = COLOR_HLS2BGR;
			break;
		case colorSpace::ycrcb:
			code = COLOR_BGR2YCrCb;
			code_r = COLOR_YCrCb2BGR;
			break;
		case colorSpace::LAB:
			code = COLOR_BGR2Lab;
			code_r = COLOR_Lab2BGR;
			break;
	}
	vector<Mat> img_channels;
	Mat intensity, img_cvt;
	cvtColor(img, img_cvt, code);
	split(img_cvt, img_channels);
	channel = (space == colorSpace::ycrcb || space == colorSpace::LAB) ? 0 : 2;

	intensity = img_channels[channel];
	CV_Assert(intensity.type() == CV_8UC1);
	const int rows = intensity.rows;
	const int cols = intensity.cols;
	// calculate image's histogram
	Mat hist = calcHist(intensity);

	// calculate cdf
	Mat cdf = Mat::zeros(Size(256, 1), CV_32SC1);
	cdf.at<int>(0, 0) = hist.at<int>(0, 0);
	for (int p = 1; p < 256; p++) {
		cdf.at<int>(0, p) = cdf.at<int>(0, p - 1) + hist.at<int>(0, p);
	}

	Mat lut_table(1, 256, CV_8UC1);
	float coff = 256.0 / (rows * cols);
	for (int p = 0; p < 256; p++) {
		float q = coff * cdf.at<int>(0, p) - 1;
		if (q >= 0)
			lut_table.at<uchar>(p) = uchar(floor(q));
		else
			lut_table.at<uchar>(p) = 0;
	}

	Mat intensity_e, output;
	LUT(intensity, lut_table, intensity_e);
	img_channels[channel] = intensity_e;
	merge(img_channels, output);
	cvtColor(output, output, code_r);

	return output;
}

Mat equalHistRGB(Mat img)
{
	vector<Mat> img_channels;
	split(img, img_channels);
	Mat intensity = img_channels[0];
	const int rows = intensity.rows;
	const int cols = intensity.cols;
	
	for (Mat& m : img_channels) {
		// calculate image's histogram
		Mat hist = calcHist(m);
		// calculate cdf
		Mat cdf = Mat::zeros(Size(256, 1), CV_32SC1);
		for (int p = 0; p < 256; p++) {
			if (p == 0)
				cdf.at<int>(0, p) = hist.at<int>(0, 0);
			else
				cdf.at<int>(0, p) = cdf.at<int>(0, p - 1) + hist.at<int>(0, p);
		}
		Mat lut_table(1, 256, CV_8UC1);
		float coff = 256.0 / (rows * cols);
		for (int p = 0; p < 256; p++) {
			float q = coff * cdf.at<int>(0, p) - 1;
			if (q >= 0)
				lut_table.at<uchar>(p) = uchar(floor(q));
			else
				lut_table.at<uchar>(p) = 0;
		}
		LUT(m, lut_table, m);
	}
	Mat output;
	merge(img_channels, output);

	return output;
}

void normalize(Mat img, Mat& output)
{
	double input_max, input_min;
	minMaxLoc(img, &input_min, &input_max, NULL, NULL);
	const double output_min = 0, output_max = 255;
	const double a = (output_max - output_min) / (input_max - input_min);
	const double b = output_min - a * input_min;
	linearTransform(img, output, a, b);
}

void gammaCorrection(const Mat& input, Mat& output, double gamma)
{
	Mat lut_table(1, 256, CV_8UC1);
	for (int i = 0; i < 256; i++) {
		lut_table.at<uchar>(i) =  (pow((double)(i / 255.0), gamma)) * 255;
	}
	LUT(input, lut_table, output);
}

Mat changeSaturation(const Mat& input, double alpha, double beta) 
{
	Mat _img;
	cvtColor(input, _img, COLOR_BGR2HSV);
	vector<Mat> imgs;
	split(_img, imgs);
	Mat saturation = imgs[1];
	saturation.convertTo(saturation, CV_32FC1);
	saturation = (saturation / 255.f) * alpha + beta / 255.f;
	saturation *= 255;
	saturation.convertTo(saturation, CV_8UC1);
	imgs[1] = saturation;
	merge(imgs, _img);
	cvtColor(_img, _img, COLOR_HSV2BGR);
	return _img;
}

Mat applyCLAHE(const Mat& input, double clipLimit, Size tile_grid_size)
{
	Mat _img, output;
	cvtColor(input, _img, COLOR_BGR2HSV);
	vector<Mat> img_channels;
	split(_img, img_channels);
	// apply CLAHE
	Ptr<CLAHE> clahe = createCLAHE(clipLimit, tile_grid_size);
	clahe->apply(img_channels[2], img_channels[2]);
	merge(img_channels, _img);
	cvtColor(_img, output, COLOR_HSV2BGR);
	return output;
}

/******************************
**     denoise functions     **
*******************************/

Mat avgBlur(const Mat& input, Size ksize, int border_type)
{
	CV_Assert(input.channels() == 3);
	const int rows = input.rows;
	const int cols = input.cols;
	const int h = (ksize.height - 1) / 2;
	const int w = (ksize.width - 1) / 2;
	Mat output(rows, cols, CV_32FC3), img_border, region;
	copyMakeBorder(input, img_border, h, h, w, w, border_type);

	for (int r = h; r < h + rows; r++) {
		for (int c = w; c < w + cols; c++) {
				region = img_border.rowRange(Range(r - h, r - h + ksize.height))
								   .colRange(Range(c - w, c - w + ksize.width));
				Scalar region_mean = mean(region);
				output.at<Vec3f>(r - h , c - w) = Vec3f(region_mean.val[0], region_mean.val[1], region_mean.val[2]);
		}
	}
	output.convertTo(output, CV_8UC3);
	return output;
}

Mat medBlur(const Mat& input, Size ksize, int border_type)
{
	const int H = ksize.height;
	const int W = ksize.width;
	CV_Assert(H > 0 && W > 0);
	CV_Assert(H % 2 == 1 && W % 2 == 1);
	const int h = (H - 1) / 2;
	const int w = (W - 1) / 2;
	
	const int rows = input.rows;
	const int cols = input.cols;
	int i = 0, j = 0;
	int index = (H * W - 1) / 2;

	vector<Mat> img_chs;
	split(input, img_chs);
	Mat b(input.size(), CV_8UC1), g(input.size(), CV_8UC1), r(input.size(), CV_8UC1);
	vector<Mat> channels{ b, g, r };
	Mat output(input.size(), CV_8UC1);

	for (int cn = 0; cn < 3; cn++) {
		copyMakeBorder(img_chs[cn], img_chs[cn], h, h, w, w, border_type);
		for (int r = h; r < h + rows; r++) {
			for (int c = w; c < w + cols; c++) {
				Mat kernel = img_chs[cn](Rect(c - w, r - h, W, H)).clone().reshape(1, 1);
				cv::sort(kernel, kernel, SORT_EVERY_ROW);
				channels[cn].at<uchar>(i, j) = kernel.at<uchar>(0, index);
				j++;
			}
			i++;
			j = 0;
		}
		i = j = 0;
	}
	merge(channels, output);
	return output;
}

void conv2D(Mat input, Mat kernel, Mat& output, int border_type)
{
	CV_Assert(input.channels() == 3);
	const int rows = input.rows;
	const int cols = input.cols;
	const int kernel_h = kernel.rows;
	const int kernel_w = kernel.cols;
	const int h = (kernel_h - 1) / 2;
	const int w = (kernel_w - 1) / 2;

	Mat region;
	vector<Mat> input_chs;
	split(input, input_chs);
	for (Mat& m : input_chs) {
		copyMakeBorder(m, m, h, h, w, w, border_type);
	}

	Mat b(input.size(), CV_64FC1), g(input.size(), CV_64FC1), r(input.size(), CV_64FC1);
	vector<Mat> channels{ b, g, r };

	for (int ch = 0; ch < 3; ch++) {
		for (int r = h; r < h + rows; r++) {
			for (int c = w; c < w + cols; c++) {
				input_chs[ch](Rect(c - w, r - h, kernel_w, kernel_h)).convertTo(region, CV_64FC1);				
				channels[ch].at<double>(r - h, c - w) = (double) region.dot(kernel);
			}
		}
	}
	merge(channels, output);
}

Mat gaussBlur(Mat input, Size ksize, float sigma, int border_type)
{
	CV_Assert(input.channels() == 3);
	CV_Assert(ksize.width % 2 == 1 && ksize.height % 2 == 1);
	// construct gaussian kernel in y and x direction
	Mat gaussKernel_y = getGaussianKernel(ksize.height, sigma, CV_64F);
	Mat gaussKernel_x = getGaussianKernel(ksize.width, sigma, CV_64F);
	gaussKernel_x = gaussKernel_x.t();
	// separable gaussian convolution
	Mat output(input.size(), CV_64FC3), conv_y(input.size(), CV_64FC3);
	conv2D(input, gaussKernel_y, conv_y, BORDER_REPLICATE);
	conv2D(conv_y, gaussKernel_x, output, BORDER_REPLICATE);
	output.convertTo(output, CV_8UC3);

	return output;
} 

Mat getCloseWeight(double sigma_g, Size size)
{
	const int height = size.height;
	const int width = size.width;
	const int h = (height - 1) / 2;
	const int w = (width - 1) / 2;
	Mat closeWeight(size, CV_64FC1);
	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width; c++) {
			double norm2 = pow(double(r - h), 2.0) + pow(double(c - w), 2.0);
			double sigma_g2 = 2.0 * pow(sigma_g, 2.0);
			closeWeight.at<double>(r, c) = exp(-norm2 / sigma_g2);
		}
	}
	return closeWeight;
}

Mat bilateralFilter(const Mat& input, Size winSize, float sigma_g, float sigma_d)
{
	const int rows = input.rows;
	const int cols = input.cols;
	const int height = winSize.height;
	const int width = winSize.width;
	CV_Assert(height > 0 && width > 0);
	CV_Assert(height % 2 == 1  && width % 2 == 1);
	if (height == 1 && width == 1)
		return input;
	const int h = (height - 1) / 2;
	const int w = (width - 1) / 2;

	// get space distance weights
	Mat closeWeight = getCloseWeight(sigma_g, winSize);

	Mat input_cp;
	input.clone().convertTo(input_cp, CV_32FC3);
	vector<Mat> img_chs;
	split(input_cp, img_chs);
	Mat b(input.size(), CV_32FC1), g(input.size(), CV_32FC1), r(input.size(), CV_32FC1), output;
	vector<Mat> channels{ b, g, r };
	
	for (int cn = 0; cn < 3; cn++) {
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				double pixel = img_chs[cn].at<float>(r, c);
				int right_top = (r - h) < 0 ? 0 : r - h;
				int right_bottom = (r + h) > rows - 1 ? rows - 1 : r + h;
				int c_left = (c - w) < 0 ? 0 : c - w;
				int c_right = (c + w) > cols - 1 ? cols - 1 : c + w;

				Mat region = img_chs[cn](Rect(Point(c_left, right_top),
											  Point(c_right + 1, right_bottom + 1))).clone();				
				Mat similarityWeight;
				pow(region - pixel, 2.0, similarityWeight);
				exp(-0.5 * similarityWeight / pow(sigma_d, 2), similarityWeight);
				similarityWeight /= pow(sigma_d, 2);

				Rect regionRect = Rect(Point(c_left - c + w, right_top - r + h),
									   Point(c_right - c + w + 1, right_bottom - r + h + 1));
				Mat closeWeightTemp;
				closeWeight(regionRect).clone().convertTo(closeWeightTemp, CV_32F);
				Mat weightTemp = (closeWeightTemp.mul(similarityWeight));
				weightTemp /= sum(weightTemp)[0];

				Mat result = weightTemp.mul(region);
				channels[cn].at<float>(r, c) = sum(result)[0];
			}
		}
	}
	merge(channels, output);
	output.convertTo(output, CV_8UC3);
	return output;
}

Mat sharpen(Mat input)
{
	CV_Assert(input.channels() == 3);
	input.convertTo(input, CV_64FC3);
	//Mat kernel = (Mat_<double>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
	Mat kernel = (Mat_<double>(3, 3) << 0, 0.2, 0, 0.2, -0.8, 0.2, 0, 0.2, 0);
	Mat output;
	Mat img_edge(input.size(), CV_64FC3);
	conv2D(input, kernel, img_edge, BORDER_REFLECT);
	subtract(input, img_edge, output);

	output.convertTo(output, CV_8UC1);

	return output;
}