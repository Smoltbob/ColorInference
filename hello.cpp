#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/* TODO
 * Echantilloner Environ 15 couleurs pour entrainer modele
 * Tester une prédiction avec input RGB
 */

int main(int argc, char** argv)
{
	// Parameters for kmeans
	const int MAX_ITERATIONS = 10;
	const auto K = 5;

	auto image = cv::imread("lamb.png", IMREAD_COLOR);
	if(!image.data){
		cout << "Could not open image" << endl;
	}

	// Make dummy bin mask
	cv::Mat dest;
	cv::Mat imageBW;
	cv::cvtColor(image, imageBW, COLOR_BGR2GRAY);
	cv::threshold(image, dest,  220, 255, THRESH_BINARY);
	//cv::imshow("result", dest);

	cout << image.depth() << endl;
	cout << dest.depth() << endl;

	int channels = image.channels();
	int nRows = image.rows;
	int nCols = image.cols;// * channels;
	if (image.isContinuous()){
		nCols *= nRows;
		nRows = 1;
	}

	// Access mask pixels. Assumes three uchar channels.
	int i,j;
	cv::Vec3b* p; 
	cv::Vec3b* b;
	cv::Vec3b white{255,255,255};
	for (i = 0; i < nRows; ++i)
	{
		p = image.ptr<cv::Vec3b>(i);
		b = dest.ptr<cv::Vec3b>(i);
		for (j = 0; j < nCols; ++j)
		{
			if (b[j] == white)
			{
				p[j][0] = 134;
				p[j][1] = 14;
				p[j][2] = 84;
			}
		}
	}
	cv::imshow("result", image);

	// We need to flatten the image to 1D for kmeans
	const auto singleLineSize = image.rows * image.cols;
	cv::Mat data = image.reshape(1, singleLineSize);
	data.convertTo(data, CV_32F);
	vector<int> labels; // label for each pixel
	cv::Mat1f colors; // output cluster centers

	cv::kmeans(data, K, labels, cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.),MAX_ITERATIONS,cv::KMEANS_PP_CENTERS,colors);

	/* Find the largest cluster*/
	int max = 0, indx= 0, id = 0;
	vector<int> clusters(K,0);

	for (size_t i = 0; i < labels.size(); i++)
	{
		id = labels[i];
		clusters[id]++;

		if (clusters[id] > max)
		{
			max = clusters[id];
			indx = id;
		}
	}

	cout << "Biggest cluster: " << colors.row(indx) << endl;

	// Go back to 2D
	for (auto i = 0 ; i < singleLineSize ; i++ ){
		data.at<float>(i, 0) = colors(labels[i], 0);
		data.at<float>(i, 1) = colors(labels[i], 1);
		data.at<float>(i, 2) = colors(labels[i], 2);
	}

	// Print centers
	cout << colors << endl;

	cv::Mat outputImage = data.reshape(3, image.rows);
	outputImage.convertTo(outputImage, CV_8U);
	cv::imshow("result", outputImage);

	waitKey();

}
