#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <algorithm>
using namespace cv;
using namespace std;

// Functions prototypes
void Homography(const vector<Mat> &Images, vector<Mat> &transforms);
void FindOutputLimits(const vector<Mat> &Images, vector<Mat> &transforms, int &xMin, int &xMax, int &yMin, int &yMax);
void warpMasks(const vector<Mat> &Images, vector<Mat> &masks_warped, const vector<Mat> &transforms, const Mat &panorama);
void warpImages(const vector<Mat> &Images, const vector<Mat> &masks_warped, const vector<Mat> &transforms, Mat &panorama);
void BlendImages(const vector<Mat> &Images, Mat &pano_feather, Mat &pano_multiband, const vector<Mat> &masks_warped, const vector<Mat> &transforms);

int main()
{
	// Initialize OpenCV nonfree module
	initModule_nonfree();

	// Set the dir/name of each image 
	const int NUM_IMAGES = 6;
	//const string IMG_NAMES[] = { "FieldB09.JPG", "FieldB10.JPG", "FieldB11.JPG","FieldC09.JPG", "FieldC10.JPG", "FieldC11.JPG" };
	//const string IMG_NAMES[] = { "img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg", "img6.jpg" };
	const string IMG_NAMES[] = { "img1.jpg", "img6.jpg", "img5.jpg", "img2.jpg", "img3.jpg", "img4.jpg" };

	// Load the images
	vector<Mat> Images;
	for (int i = 0; i < NUM_IMAGES; i++)
	{
		Images.push_back(imread(IMG_NAMES[i]));
		//imshow("check", Images[i]);
		//waitKey(1);
	}

	// 1. Initialize all the transforms to the identity matrix
	vector<Mat> transforms;
	for (int i = 0; i < NUM_IMAGES; i++) {
		transforms.push_back(Mat::eye(3, 3, CV_64F));
	}

	// 2. Calculate the transformation matrices
	Homography(Images, transforms);

	// 3. Compute the min and max limits of the transformations
	int xMin, xMax, yMin, yMax;
	FindOutputLimits(Images, transforms, xMin, xMax, yMin, yMax);

	// 4. Compute the size of the panorama

	// 5. Initialize the panorama image	
	Mat panorama = Mat::zeros(yMax - yMin + 1, xMax - xMin + 1, CV_64F);

	cout << "X total = " << xMax - xMin + 1 << endl;
	cout << "Y total = " << yMax - yMin + 1 << endl;

	cout << "panorama height = " << panorama.size().height << endl;
	cout << "panorama width = " << panorama.size().width << endl;


	// 6. Initialize warped mask images

	vector<Mat> masks_warped(NUM_IMAGES);
	//masks_warped.reserve(NUM_IMAGES);
	

	// 7. Warp image masks
	warpMasks(Images, masks_warped, transforms, panorama);

	// 8. Warp the images
	warpImages(Images, masks_warped, transforms, panorama);

	// 9. Initialize the blended panorama images	

	Mat pano_feather = Mat::zeros(panorama.size(),CV_64F);
	Mat pano_multiband = Mat::zeros(panorama.size(), CV_64F);

	// 10. Blend
	BlendImages(Images, pano_feather, pano_multiband, masks_warped, transforms);

	return 0;
}

void Homography(const vector<Mat> &Images, vector<Mat> &transforms)
{
	/*Ptr<FeatureDetector> feature_detector = FeatureDetector::create("SIFT");
	Ptr<DescriptorExtractor> descriptor_extract = DescriptorExtractor::create("SIFT");

	Ptr<DescriptorMatcher> desc_matcher = DescriptorMatcher::create("BruteForce");

	vector<KeyPoint> store_kp_first; 
	Mat current_d_first;

	vector<KeyPoint> store_kp_sec;
	Mat current_d_sec;

	vector<DMatch> match_data;
	Mat image_out;*/

	std::cout << "length of vector: " << Images.size() << std::endl;
	int num_images = Images.size();

	/*vector<Point2d> first;
	vector<Point2d> second;*/

	for (int i = 1; i < num_images; i++) {

		Ptr<FeatureDetector> feature_detector = FeatureDetector::create("SIFT");
		Ptr<DescriptorExtractor> descriptor_extract = DescriptorExtractor::create("SIFT");

		Ptr<DescriptorMatcher> desc_matcher = DescriptorMatcher::create("BruteForce");

		vector<KeyPoint> store_kp_first;
		Mat current_d_first;

		vector<KeyPoint> store_kp_sec;
		Mat current_d_sec;

		vector<DMatch> match_data;
		Mat image_out;

		vector<Point2d> first;
		vector<Point2d> second;

		feature_detector->detect(Images[i], store_kp_first);
		descriptor_extract->compute(Images[i], store_kp_first, current_d_first);

		feature_detector->detect(Images[i-1], store_kp_sec);
		descriptor_extract->compute(Images[i-1], store_kp_sec, current_d_sec);
	
		desc_matcher->match(current_d_first, current_d_sec, match_data);
		drawMatches(Images[i], store_kp_first, Images[i - 1], store_kp_sec, match_data, image_out, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		std:ostringstream os;
		os << "Dmatched_image_" << i << ".jpg";
		imwrite(os.str(), image_out);

		double min_dist = DBL_MAX;

		for (int j = 0; j < match_data.size(); j++) {
			double distance = match_data[j].distance;
			if (distance < min_dist)
				min_dist = distance;
		}
		min_dist *= 3.2;
		match_data.erase(
			std::remove_if(
				match_data.begin(), match_data.end(),
				[min_dist](DMatch remove) {return (remove.distance > min_dist);}
				),
			match_data.end()
			);


		for (int j = 0; j < match_data.size(); j++) {
			first.push_back(store_kp_first[match_data[j].queryIdx].pt);
			second.push_back(store_kp_sec[match_data[j].trainIdx].pt);
		}

		transforms[i] = findHomography(first, second,RANSAC);

		//transforms[i] = findHomography(first, transforms[0], RANSAC);
	}

	Mat transform_extra = transforms[0] * transforms[5];
	std::cout << transform_extra << endl;
	for (int i = 1; i < num_images; i++) {
		transforms[i] = transforms[i-1] * transforms[i];
		//std:ostringstream os;
		
		//imwrite(os.str(), transforms[i]);*/
	}
	std::cout << transforms[5] << endl;
	

}



void FindOutputLimits(const vector<Mat> &Images, vector<Mat> &transforms, int &xMin, int &xMax, int &yMin, int &yMax)
{
	int num_images = Images.size();
	std::cout << "length of vector: " << Images.size() << std::endl;
	xMin = INT_MAX;
	yMin = INT_MAX;
	xMax = INT_MIN;
	yMax = INT_MIN;

	printf("[xMin xMax yMin yMax] = [%d, %d, %d, %d]\n", xMin, xMax, yMin, yMax);

	Mat proj = Mat::ones(3, 1, CV_64F);
	Mat corn = Mat::ones(3, 1, CV_64F);
	Mat trans = Mat::eye(3, 3, CV_64F);
	vector<Mat> corners(4);
	for (int i = 0; i < corners.size(); i++) {
		corners[i] = Mat::ones(3, 1, CV_64F);
	}
	for (int i = 0; i < num_images; i++) {

		corners[0].at<double>(0, 0) = 0;
		corners[0].at<double>(1, 0) = 0;

		corners[1].at<double>(0, 0) = 0;
		corners[1].at<double>(1, 0) = Images[i].size().height - 1;

		corners[2].at<double>(0, 0) = Images[i].size().width - 1;;
		corners[2].at<double>(1, 0) = 0;

		corners[3].at<double>(0, 0) = Images[i].size().width - 1;;
		corners[3].at<double>(1, 0) = Images[i].size().height - 1;

		for (int j = 0; j < corners.size(); j++) {
			corn.at<double>(0, 0) = corners[j].at<double>(0, 0);
			corn.at<double>(1, 0) = corners[j].at<double>(1, 0);
			proj = transforms[i] * corn;
			proj /= proj.at<double>(2,0);
			if (proj.at<double>(0, 0) > xMax) {
				xMax = proj.at<double>(0, 0);
			}
			if(proj.at<double>(0,0) < xMin){
				xMin = proj.at<double>(0, 0);
			}
			if (proj.at<double>(1, 0) > yMax) {
				yMax = proj.at<double>(1, 0);
			}
			if (proj.at<double>(1, 0) < yMin) {
				yMin = proj.at<double>(1, 0);
			}
			

		}
	}
	trans.at<double>(0, 2) = -xMin;
	trans.at<double>(1, 2) = -yMin;

	for (int i = 0; i < num_images; i++) {
		transforms[i] = trans * transforms[i];
	}
	cout << trans << endl;
	printf("[xMin xMax yMin yMax] = [%d, %d, %d, %d]\n", xMin, xMax, yMin, yMax);
}

void warpMasks(const vector<Mat> &Images, vector<Mat> &masks_warped, const vector<Mat> &transforms, const Mat &panorama)
{
	int num_images = Images.size();
	std::cout << "length of vector: " << Images.size() << std::endl;
	vector<Mat> masks(num_images);

	for (int i = 0; i < num_images; i++) {
		masks[i] = Mat::ones(Images[i].size().height, Images[i].size().width, CV_8U);
		masks[i].setTo(cv::Scalar::all(255));
		
		warpPerspective(masks[i], masks_warped[i], transforms[i], panorama.size());

		std::ostringstream os;
		os << "warped_masks_" << i << ".jpg";
		imwrite(os.str(), masks_warped[i]);
	}
	
}

void warpImages(const vector<Mat> &Images, const vector<Mat> &masks_warped, const vector<Mat> &transforms, Mat &panorama)
{
	int num_images = Images.size();
	vector<Mat> Images_out(num_images);

	for (int i = 0; i < num_images; i++) {
		warpPerspective(Images[i],Images_out[i],transforms[i],panorama.size(), INTER_LINEAR, BORDER_CONSTANT,1);
		Images_out[i].copyTo(panorama,masks_warped[i]);
		std::ostringstream os;
		os << "warped_image_" << i << ".jpg";
		imwrite(os.str(), Images_out[i]);
	}
	imwrite("panorma_warped.jpg", panorama);
	//imshow("image", panorama);
	waitKey(1);
	//system("pause");

}

void BlendImages(const vector<Mat> &Images, Mat &pano_feather, Mat &pano_multiband, const vector<Mat> &masks_warped, const vector<Mat> &transforms)
{
	detail::FeatherBlender f_blend;
	detail::MultiBandBlender mb_blend;

	f_blend.prepare(Rect(0, 0, pano_feather.cols, pano_feather.rows));
	mb_blend.prepare(Rect(0, 0, pano_feather.cols, pano_feather.rows));

	for (int i = 0; i < Images.size(); i++) {
		Mat image_warped;
		warpPerspective(Images[i], image_warped, transforms[i], pano_feather.size(), INTER_LINEAR, BORDER_REPLICATE, 1);
		image_warped.convertTo(image_warped, CV_16S);
		f_blend.feed(image_warped, masks_warped[i], Point(0, 0));
		mb_blend.feed(image_warped, masks_warped[i], Point(0, 0));
	}
	Mat f_empty = Mat::zeros(pano_feather.size(), CV_8U);
	Mat mb_empty = Mat::zeros(pano_multiband.size(), CV_8U);

	f_blend.blend(pano_feather, f_empty);
	mb_blend.blend(pano_multiband, mb_empty);
	pano_feather.convertTo(pano_feather, CV_8U);
	pano_multiband.convertTo(pano_multiband, CV_8U);

	imwrite("pano_feather.jpg", pano_feather);	
	imwrite("pano_multiband.jpg", pano_multiband);
}