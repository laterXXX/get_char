#include  <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <features2d\features2d.hpp>
#include <vector> 
#include <fstream>
#include <ctime>
using namespace cv;
using namespace std;

void main123()
{

	ifstream file;
	file.open("H:/rect_license/img_names.txt");
	assert(file.is_open());
	string img_name;
	while (getline(file, img_name))
	{
		//������С��Ӿ��εĻ���
		string imgstr(img_name);
		Mat srcImg = imread(imgstr);
		int w = srcImg.cols;
		int h = srcImg.rows;
		resize(srcImg, srcImg, Size(1200, 400));
		Mat dstImg = srcImg.clone();
		cvtColor(srcImg, srcImg, CV_BGR2GRAY);
		threshold(srcImg, srcImg, 100, 255, CV_THRESH_BINARY); //��ֵ��
		srcImg = 255 - srcImg;
		imshow("threshold", srcImg);
		waitKey(0);

		vector<vector<Point>> contours;
		vector<Vec4i> hierarcy;
		findContours(srcImg, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		//vector<Rect> boundRect(contours.size());  //������Ӿ��μ���
		vector<RotatedRect> box(contours.size()); //������С��Ӿ��μ���
		Point2f rect[4];
		for (int i = 0; i < contours.size(); i++)
		{
			box[i] = minAreaRect(Mat(contours[i]));  //����ÿ��������С��Ӿ���
													 //boundRect[i] = boundingRect(Mat(contours[i]));
			circle(dstImg, Point(box[i].center.x, box[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  //������С��Ӿ��ε����ĵ�
			box[i].points(rect);  //����С��Ӿ����ĸ��˵㸴�Ƹ�rect����
								  //rectangle(dstImg, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(0, 255, 0), 2, 8);
			for (int j = 0; j < 4; j++)
			{
				line(dstImg, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //������С��Ӿ���ÿ����
			}
		}
		imshow("dst", dstImg);
		waitKey(0);

		//�Ƴ���������̵�����  
		int cmin = 100; //��С��������  
		int cmax = 1000;    //�������  
		vector<vector<Point>>::const_iterator itc = contours.begin();
		while (itc != contours.end())
		{
			if (itc->size() < cmin || itc->size() > cmax)
				itc = contours.erase(itc);
			else
				++itc;
		}

		//�ڰ�ɫͼ���ϻ��ƺ�ɫ����  
		Mat result_erase(srcImg.size(), CV_8U, Scalar(255));
		drawContours(result_erase, contours,
			-1, //������������  
			Scalar(0),  //��ɫΪ��ɫ  
			2); //�����ߵĻ��ƿ��Ϊ2  

		namedWindow("contours_erase");
		imshow("contours_erase", result_erase);
		waitKey(0);
	}
}

void drawLine(cv::Mat &image, double theta, double rho, cv::Scalar color)
{
	if (theta < 3.1415 / 4. || theta > 3.*3.1415 / 4.)// ~vertical line
	{
		cv::Point pt1(rho / cos(theta), 0);
		cv::Point pt2((rho - image.rows * sin(theta)) / cos(theta), image.rows);
		cv::line(image, pt1, pt2, cv::Scalar(255), 1);
	}
	else
	{
		cv::Point pt1(0, rho / sin(theta));
		cv::Point pt2(image.cols, (rho - image.cols * cos(theta)) / sin(theta));
		cv::line(image, pt1, pt2, color, 1);
	}
}

//����[0,1]֮����Ͼ��ȷֲ�����
double uniformRandom(void)
{
	return (double)rand() / (double)RAND_MAX;
}

//����[0,1]֮����ϸ�˹�ֲ�����
double gaussianRandom(void)
{
	/* This Gaussian routine is stolen from Numerical Recipes and is their
	copyright. */
	static int next_gaussian = 0;
	static double saved_gaussian_value;

	double fac, rsq, v1, v2;

	if (next_gaussian == 0) {
		do {
			v1 = 2 * uniformRandom() - 1;
			v2 = 2 * uniformRandom() - 1;
			rsq = v1 * v1 + v2 * v2;
		} while (rsq >= 1.0 || rsq == 0.0);
		fac = sqrt(-2 * log(rsq) / rsq);
		saved_gaussian_value = v1 * fac;
		next_gaussian = 1;
		return v2 * fac;
	}
	else {
		next_gaussian = 0;
		return saved_gaussian_value;
	}
}

//���ݵ㼯���ֱ��ax+by+c=0��resΪ�в�
void calcLinePara(vector<Point2d> pts, double &a, double &b, double &c, double &res)
{
	res = 0;
	Vec4f line;
	vector<Point2f> ptsF;
	for (unsigned int i = 0; i < pts.size(); i++)
		ptsF.push_back(pts[i]);

	fitLine(ptsF, line, CV_DIST_L2, 0, 1e-2, 1e-2);
	a = line[1];
	b = -line[0];
	c = line[0] * line[3] - line[1] * line[2];

	for (unsigned int i = 0; i < pts.size(); i++)
	{
		double resid_ = fabs(pts[i].x * a + pts[i].y * b + c);
		res += resid_;
	}
	res /= pts.size();
}

//�õ�ֱ���������������ֱ�߲����㼯�����ѡ2����
bool getSample(vector<int> set, vector<int> &sset)
{
	int i[2];
	if (set.size() > 2)
	{
		do
		{
			for (int n = 0; n < 2; n++)
				i[n] = int(uniformRandom() * (set.size() - 1));
		} while (!(i[1] != i[0]));
		for (int n = 0; n < 2; n++)
		{
			sset.push_back(i[n]);
		}
	}
	else
	{
		return false;
	}
	return true;
}

//ֱ���������������λ�ò���̫��
bool verifyComposition(const vector<Point2d> pts)
{
	cv::Point2d pt1 = pts[0];
	cv::Point2d pt2 = pts[1];
	if (abs(pt1.x - pt2.x) < 5 && abs(pt1.y - pt2.y) < 5)
		return false;

	return true;
}

//RANSACֱ�����
void fitLineRANSAC(vector<Point2d> ptSet, double &a, double &b, double &c, vector<bool> &inlierFlag)
{
	double residual_error = 2.99; //�ڵ���ֵ

	bool stop_loop = false;
	int maximum = 0;  //����ڵ���

					  //�����ڵ��ʶ����в�
	inlierFlag = vector<bool>(ptSet.size(), false);
	vector<double> resids_(ptSet.size(), 3);
	int sample_count = 0;
	int N = 500;

	double res = 0;

	// RANSAC
	srand((unsigned int)time(NULL)); //�������������
	vector<int> ptsID;
	for (unsigned int i = 0; i < ptSet.size(); i++)
		ptsID.push_back(i);
	while (N > sample_count && !stop_loop)
	{
		vector<bool> inlierstemp;
		vector<double> residualstemp;
		vector<int> ptss;
		int inlier_count = 0;
		if (!getSample(ptsID, ptss))
		{
			stop_loop = true;
			continue;
		}

		vector<Point2d> pt_sam;
		pt_sam.push_back(ptSet[ptss[0]]);
		pt_sam.push_back(ptSet[ptss[1]]);

		if (!verifyComposition(pt_sam))
		{
			++sample_count;
			continue;
		}

		// ����ֱ�߷���
		calcLinePara(pt_sam, a, b, c, res);
		//�ڵ����
		for (unsigned int i = 0; i < ptSet.size(); i++)
		{
			Point2d pt = ptSet[i];
			double resid_ = fabs(pt.x * a + pt.y * b + c);
			residualstemp.push_back(resid_);
			inlierstemp.push_back(false);
			if (resid_ < residual_error)
			{
				++inlier_count;
				inlierstemp[i] = true;
			}
		}
		// �ҵ�������ֱ��
		if (inlier_count >= maximum)
		{
			maximum = inlier_count;
			resids_ = residualstemp;
			inlierFlag = inlierstemp;
		}
		// ����RANSAC�����������Լ��ڵ����
		if (inlier_count == 0)
		{
			N = 500;
		}
		else
		{
			double epsilon = 1.0 - double(inlier_count) / (double)ptSet.size(); //Ұֵ�����
			double p = 0.99; //���������д���1���������ĸ���
			double s = 2.0;
			N = int(log(1.0 - p) / log(1.0 - pow((1.0 - epsilon), s)));
		}
		++sample_count;
	}

	//���������ڵ��������ֱ��
	vector<Point2d> pset;
	for (unsigned int i = 0; i < ptSet.size(); i++)
	{
		if (inlierFlag[i])
			pset.push_back(ptSet[i]);
	}

	calcLinePara(pset, a, b, c, res);
}


bool comp1(const Point &a, const Point &b)
{
	return a.x < b.x;
}

bool comp2(const RotatedRect &a, const RotatedRect &b)
{
	return a.center.x < b.center.x;
}

bool comp3(const Rect &a, const Rect &b)
{
	return a.x < b.x;
}

float getDist_P2L(Point pointP, Point pointA, Point pointB)
{
	//��ֱ�߷���
	int A = 0, B = 0, C = 0;
	A = pointA.y - pointB.y;
	B = pointB.x - pointA.x;
	C = pointA.x*pointB.y - pointA.y*pointB.x;
	//����㵽ֱ�߾��빫ʽ
	float distance = 0;
	distance = ((float)abs(A*pointP.x + B * pointP.y + C)) / ((float)sqrtf(A*A + B * B));
	return distance;
}


int main() {
	ifstream file;
	file.open("H:/rect_license/img_names.txt");
	assert(file.is_open());
	string img_name;
	while (getline(file, img_name))
	{
		Mat image = imread(img_name);
		resize(image, image, Size(1200, 300));
		
		Mat grayImage;
		cvtColor(image, grayImage, CV_BGR2GRAY);

		medianBlur(grayImage, grayImage, 9);
		imshow("grayImage", grayImage);

		//ת��Ϊ��ֵͼ    
		Mat binaryImage;
		threshold(grayImage, binaryImage, 120, 255, CV_THRESH_BINARY);

		//Canny(grayImage, binaryImage, 100, 200);
		//imshow("binaryImage", binaryImage);

		//Mat element = getStructuringElement(MORPH_RECT, Size(30, 30), Point(-1, -1)); //����ṹԪ��
		//dilate(binaryImage, binaryImage, element); //����
		//imshow("dilate", binaryImage);
		//erode(binaryImage, binaryImage, element);
		//imshow("erode", binaryImage);


		//��ֵͼ ������������ط�ת����Ϊһ��������255��ɫ��ʾǰ�������壩����0��ɫ��ʾ����    
		Mat reverseBinaryImage;
		bitwise_not(binaryImage, reverseBinaryImage);
		imshow("bitwise_not", reverseBinaryImage);

		vector <vector<Point>>contours;
		findContours(reverseBinaryImage,
			contours,   //����������  
			CV_RETR_EXTERNAL,   //��ȡ������  
			CV_CHAIN_APPROX_NONE);  //��ȡÿ��������ÿ������  
									//�ڰ�ɫͼ���ϻ��ƺ�ɫ���� 

		Mat result(reverseBinaryImage.size(), CV_8U, Scalar(255));
		drawContours(result, contours,
			-1, //������������  
			Scalar(0),  //��ɫΪ��ɫ  
			2); //�����ߵĻ��ƿ��Ϊ2  

		namedWindow("contours");
		imshow("contours", result);

		//�Ƴ���������̵�����  
		int cmin = 100; //��С��������  
		int cmax = 1500;    //�������  
		vector<vector<Point>>::const_iterator itc = contours.begin();
		while (itc != contours.end())
		{
			if (itc->size() < cmin || itc->size() > cmax)
				itc = contours.erase(itc);
			else
				++itc;
		}

		double max_area = 0;
		vector<double> areas;
		int index = 0;
		Mat ROI;
		vector<Mat> rois;
		/*for (int i = 0; i < contours.size(); i++)
		{
			cout << contourArea(contours[i]) << endl;
		}
		cout <<"----------------------------" << endl;*/
		Mat minAreaRect_erase(binaryImage.size(), CV_8U, Scalar(255));
		vector<RotatedRect> box(contours.size()); //������С��Ӿ��μ���
		vector<Rect> boundRect(contours.size());  //������Ӿ��μ���
		Point2f rect[4];
		vector<Point> p;
		Mat dst = binaryImage.clone();
		for (int i = 0; i < contours.size(); i++)
		{
			boundRect[i] = boundingRect(Mat(contours[i]));
			rectangle(image, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(0, 255, 0), 2, 8);

			box[i] = minAreaRect(Mat(contours[i]));  //����ÿ��������С��Ӿ���

			p.push_back(box[i].center);	

			circle(binaryImage, Point(box[i].center.x, box[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  //������С��Ӿ��ε����ĵ�
			
			box[i].points(rect);  //����С��Ӿ����ĸ��˵㸴�Ƹ�rect����
			
			for (int j = 0; j < 4; j++)
			{
				line(binaryImage, rect[j], rect[(j + 1) % 4], Scalar(0, 0, 255), 2, 8);  //������С��Ӿ���ÿ����
			}

			//float angle;
			// cout << "angle=" << box[i].angle << endl;
			// angle = box[i].angle;
			////���÷���任������ת        ��һ�ַ�����͸�ӱ任
			//if (0< abs(angle) && abs(angle) <= 45)
			//	angle = angle;//������˳ʱ����ת
			//else if (45< abs(angle) && abs(angle)<90)
			//	angle = 90 - abs(angle);//��������ʱ����ת
			//Point2f center = box[i].center;  //������ת��������
			//double angle0 = angle;
			//double scale = 1;
			//Mat roateM = getRotationMatrix2D(center, angle0, scale);  //�����ת����,˳ʱ��Ϊ������ʱ��Ϊ��
			//if (i == 0) {
			//	warpAffine(dst, dst, roateM, dst.size()); //����任
			//}
			//		


			//int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
			//x0 = boundRect[i].x;
			//y0 = boundRect[i].y;
			//w0 = boundRect[i].width;
			//h0 = boundRect[i].height;
			//ROI = dst(Rect(x0, y0, w0, h0));
			//imshow("pg", ROI);
			//waitKey(0);
		}

		

		double cos_theta, sin_theta;
		double sin;
		double x0, y0;

		double phi;
		double rho;
		double PI = 3.1415;
		if(p.size() > 0){
			vector<Vec2f> lines;
			
			cv::Vec4f lin;
			cv::fitLine(p,
				lin,
				CV_DIST_HUBER,
				0,
				0.01,
				0.01);
			cos_theta = lin[0];
			sin_theta = lin[1];
			 x0 = lin[2], y0 = lin[3];

			phi = atan2(sin_theta, cos_theta) + 3.1415 / 2.0;
			rho = y0 * cos_theta - x0 * sin_theta;

			//std::cout << "phi = " << phi / PI * 180 << std::endl;
			//std::cout << "rho = " << rho << std::endl;

			drawLine(binaryImage, phi, rho, cv::Scalar(0));

		}

		double k = sin_theta / cos_theta;
		double b = y0 - k * x0;

		double x = 0;
		double y = k * x + b;

		double A, B, C, dis;


		A = y - y0;
		B = x0 - x;
		C = x * y0 - x0 * y;

		if (p.size() > 0) {
			for (int i = 0; i < box.size(); i++) {
				dis = abs(A * p[i].x+ B * p[i].y + C) / sqrt(A * A + B * B);
				cout << dis << endl;
			}
				
		}


		sort(p.begin(), p.end(), comp1);
		sort(box.begin(), box.end(), comp2);
		sort(boundRect.begin(), boundRect.end(), comp3);


		for (int i = 0; i < box.size(); i++)
		{
			Mat result1(800, 800, CV_8U, Scalar(255));
			int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
			x0 = boundRect[i].x;
			y0 = boundRect[i].y;
			w0 = boundRect[i].width;
			h0 = boundRect[i].height;
			ROI = dst(Rect(x0, y0, w0, h0));

			Mat rr = result1(Rect(result1.cols / 2 - w0 / 2, result1.rows / 2 - h0 / 2, ROI.cols, ROI.rows));
			ROI.copyTo(rr);

			imshow("ROI", ROI);
			float angle;
			//cout << "angle=" << box[i].angle << endl;
			angle = box[i].angle;
			//���÷���任������ת        ��һ�ַ�����͸�ӱ任
			if (0< abs(angle) && abs(angle) <= 45)
				angle = angle;//������˳ʱ����ת
			else if (45< abs(angle) && abs(angle)<90)
				angle = 90 - abs(angle);//��������ʱ����ת
			Point2f center(ROI.cols, ROI.rows);  //������ת��������
			double angle0 = angle;
			double scale = 1;
			Mat roateM = getRotationMatrix2D(center, angle0, scale);  //�����ת����,˳ʱ��Ϊ������ʱ��Ϊ��

			warpAffine(result1, result1, roateM, result1.size()); //����任

																  //Mat roi = result1(Rect(0, 32, src.cols, src.rows));

			imshow("pg", result1);
			imshow("ROI1", ROI);
			//waitKey(0);
		}





#pragma region ransac
		//double A, B, C;
		//vector<bool> inliers;
		//fitLineRANSAC(p, A, B, C, inliers);

		//B = B / A;
		//C = C / A;
		//A = A / A;

		////����ֱ��
		//Point2d ptStart, ptEnd;
		//ptStart.x = 0;
		//ptStart.y = -(A*ptStart.x + C) / B;
		//ptEnd.x = -(B*ptEnd.y + C) / A;
		//ptEnd.y = 0;
		//line(binaryImage, ptStart, ptEnd, Scalar(0, 255, 255));
		//cout << "A:" << A << " " << "B:" << B << " " << "C:" << C << " " << endl;
		//imshow("line fitting", binaryImage);
#pragma endregion ransac
		imshow("src", image);
		imshow("minAreaRect", binaryImage);


		////�ڰ�ɫͼ���ϻ��ƺ�ɫ����  
		//Mat result_erase(binaryImage.size(), CV_8U, Scalar(255));
		//drawContours(result_erase, contours,
		//	-1, //������������  
		//	Scalar(0),  //��ɫΪ��ɫ  
		//	2); //�����ߵĻ��ƿ��Ϊ2  

		//		//namedWindow("contours_erase");  
		//		//imshow("contours_erase", result_erase);  

		//		//���԰�Χ��  
		//Rect r0 = boundingRect(Mat(contours[0]));
		//rectangle(result_erase, r0, Scalar(128), 2);
		//Rect r1 = boundingRect(Mat(contours[1]));
		//rectangle(result_erase, r1, Scalar(128), 2);

		////������С��ΧԲ  
		//float radius;
		//Point2f center;
		////minEnclosingCircle(Mat(contours[2]), center, radius);
		////circle(result_erase, Point(center), static_cast<int>(radius), Scalar(128), 2);

		////���Զ���ν���  
		//vector <Point> poly;
		//approxPolyDP(Mat(contours[3]),
		//	poly,
		//	5,  //���Ƶľ�ȷ��  
		//	true);  //���Ǹ��պ���״  

		//			//����ÿ��Ƭ�ν��л���  
		//vector<Point>::const_iterator itp = poly.begin();
		//while (itp != (poly.end() - 1))
		//{
		//	line(result_erase, *itp, *(itp + 1), Scalar(128), 2);
		//	++itp;
		//}

		////��β��ֱ������  
		//line(result_erase, *(poly.begin()), *(poly.end() - 1), Scalar(128), 2);

		////͹������һ�ֶ���ν���,����͹��  
		//vector <Point> hull;
		//convexHull(Mat(contours[4]), hull);

		//vector<Point>::const_iterator ith = hull.begin();
		//while (ith != (hull.end() - 1))
		//{
		//	line(result_erase, *ith, *(ith + 1), Scalar(128), 2);
		//	++ith;
		//}
		//line(result_erase, *(hull.begin()), *(hull.end() - 1), Scalar(128), 2);

		////��һ��ǿ�������������  
		////��������  
		////������������  
		//itc = contours.begin();
		//while (itc != contours.end())
		//{
		//	//�������е�����  
		//	Moments mom = moments(Mat(*itc++));
		//	//��������  
		//	circle(result_erase,
		//		Point(mom.m10 / mom.m00, mom.m01 / mom.m00),    //��������ת��Ϊ����  
		//		2,
		//		Scalar(0),
		//		2); //���ƺڵ�  
		//}

		//namedWindow("contours_erase");
		//imshow("contours_erase", result_erase);
		waitKey(0);
	}
}
