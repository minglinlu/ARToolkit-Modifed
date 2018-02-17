#include "commonCvFunctions.h"
#include "orException.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

namespace cvar{
    // Convert Point2f structure vector to Mat type in the homogeneous coordinates
    Mat transPointVecToMatHom(vector<Point2f>& pt_vec)
    {
        int size = pt_vec.size();
        Mat retMat(3, size, CV_64FC1);
        
        for(int i=0; i<size; i++){
            retMat.at<double>(0,i) = (double)(pt_vec[i].x);
            retMat.at<double>(1,i) = (double)(pt_vec[i].y);
            retMat.at<double>(2,i) = 1.0;
        }
        
        return retMat;
    }
    
    
    // Order of output is: Top Left, Bottom Left, Bottom Right, Top Right
    vector<Point2f> calcAffineTransformRect(cv::Size& img_size, cv::Mat& transMat)
    {
        float width = (float)(img_size.width) - 1;
        float height = (float)(img_size.height) - 1;
        
        Mat src_mat = (Mat_<double>(3,4) << 0,0,width,width,0,height,height,0,1,1,1,1);
        //	Mat dest_mat(3,4,CV_64FC1);
        
        Mat dest_mat = transMat * src_mat;
        
        vector<Point2f>	ret_vec;
        
        Point2f pt;
        for(int i=0; i<4; i++){
            pt.x = (float)(dest_mat.at<double>(0,i) / dest_mat.at<double>(2,i));
            pt.y = (float)(dest_mat.at<double>(1,i) / dest_mat.at<double>(2,i));
            ret_vec.push_back(pt);
        }
        
        return ret_vec;
    }
    
    
    vector<Point2f> calcAffineTransformPoints(vector<Point2f>& pts_vec, cv::Mat& transMat)
    {
        vector<Point2f>	ret_vec;
        if(pts_vec.empty())	return ret_vec;
        
        Mat src_mat = transPointVecToMatHom(pts_vec);
        
        Mat dest_mat = transMat * src_mat;
        
        Point2f pt;
        int cols = dest_mat.cols;
        for(int i=0; i<cols; i++){
            pt.x = (float)(dest_mat.at<double>(0,i) / dest_mat.at<double>(2,i));
            pt.y = (float)(dest_mat.at<double>(1,i) / dest_mat.at<double>(2,i));
            ret_vec.push_back(pt);
        }
        
        return ret_vec;
    }
    
    
    // Check the validity of transformed rectangle shape
    // the sign of outer products of each edge vector must be same
    bool checkRectShape(vector<Point2f>& rect_pts)
    {
        CV_Assert(rect_pts.size()==4);
        
        bool result_f = true;
        float vec[4][2];
        int i;
        
        vec[0][0] = rect_pts[1].x - rect_pts[0].x;
        vec[0][1] = rect_pts[1].y - rect_pts[0].y;
        vec[1][0] = rect_pts[2].x - rect_pts[1].x;
        vec[1][1] = rect_pts[2].y - rect_pts[1].y;
        vec[2][0] = rect_pts[3].x - rect_pts[2].x;
        vec[2][1] = rect_pts[3].y - rect_pts[2].y;
        vec[3][0] = rect_pts[0].x - rect_pts[3].x;
        vec[3][1] = rect_pts[0].y - rect_pts[3].y;
        
        int s;
        float val = vec[3][0] * vec[0][1] - vec[3][1] * vec[0][0];
        if(val > 0)
            s = 1;
        else
            s = -1;
        
        for(i=0; i<3; i++){
            val = vec[i][0] * vec[i+1][1] - vec[i][1] * vec[i+1][0];
            if( val * s <= 0){
                result_f = false;
                break;
            }
        }
        
        return result_f;
    }
    
    // corner_pts[0]:Top Left, corner_pts[1]:Bottom Left, corner_pts[2]:Bottom Right, corner_pts[3]:Top Right
    int checkInsideArea(vector<Point2f>& points, vector<Point2f>& corner_pts, vector<unsigned char>& status)
    {
        CV_Assert(corner_pts.size() == 4);
        CV_Assert(points.size() == status.size());
        
        // ax+by+c=0
        float a[4];
        float b[4];
        float c[4];
        
        a[0] = corner_pts[3].y - corner_pts[0].y;
        a[1] = corner_pts[2].y - corner_pts[1].y;
        a[2] = corner_pts[1].y - corner_pts[0].y;
        a[3] = corner_pts[2].y - corner_pts[3].y;
        
        b[0] = corner_pts[0].x - corner_pts[3].x;
        b[1] = corner_pts[1].x - corner_pts[2].x;
        b[2] = corner_pts[0].x - corner_pts[1].x;
        b[3] = corner_pts[3].x - corner_pts[2].x;
        
        c[0] = corner_pts[0].y * corner_pts[3].x - corner_pts[3].y * corner_pts[0].x;
        c[1] = corner_pts[1].y * corner_pts[2].x - corner_pts[2].y * corner_pts[1].x;
        c[2] = corner_pts[0].y * corner_pts[1].x - corner_pts[1].y * corner_pts[0].x;
        c[3] = corner_pts[3].y * corner_pts[2].x - corner_pts[2].y * corner_pts[3].x;
        
        float max_x, min_x, max_y, min_y;
        max_x = corner_pts[0].x;
        min_x = corner_pts[0].x;
        max_y = corner_pts[0].y;
        min_y = corner_pts[0].y;
        
        int i;
        for(i=1;i<4;i++){
            if(corner_pts[i].x > max_x)
                max_x = corner_pts[i].x;
            if(corner_pts[i].x < min_x)
                min_x = corner_pts[i].x;
            if(corner_pts[i].y > max_y)
                max_y = corner_pts[i].y;
            if(corner_pts[i].y < min_y)
                min_y = corner_pts[i].y;
        }
        
        float val[4];
        int size = points.size();
        int count = 0;
        for(int j=0;j<size;j++){
            if(status[j] > 0){
                for(i=0; i<4; i++){
                    val[i] = a[i] * points[j].x + b[i] * points[j].y + c[i];
                }
                if(val[0]*val[1] <= 0 && val[2]*val[3] <= 0){
                    count++;
                }else{
                    status[j] = 0;
                }
            }
        }
        
        return count;
    }
    
    
    // judgment pts is whether all there within the image area
    bool checkPtInsideImage(Size img_size, vector<Point2f>& pts)
    {
        vector<Point2f>::iterator itr = pts.begin();
        while(itr != pts.end()){
            if(itr->x < 0 || itr->x >= img_size.width || itr->y < 0 || itr->y >= img_size.height){
                return false;
            }
            else{
                itr++;
            }
        }
        return true;
    }
    
    
    
 
    
    
    template<typename _Tp> void decomposeHomographyType(Mat& H_mat, Mat& camera_matrix, Mat& rotation, Mat& translation, Point2f marker_center)
    {
        try{
            CV_Assert(H_mat.type() == camera_matrix.type());
            CV_Assert(H_mat.cols == 3 && H_mat.rows == 3 && camera_matrix.cols == 3 && camera_matrix.rows == 3);
            
            int i,j;
            Mat Amarker = camera_matrix.clone();	// Matrix to convert the world coordinates to the scale of the image coordinates
            Amarker.at<_Tp>(0,2) = marker_center.x;
            Amarker.at<_Tp>(1,2) = marker_center.y;
            
            Mat invA = camera_matrix.inv();
            Mat tmpM = invA * H_mat * Amarker;
            //	Mat tmpM = invA * H_mat;
            
            Mat Rcol[2];
            Mat rMat[3];
            
            double lambda[2];
            for(i=0; i<2; i++){
                //			Rcol[i].create(3,1,CV_32FC1);
                Rcol[i] = tmpM.col(i);
                lambda[i] = 1.0 / cv::norm(Rcol[i], NORM_L2);
                rMat[i] = Rcol[i] * lambda[i];
                //			lambda[i] /= camera_matrix.at<_Tp>(i, i);
                printf("lambda %d: %f\n", i, lambda[i]);
            }
            
            rMat[2] = rMat[0].cross(rMat[1]);
            printf("rotation & translation:\n");
            rotation.create(3,3,tmpM.type());
            translation.create(3,1,tmpM.type());
            for(j=0; j<3; j++){
                for(i=0; i<3; i++){
                    rotation.at<_Tp>(i,j) = rMat[j].at<_Tp>(i,0);
                    printf("%f,",rotation.at<_Tp>(i,j));
                }
                translation.at<_Tp>(j,0) = tmpM.at<_Tp>(j,2) * lambda[0];
                printf("\t%f\n", translation.at<_Tp>(j,0));
            }
        }
        catch(std::exception& e){
            throw e;
        }
    }
    
    
    // Change homography to the rotation matrix and translation matrix
    // H_mat: homography matrix
    void decomposeHomography(Mat& H_mat, Mat& camera_matrix, Mat& rotation, Mat& translation, Point2f marker_center)
    {
        try{
            //		CV_Assert(rotation.cols == 3 && rotation.rows == 3 && translation.cols == 1 && translation.rows == 3);
            CV_Assert(H_mat.type() == CV_32FC1 || H_mat.type() == CV_64FC1);
            
            if(H_mat.type() == CV_32FC1){
                decomposeHomographyType<float>(H_mat, camera_matrix, rotation, translation, marker_center);
            }
            else if(H_mat.type() == CV_64FC1){
                decomposeHomographyType<double>(H_mat, camera_matrix, rotation, translation, marker_center);
            }
        }
        catch(std::exception& e){
            throw e;
        }
    }
    
    
    template<typename _Tp> void decomposeHomographyType(Mat& H_mat, Mat& camera_matrix, Mat& rotation, Mat& translation)
    {
        Point2f marker_center(camera_matrix.at<_Tp>(0,2), camera_matrix.at<_Tp>(1,2));
        try{
            decomposeHomographyType<_Tp>(H_mat, camera_matrix, rotation, translation, marker_center);
        }
        catch(std::exception& e){
            throw e;
        }
    }
    
    
    // Calculate Rotation and Translation Matrix from Homography
    // H_mat: Homography Matrix
    void decomposeHomography(Mat& H_mat, Mat& camera_matrix, Mat& rotation, Mat& translation)
    {
        try{
            CV_Assert(H_mat.type() == CV_32FC1 || H_mat.type() == CV_64FC1);
            
            if(H_mat.type() == CV_32FC1){
                decomposeHomographyType<float>(H_mat, camera_matrix, rotation, translation);
            }
            else if(H_mat.type() == CV_64FC1){
                decomposeHomographyType<double>(H_mat, camera_matrix, rotation, translation);
            }
        }
        catch(std::exception& e){
            throw e;
        }
    }
};
