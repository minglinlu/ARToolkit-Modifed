/*
 *	nftSimple.c
 *  ARToolKit5
 *
 *	Demonstration of ARToolKit NFT. Renders a colour cube.
 *
 *  Press '?' while running for help on available key commands.
 *
 *  Disclaimer: IMPORTANT:  This Daqri software is supplied to you by Daqri
 *  LLC ("Daqri") in consideration of your agreement to the following
 *  terms, and your use, installation, modification or redistribution of
 *  this Daqri software constitutes acceptance of these terms.  If you do
 *  not agree with these terms, please do not use, install, modify or
 *  redistribute this Daqri software.
 *
 *  In consideration of your agreement to abide by the following terms, and
 *  subject to these terms, Daqri grants you a personal, non-exclusive
 *  license, under Daqri's copyrights in this original Daqri software (the
 *  "Daqri Software"), to use, reproduce, modify and redistribute the Daqri
 *  Software, with or without modifications, in source and/or binary forms;
 *  provided that if you redistribute the Daqri Software in its entirety and
 *  without modifications, you must retain this notice and the following
 *  text and disclaimers in all such redistributions of the Daqri Software.
 *  Neither the name, trademarks, service marks or logos of Daqri LLC may
 *  be used to endorse or promote products derived from the Daqri Software
 *  without specific prior written permission from Daqri.  Except as
 *  expressly stated in this notice, no other rights or licenses, express or
 *  implied, are granted by Daqri herein, including but not limited to any
 *  patent rights that may be infringed by your derivative works or by other
 *  works in which the Daqri Software may be incorporated.
 *
 *  The Daqri Software is provided by Daqri on an "AS IS" basis.  DAQRI
 *  MAKES NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION
 *  THE IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE, REGARDING THE DAQRI SOFTWARE OR ITS USE AND
 *  OPERATION ALONE OR IN COMBINATION WITH YOUR PRODUCTS.
 *
 *  IN NO EVENT SHALL DAQRI BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL
 *  OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION,
 *  MODIFICATION AND/OR DISTRIBUTION OF THE DAQRI SOFTWARE, HOWEVER CAUSED
 *  AND WHETHER UNDER THEORY OF CONTRACT, TORT (INCLUDING NEGLIGENCE),
 *  STRICT LIABILITY OR OTHERWISE, EVEN IF DAQRI HAS BEEN ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Copyright 2015 Daqri LLC. All Rights Reserved.
 *  Copyright 2007-2015 ARToolworks, Inc. All Rights Reserved.
 *
 *  Author(s): Philip Lamb.
 *
 */


// ============================================================================
//	Includes
// ============================================================================

#ifdef _WIN32
#  include <windows.h>
#endif
#include <stdio.h>
#ifdef _WIN32
#  define snprintf _snprintf
#endif
#include <string.h>
#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glut.h>
#endif

#include <AR/ar.h>
#include <AR/arMulti.h>
#include <AR/video.h>
#include <AR/gsub_lite.h>
#include <AR/arFilterTransMat.h>
#include <AR2/tracking.h>

#include "ARMarkerNFT.h"
#include "trackingSub.h"
#include "commonCvFunctions.h"

// =========================LML=ADD============================================
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <thread>
#include <mutex>
#include <unistd.h>
#include "gflags/gflags.h"
#include "vocab_tree/vocab_tree.h"
#include "feature/opencv_libvot_api.h"
#include "utils/io_utils.h"
#include "utils/data_structures.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/video/tracking.hpp>

#include <Eigen/Dense>
#include <Eigen/SVD>
using namespace std;
using namespace cv;
using namespace cvar;

DEFINE_string(output_folder, "", "Output folder for ranked list");
DEFINE_int32(match_num, 10, "The length of the ranked list (top-k)");
DEFINE_bool(output_filename, true, "Output image name instead of image index");

typedef struct _target{
    unsigned int id;
    bool valid;
    ARPose pose;
    vector<cv::Point2f> object_position;
} target;
// ============================================================================

// ============================================================================
//	Constants
// ============================================================================

#define PAGES_MAX               10          // Maximum number of pages expected. You can change this down (to save memory) or up (to accomodate more pages.)

#define VIEW_SCALEFACTOR		1.0			// Units received from ARToolKit tracking will be multiplied by this factor before being used in OpenGL drawing.
#define VIEW_DISTANCE_MIN		10.0		// Objects closer to the camera than this will not be displayed. OpenGL units.
#define VIEW_DISTANCE_MAX		10000.0		// Objects further away from the camera than this will not be displayed. OpenGL units.

// ============================================================================
//	Global variables
// ============================================================================

// Preferences.
static int prefWindowed = TRUE;
static int prefWidth = 640;					// Fullscreen mode width.
static int prefHeight = 480;				// Fullscreen mode height.
static int prefDepth = 32;					// Fullscreen mode bit depth.
static int prefRefresh = 0;					// Fullscreen mode refresh rate. Set to 0 to use default rate.
static int targetNum = 0;                   // Number of detect targets
static int targetId = 0;                    // Initial target ID
mutex mutex_targetsList;
vector<target*> targetsList;
static cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << 678.29, 0, 318.29, 0 , 637.774, 237.9, 0, 0, 1);
static cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion

// Image acquisition.
static ARUint8		*gARTImage = NULL;
static long			gCallCountMarkerDetect = 0;

// Markers.
ARMarkerNFT *markersNFT = NULL;
int markersNFTCount = 0;

// NFT.
static THREAD_HANDLE_T     *threadHandle = NULL;
static AR2HandleT          *ar2Handle = NULL;
static KpmHandle           *kpmHandle = NULL;
static int                  surfaceSetCount = 0;
static AR2SurfaceSetT      *surfaceSet[PAGES_MAX];
// NFT results.
static int detectedPage = -2; // -2 Tracking not inited, -1 tracking inited OK, >= 0 tracking online on page.
static float trackingTrans[3][4];


// Drawing.
static int gWindowW;
static int gWindowH;
static ARParamLT *gCparamLT = NULL;
static ARGL_CONTEXT_SETTINGS_REF gArglSettings = NULL;
static int gDrawRotate = FALSE;
static float gDrawRotateAngle = 0;			// For use in drawing.
static ARdouble cameraLens[16];


// ============================================================================
//	Function prototypes
// ============================================================================

static int setupCamera(const char *cparam_name, char *vconf, ARParamLT **cparamLT_p);
static int initNFT(ARParamLT *cparamLT, AR_PIXEL_FORMAT pixFormat);
static int loadNFTData(void);
static void cleanup(void);
static void Keyboard(unsigned char key, int x, int y);
static void Visibility(int visible);
static void Reshape(int w, int h);
static void Display(void);
static void detect(int a, int b, int c);
static void track(cv::Mat capImage,string queryImage);

// ============================================================================
//	Functions
string sift_filename = "/Users/lml/Desktop/image.orig/173.sift";
const char *db_image_list = "/Users/lml/Desktop/CPP/libvot/build/bin/image_list";
const char *image_db = "/Users/lml/Desktop/CPP/libvot/build/bin/vocab_out/db.out";
const std::string output_folder = FLAGS_output_folder;
const int num_matches = FLAGS_match_num;
const bool is_output_name = FLAGS_output_filename;
std::vector<std::string> db_image_filenames;
vot::VocabTree *tree = new vot::VocabTree();
cv::Mat input_img;
cv::VideoCapture webcam = cv::VideoCapture(0);
cv::SiftDescriptorExtractor cv_sift_detector;

void detect(int a, int b, int c)
{
    while(1){
        if(detectedPage==-2){
            //1.extract sift descriptor
            std::vector<cv::KeyPoint> cv_keypoints;
            cv::Mat sift_descriptors;
            cv_sift_detector.detect(input_img, cv_keypoints);
            cv_sift_detector.compute(input_img, cv_keypoints, sift_descriptors);
            vot::SiftData sift_data;
            vot::OpencvKeyPoints2libvotSift(cv_keypoints, sift_descriptors, sift_data);
            //2.query the most similar image.
            int db_image_num = tree->database_image_num;
            float *scores = new float[db_image_num];
            size_t *indexed_scores = new size_t[db_image_num];
            memset(scores, 0.0, sizeof(float) * db_image_num);
            tree->Query(sift_data, scores);
            
            std::iota(indexed_scores, indexed_scores+db_image_num, 0);
            std::sort(indexed_scores, indexed_scores+db_image_num, [&](size_t i0, size_t i1) {
                return scores[i0] > scores[i1];
            });
            std::string image_name = tw::IO::GetFilename(db_image_filenames[indexed_scores[0]]);
            if(scores[indexed_scores[0]]>=0.03){
                cout<<"found image:"<<image_name.c_str()<<"sim:"<<scores[indexed_scores[0]]<<endl;
                detectedPage=atoi(image_name.c_str());
                std::thread tracker(track,input_img,image_name);
//                std::thread tracker(track,input_img,"0");
                tracker.join();
            }
        }
    }
}

SurfFeatureDetector surf_detector(2000,4);
FREAK extractor;
bool isMatched(Mat &srcImage,Mat &dstImage,vector<KeyPoint> &src_points,vector<KeyPoint> &dst_points,vector<DMatch> &matches){
    Mat src_descriptor,dst_descriptor;
    surf_detector.detect(srcImage,src_points);
    surf_detector.detect(dstImage,dst_points);
    
    extractor.compute(srcImage,src_points,src_descriptor);
    extractor.compute(dstImage,dst_points,dst_descriptor);
    if(dst_points.size()<6||src_points.size()<6){
        cout<<"tracking lost, matches number <6. "<<endl;
        detectedPage=-2;
        return false;
    }
    
    BFMatcher matcher (NORM_HAMMING,true);
    matcher.match(src_descriptor,dst_descriptor,matches);
    
    //cout<<"original match points     :"<<matches.size()<<endl;
    if(matches.size()<6){
        cout<<"tracking lost, matches number <6. "<<endl;
        
        return false;
    }
    return true;
}

vector<KeyPoint> getInliners(vector<KeyPoint> src_points,vector<KeyPoint> dst_points,vector<DMatch> &final_matches,vector<cv::Point3f> &src_3D,vector<cv::Point2f> &dst_2D,cv::Mat &H){
    vector<cv::Point2f> trainmatches, querymatches;
    vector<cv::KeyPoint> p1, p2, src_p1, dst_p2,inliners;
    
    for (int i = 0; i < final_matches.size(); i++){
        p1.push_back (src_points[final_matches[i].queryIdx]);
        p2.push_back (dst_points[final_matches[i].trainIdx]);
    }
    
    for (int i = 0; i < p1.size(); i++){
        querymatches.push_back (p1[i].pt);
        trainmatches.push_back (p2[i].pt);
    }
    
    vector<uchar> status;
    H = findHomography (querymatches, trainmatches, status, CV_FM_RANSAC, 10);
    //cout<<h<<endl;
    int index = 0;
    vector<DMatch> super_final_matches;
    
    src_3D.clear();
    dst_2D.clear();
    for (int i = 0; i < final_matches.size(); i++)
    {
        if (status[i] != 0)
        {
            super_final_matches.push_back (final_matches[i]);
            inliners.push_back(dst_points[final_matches[i].trainIdx]);
            //src_p1.push_back(src_points[final_matches[i].queryIdx]);
            //dst_p2.push_back(dst_points[final_matches[i].trainIdx]);
            src_3D.push_back(cv::Point3d(src_points[final_matches[i].queryIdx].pt.x, src_points[final_matches[i].queryIdx].pt.y, 0.0f));
            dst_2D.push_back(dst_points[final_matches[i].trainIdx].pt);
            index++;
        }
    }
    return inliners;
}

void updateCamPose(vector<cv::Point3f> &src_3D,vector<cv::Point2f> &dst_2D,Mat &rotation_vector,Mat &translation_vector,Mat &_R_matrix){
    // Solve for pose
    //cv::solvePnPRansac(src_3D, dst_2D, camera_matrix, dist_coeffs, rotation_vector, translation_vector);
    cv::solvePnP(src_3D, dst_2D, camera_matrix, dist_coeffs, rotation_vector, translation_vector);
    Rodrigues(rotation_vector,_R_matrix);                   // converts Rotation Vector to Matrix
    _R_matrix.convertTo(_R_matrix, CV_32FC1);
    translation_vector.convertTo(translation_vector, CV_32FC1);
    //    cout<<_R_matrix<<endl;
    //    cout<<translation_vector<<endl;
}

void trackingLost(target *new_target){
    cout<<"tracking lost ,now size is:"<<targetsList.size()<<endl;
    mutex_targetsList.lock();
    for(auto itarget=targetsList.begin();itarget!=targetsList.end();){
        if((*itarget)->id==new_target->id){
            itarget=targetsList.erase(itarget);
        }else{
            itarget++;
        }
    }
    mutex_targetsList.unlock();
    delete new_target;
    detectedPage=-2;
}

void track(cv::Mat capImage,string queryImage){
    queryImage="/Users/lml/Desktop/image.orig/"+queryImage+".jpg";
    //cout<<queryImage<<endl;
    Mat dstImage,prevImage;
    Mat srcImage=imread(queryImage,0); // 数据库中的图像
    cvtColor(capImage, dstImage, CV_BGR2GRAY); // 转为灰度图像，摄像头的输入图像
    prevImage=dstImage.clone(); // 上一帧
    
    vector<KeyPoint> src_points,dst_points;
    vector<DMatch> final_matches,matches;
    
    if(!isMatched(srcImage, dstImage, src_points, dst_points,matches)){
        detectedPage=-2;
        return;
    }
    
    //接下来是RANSAC剔除误匹配
    final_matches = matches;
    vector<cv::Point3f> src_3D;
    vector<cv::Point2f> dst_2D;
    cv::Mat H;
    vector<KeyPoint> inliners=getInliners(src_points, dst_points,final_matches,src_3D,dst_2D,H);
    int match_num =(int)inliners.size();
    //cout << "number of inlier_matches : " << match_num << endl;
    if(match_num<6){
        cout<<"tracking lost, matches number <6. "<<endl;
        detectedPage=-2;
        return;
    }
//    Mat outimg;
//    drawKeypoints(prevImage, inliners, outimg , Scalar(255,0,0));
//    imshow("outimg",outimg);
//    return;
    // Output rotation and translation
    cv::Mat rotation_vector; // Rotation in axis-angle form
    cv::Mat translation_vector;
    cv::Mat _R_matrix;
    
    /*
    Mat Amarker=camera_matrix.clone();
    Mat invA =camera_matrix.inv();
    Mat tmpM = invA*H*Amarker;
    
    Mat Rcol[2],rMat[3];
    
    double lambda[2];
    for(int i=0; i<2; i++){
        //            Rcol[i].create(3,1,CV_32FC1);
        Rcol[i] = tmpM.col(i);
        lambda[i] = 1.0 / cv::norm(Rcol[i], NORM_L2);
        rMat[i] = Rcol[i] * lambda[i];
        //            lambda[i] /= camera_matrix.at<_Tp>(i, i);
        printf("lambda %d: %f\n", i, lambda[i]);
    }
    
    rMat[2] = rMat[0].cross(rMat[1]);
    cout<<tmpM<<endl;
    //printf("rotation & translation:\n");
    rotation_vector.create(3,3,tmpM.type());
    translation_vector.create(3,1,tmpM.type());
    for(int j=0; j<3; j++){
        for(int i=0; i<3; i++){
            rotation_vector.at<double>(i,j) = rMat[j].at<double>(i,0);
            printf("%f,",rotation_vector.at<double>(i,j));
        }
        translation_vector.at<double>(j,0) = tmpM.at<double>(j,2) * lambda[0];
        printf("\t%f\n", translation_vector.at<double>(j,0));
    }*/
    
    //updateCamPose(src_3D,dst_2D,rotation_vector,translation_vector,_R_matrix);
    
    //cout<<_R_matrix.size()<<endl;
    Size size=srcImage.size();
    vector<cv::Point2f> pos_points=calcAffineTransformRect(size, H);
    //cout<<pos_points<<endl;
    
    target *new_target = new target();
    new_target->object_position=pos_points;
    new_target->id=++targetId;
//    new_target->pose={0.7657,0.1866,-0.6155,0,-0.2725,0.9609,-0.0477,0,0.5826,0.2042,0.7866,0,-232.2759,-92.8866,-826.4772,1};
    new_target->pose={
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        320,240,-800,1
    };
//    new_target->pose={
//        rotation_vector.at<double>(0,0),rotation_vector.at<double>(1,0),rotation_vector.at<double>(2,0),0,
//        rotation_vector.at<double>(0,1),rotation_vector.at<double>(1,1),rotation_vector.at<double>(2,1),0,
//        rotation_vector.at<double>(0,2),rotation_vector.at<double>(1,2),rotation_vector.at<double>(2,2),0,
//        translation_vector.at<double>(0,0),translation_vector.at<double>(0,1),translation_vector.at<double>(0,2),1
//    };
    new_target->valid=true;
    targetsList.push_back(new_target);
    
    while(1){
        //continue;
        //cout<<"tracking..."<<endl;
        dstImage=input_img.clone();
        cvtColor(dstImage, dstImage, CV_BGR2GRAY); // 转为灰度图像，摄像头的输入图像
        vector<cv::Point2f> next_corners;
        vector<float> err;
        vector<unsigned char> track_status;
        
        cv::calcOpticalFlowPyrLK(prevImage, dstImage, dst_2D, next_corners, track_status, err);
        new_target->pose.T[12]=next_corners[0].x-320;
        new_target->pose.T[13]=240-next_corners[0].y;
        //cout<<dst_2D<<endl;
        Mat outimg;
        inliners.clear();
        for( size_t i = 0; i < dst_2D.size(); i++ ) {
            inliners.push_back(cv::KeyPoint(dst_2D[i], 1.f));
        }
//        drawKeypoints(dstImage, inliners, outimg , Scalar(255,0,0));
//        namedWindow("outimg");
//        imshow("outimg",outimg);
        //startWindowThread();
        //imwrite("/Users/lml/Desktop/tmp.jpg",outimg);
        int tr_num = 0;
        vector<unsigned char>::iterator status_itr = track_status.begin();
        while(status_itr != track_status.end()){
            if(*status_itr > 0)
                tr_num++;
            status_itr++;
        }
        if(tr_num < 6){
            trackingLost(new_target);
            return ;
        }
        else{
            H = findHomography(Mat(dst_2D), Mat(next_corners), track_status,CV_RANSAC,5);
            if(countNonZero(H)==0){
                return;
            }
            else{
                cout<<new_target->object_position<<endl;
                vector<cv::Point2f> next_object_position = calcAffineTransformPoints(new_target->object_position, H);
                if(!checkPtInsideImage(prevImage.size(), next_object_position)||!checkRectShape(next_object_position)||checkInsideArea(next_corners, next_object_position, track_status)<6){
                    trackingLost(new_target);
                    return;
                }
                Mat outimg=dstImage.clone();
                cv::line(outimg,next_object_position[3],next_object_position[0],Scalar(255,0,0));
                for(int i=0;i<3;i++){
                    line(outimg,next_object_position[i],next_object_position[i+1],Scalar(255,0,5));
                }
                inliners.clear();
                for( size_t i = 0; i < next_corners.size(); i++ ) {
                    inliners.push_back(cv::KeyPoint(next_corners[i], 1.f));
                }
                drawKeypoints(outimg, inliners, outimg , Scalar(255,0,0));
                namedWindow("result");
                imshow("result",outimg);
                new_target->object_position=next_object_position;
                dstImage.copyTo(prevImage);
                dst_2D = next_corners;
            }
         }
//        if(!isMatched(srcImage, dstImage, src_points, dst_points,matches)){
//            trackingLost(new_target);
//            detectedPage=-2;
//            break;
//        }
        
        //接下来是RANSAC剔除误匹配
//        final_matches = matches;
//        vector<cv::Point3d> src_3D;
//        vector<cv::Point2d> dst_2D;
//        inliners=getInliners(src_points, dst_points,final_matches,src_3D,dst_2D,H);
//        int match_num=(int)inliners.size();
//        Mat imgMatch;
//        drawMatches(srcImage, src_points, dstImage, dst_points, super_final_matches, imgMatch);
//        imshow("imgMatch",imgMatch);
       // cout << "number of inlier_matches : " << match_num << endl;
//        if(match_num<6){
//            cout<<"tracking lost, inlier_matches number <6. "<<endl;
//            trackingLost(new_target);
//            detectedPage=-2;
//            break;
//        }
        
        //updateCamPose(src_3D,dst_2D,rotation_vector,translation_vector,_R_matrix);
//        new_target->pose={
//            _R_matrix.at<float>(0,0),_R_matrix.at<float>(1,0),_R_matrix.at<float>(2,0),0,
//            _R_matrix.at<float>(0,1),_R_matrix.at<float>(1,1),_R_matrix.at<float>(2,1),0,
//            _R_matrix.at<float>(0,2),_R_matrix.at<float>(1,2),_R_matrix.at<float>(2,2),0,
//            translation_vector.at<float>(0,0),translation_vector.at<float>(0,1),translation_vector.at<float>(0,2),1
//        };
        
        detectedPage=-1;
    }
}

// ============================================================================

int main(int argc, char** argv)
{
    // ============================================================================
    webcam.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    webcam.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    if (!webcam.isOpened()){
        webcam.release();
        std::cerr << "Error during opening capture device!" << std::endl;
        return -1;
    }
    
    // get db image filenames
    tw::IO::ExtractLines(db_image_list, db_image_filenames);
    
    tree->ReadTree(image_db);
    std::cout << "[VocabMatch] Successfully read vocabulary tree (with image database) file " << image_db << std::endl;
    tree->Show();
    
    vot::SiftData sift;
    std::string sift_str(sift_filename);
    const std::string sift_str_ext = tw::IO::SplitPathExt(sift_str).second;
    if (sift_str_ext == "sift") {
        if (!sift.ReadSiftFile(sift_str)) {
            std::cout << "[VocabMatch] ReadSiftFile error\n";
            exit(-1);
        }
    }
    else if(sift_str_ext == "desc") {
        if (!sift.ReadOpenmvgDesc<DTYPE, FDIM>(sift_str)) {
            std::cout << "[VocabMatch] ReadOpenmvgDesc error\n";
            exit(-1);
        }
    }
    else
        std::cout << "[VocabMatch] Ext not supported by libvot\n";
    
    // get rank list output path from sift_filename
    std::string output_path = sift_str;
    output_path = tw::IO::GetFilename(sift_str) + ".rank";
    output_path = tw::IO::JoinPath(output_folder, output_path);
    
    FILE *match_file = fopen(output_path.c_str(), "w");
    if (match_file == NULL) {
        std::cout << "[VocabMatch] Fail to open the match file.\n";
        return -1;
    }
    int db_image_num = tree->database_image_num;
    float *scores = new float[db_image_num];
    size_t *indexed_scores = new size_t[db_image_num];
    memset(scores, 0.0, sizeof(float) * db_image_num);
    tree->Query(sift, scores);
    
    std::iota(indexed_scores, indexed_scores+db_image_num, 0);
    std::sort(indexed_scores, indexed_scores+db_image_num, [&](size_t i0, size_t i1) {
        return scores[i0] > scores[i1];
    });
    
    int top = num_matches > db_image_num ? db_image_num : num_matches;
    if (is_output_name) {
        for (size_t j = 0; j < top; j++) {
            std::string image_name = tw::IO::GetFilename(db_image_filenames[indexed_scores[j]]);
            cout<<j<<":"<<image_name<<":"<<scores[indexed_scores[j]]<<endl;
            fprintf(match_file, "%d:%s:%f\n", j,image_name.c_str(),scores[indexed_scores[j]]);
        }
    }
    else {
        for (size_t j = 0; j < top; j++)
            fprintf(match_file, "%zu\n", indexed_scores[j]);
    }
    char buffer[250];
    getcwd(buffer, 250);
    std::cout << "[VocabMatch] Successful query and the rank list is output to " << buffer<<"/"<<output_path << ".\n";
    fclose(match_file);
    
    //创建线程对象detector,绑定线程函数为detector
    std::thread detector(detect, 1, 2, 3);
    //输出t1的线程ID
    //std::cout << "ID:" << t1.get_id() << std::endl;
    //等待t1线程函数执行结束
    //detector.join();
    // ============================================================================
    char glutGamemode[32];
    const char *cparam_name = "Data2/camera_para.dat";
    char vconf[] = "";
    const char markerConfigDataFilename[] = "Data2/markers.dat";
    
#ifdef DEBUG
    arLogLevel = AR_LOG_LEVEL_DEBUG;
#endif
    
    //
    // Library inits.
    //
    
    glutInit(&argc, argv);
    
    //
    // Video setup.
    //
    
#ifdef _WIN32
    CoInitialize(NULL);
#endif
    
    if (!setupCamera(cparam_name, vconf, &gCparamLT)) {
        ARLOGe("main(): Unable to set up AR camera.\n");
        exit(-1);
    }
    
    //
    // AR init.
    //
    
    // Create the OpenGL projection from the calibrated camera parameters.
    arglCameraFrustumRH(&(gCparamLT->param), VIEW_DISTANCE_MIN, VIEW_DISTANCE_MAX, cameraLens);
    /*
     if (!initNFT(gCparamLT, arVideoGetPixelFormat())) {
     ARLOGe("main(): Unable to init NFT.\n");
     exit(-1);
     }*/
    
    //
    // Graphics setup.
    //
    
    // Set up GL context(s) for OpenGL to draw into.
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    if (!prefWindowed) {
        if (prefRefresh) sprintf(glutGamemode, "%ix%i:%i@%i", prefWidth, prefHeight, prefDepth, prefRefresh);
        else sprintf(glutGamemode, "%ix%i:%i", prefWidth, prefHeight, prefDepth);
        glutGameModeString(glutGamemode);
        glutEnterGameMode();
    } else {
        glutInitWindowSize(gCparamLT->param.xsize, gCparamLT->param.ysize);
        glutCreateWindow(argv[0]);
    }
    
    // Setup ARgsub_lite library for current OpenGL context.
    if ((gArglSettings = arglSetupForCurrentContext(&(gCparamLT->param), arVideoGetPixelFormat())) == NULL) {
        ARLOGe("main(): arglSetupForCurrentContext() returned error.\n");
        cleanup();
        exit(-1);
    }
    arUtilTimerReset();
    
    //
    // Markers setup.
    //
    
    // Load marker(s).
    /*newMarkers(markerConfigDataFilename, &markersNFT, &markersNFTCount);
     if (!markersNFTCount) {
     ARLOGe("Error loading markers from config. file '%s'.\n", markerConfigDataFilename);
     cleanup();
     exit(-1);
     }
     ARLOGi("Marker count = %d\n", markersNFTCount);
     
     // Marker data has been loaded, so now load NFT data.
     if (!loadNFTData()) {
     ARLOGe("Error loading NFT data.\n");
     cleanup();
     exit(-1);
     }    */
    
    // Start the video.
    if (arVideoCapStart() != 0) {
        ARLOGe("setupCamera(): Unable to begin camera data capture.\n");
        return (FALSE);
    }
    
    // Register GLUT event-handling callbacks.
    // NB: mainLoop() is registered by Visibility.
    glutDisplayFunc(Display);
    glutReshapeFunc(Reshape);
    glutVisibilityFunc(Visibility);
    glutKeyboardFunc(Keyboard);
    
    glutMainLoop();
    
    return (0);
}

// Something to look at, draw a rotating colour cube.
static void DrawCube(void)
{
    // Colour cube data.
    int i;
    float fSize = 40.0f;
    const GLfloat cube_vertices [8][3] = {
        /* +z */ {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {-0.5f, -0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f},
        /* -z */ {0.5f, 0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f} };
    const GLubyte cube_vertex_colors [8][4] = {
        {255, 255, 255, 255}, {255, 255, 0, 255}, {0, 255, 0, 255}, {0, 255, 255, 255},
        {255, 0, 255, 255}, {255, 0, 0, 255}, {0, 0, 0, 255}, {0, 0, 255, 255} };
    const GLubyte cube_faces [6][4] = { /* ccw-winding */
        /* +z */ {3, 2, 1, 0}, /* -y */ {2, 3, 7, 6}, /* +y */ {0, 1, 5, 4},
        /* -x */ {3, 0, 4, 7}, /* +x */ {1, 2, 6, 5}, /* -z */ {4, 5, 6, 7} };
    
    glPushMatrix(); // Save world coordinate system.
    glRotatef(gDrawRotateAngle, 0.0f, 0.0f, 1.0f); // Rotate about z axis.
    glScalef(fSize, fSize, fSize);
    glTranslatef(0.0f, 0.0f, 0.5f); // Place base of cube on marker surface.
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_BLEND);
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, cube_vertex_colors);
    glVertexPointer(3, GL_FLOAT, 0, cube_vertices);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    for (i = 0; i < 6; i++) {
        glDrawElements(GL_TRIANGLE_FAN, 4, GL_UNSIGNED_BYTE, &(cube_faces[i][0]));
    }
    glDisableClientState(GL_COLOR_ARRAY);
    glColor4ub(0, 0, 0, 255);
    for (i = 0; i < 6; i++) {
        glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_BYTE, &(cube_faces[i][0]));
    }
    glPopMatrix();    // Restore world coordinate system.
}

static void DrawCubeUpdate(float timeDelta)
{
    if (gDrawRotate) {
        gDrawRotateAngle += timeDelta * 45.0f; // Rotate cube at 45 degrees per second.
        if (gDrawRotateAngle > 360.0f) gDrawRotateAngle -= 360.0f;
    }
}

static int setupCamera(const char *cparam_name, char *vconf, ARParamLT **cparamLT_p)
{	
    ARParam			cparam;
    int				xsize, ysize;
    AR_PIXEL_FORMAT pixFormat;
    
    // Open the video path.
    if (arVideoOpen(vconf) < 0) {
        ARLOGe("setupCamera(): Unable to open connection to camera.\n");
        return (FALSE);
    }
    
    // Find the size of the window.
    if (arVideoGetSize(&xsize, &ysize) < 0) {
        ARLOGe("setupCamera(): Unable to determine camera frame size.\n");
        arVideoClose();
        return (FALSE);
    }
    ARLOGi("Camera image size (x,y) = (%d,%d)\n", xsize, ysize);
    
    // Get the format in which the camera is returning pixels.
    pixFormat = arVideoGetPixelFormat();
    if (pixFormat == AR_PIXEL_FORMAT_INVALID) {
        ARLOGe("setupCamera(): Camera is using unsupported pixel format.\n");
        arVideoClose();
        return (FALSE);
    }
    
    // Load the camera parameters, resize for the window and init.
    if (arParamLoad(cparam_name, 1, &cparam) < 0) {
        ARLOGe("setupCamera(): Error loading parameter file %s for camera.\n", cparam_name);
        arVideoClose();
        return (FALSE);
    }
    if (cparam.xsize != xsize || cparam.ysize != ysize) {
        ARLOGw("*** Camera Parameter resized from %d, %d. ***\n", cparam.xsize, cparam.ysize);
        arParamChangeSize(&cparam, xsize, ysize, &cparam);
    }
#ifdef DEBUG
    ARLOG("*** Camera Parameter ***\n");
    arParamDisp(&cparam);
#endif
    if ((*cparamLT_p = arParamLTCreate(&cparam, AR_PARAM_LT_DEFAULT_OFFSET)) == NULL) {
        ARLOGe("setupCamera(): Error: arParamLTCreate.\n");
        arVideoClose();
        return (FALSE);
    }
    
    return (TRUE);
}

// Modifies globals: kpmHandle, ar2Handle.
static int initNFT(ARParamLT *cparamLT, AR_PIXEL_FORMAT pixFormat)
{
    ARLOGd("Initialising NFT.\n");
    //
    // NFT init.
    //
    
    // KPM init.
    kpmHandle = kpmCreateHandle(cparamLT, pixFormat);
    if (!kpmHandle) {
        ARLOGe("Error: kpmCreateHandle.\n");
        return (FALSE);
    }
    //kpmSetProcMode( kpmHandle, KpmProcHalfSize );
    
    // AR2 init.
    if( (ar2Handle = ar2CreateHandle(cparamLT, pixFormat, AR2_TRACKING_DEFAULT_THREAD_NUM)) == NULL ) {
        ARLOGe("Error: ar2CreateHandle.\n");
        kpmDeleteHandle(&kpmHandle);
        return (FALSE);
    }
    if (threadGetCPU() <= 1) {
        ARLOGi("Using NFT tracking settings for a single CPU.\n");
        ar2SetTrackingThresh(ar2Handle, 5.0);
        ar2SetSimThresh(ar2Handle, 0.50);
        ar2SetSearchFeatureNum(ar2Handle, 16);
        ar2SetSearchSize(ar2Handle, 6);
        ar2SetTemplateSize1(ar2Handle, 6);
        ar2SetTemplateSize2(ar2Handle, 6);
    } else {
        ARLOGi("Using NFT tracking settings for more than one CPU.\n");
        ar2SetTrackingThresh(ar2Handle, 5.0);
        ar2SetSimThresh(ar2Handle, 0.50);
        ar2SetSearchFeatureNum(ar2Handle, 16);
        ar2SetSearchSize(ar2Handle, 12);
        ar2SetTemplateSize1(ar2Handle, 6);
        ar2SetTemplateSize2(ar2Handle, 6);
    }
    // NFT dataset loading will happen later.
    return (TRUE);
}

// Modifies globals: threadHandle, surfaceSet[], surfaceSetCount
static int unloadNFTData(void)
{
    int i, j;
    
    if (threadHandle) {
        ARLOGi("Stopping NFT2 tracking thread.\n");
        trackingInitQuit(&threadHandle);
    }
    j = 0;
    for (i = 0; i < surfaceSetCount; i++) {
        if (j == 0) ARLOGi("Unloading NFT tracking surfaces.\n");
        ar2FreeSurfaceSet(&surfaceSet[i]); // Also sets surfaceSet[i] to NULL.
        j++;
    }
    if (j > 0) ARLOGi("Unloaded %d NFT tracking surfaces.\n", j);
    surfaceSetCount = 0;
    
    return 0;
}

// References globals: markersNFTCount
// Modifies globals: threadHandle, surfaceSet[], surfaceSetCount, markersNFT[]
static int loadNFTData(void)
{
    int i;
    KpmRefDataSet *refDataSet;
    
    // If data was already loaded, stop KPM tracking thread and unload previously loaded data.
    if (threadHandle) {
        ARLOGi("Reloading NFT data.\n");
        unloadNFTData();
    } else {
        ARLOGi("Loading NFT data.\n");
    }
    
    refDataSet = NULL;
    
    for (i = 0; i < markersNFTCount; i++) {
        // Load KPM data.
        KpmRefDataSet  *refDataSet2;
        ARLOGi("Reading %s.fset3\n", markersNFT[i].datasetPathname);
        if (kpmLoadRefDataSet(markersNFT[i].datasetPathname, "fset3", &refDataSet2) < 0 ) {
            ARLOGe("Error reading KPM data from %s.fset3\n", markersNFT[i].datasetPathname);
            markersNFT[i].pageNo = -1;
            continue;
        }
        markersNFT[i].pageNo = surfaceSetCount;
        ARLOGi("  Assigned page no. %d.\n", surfaceSetCount);
        if (kpmChangePageNoOfRefDataSet(refDataSet2, KpmChangePageNoAllPages, surfaceSetCount) < 0) {
            ARLOGe("Error: kpmChangePageNoOfRefDataSet\n");
            exit(-1);
        }
        if (kpmMergeRefDataSet(&refDataSet, &refDataSet2) < 0) {
            ARLOGe("Error: kpmMergeRefDataSet\n");
            exit(-1);
        }
        ARLOGi("  Done.\n");
        
        // Load AR2 data.
        ARLOGi("Reading %s.fset\n", markersNFT[i].datasetPathname);
        
        if ((surfaceSet[surfaceSetCount] = ar2ReadSurfaceSet(markersNFT[i].datasetPathname, "fset", NULL)) == NULL ) {
            ARLOGe("Error reading data from %s.fset\n", markersNFT[i].datasetPathname);
        }
        ARLOGi("  Done.\n");
        
        surfaceSetCount++;
        if (surfaceSetCount == PAGES_MAX) break;
    }
    if (kpmSetRefDataSet(kpmHandle, refDataSet) < 0) {
        ARLOGe("Error: kpmSetRefDataSet\n");
        exit(-1);
    }
    kpmDeleteRefDataSet(&refDataSet);
    
    // Start the KPM tracking thread.
    threadHandle = trackingInitInit(kpmHandle);
    if (!threadHandle) exit(-1);
    
    ARLOGi("Loading of NFT data complete.\n");
    return (TRUE);
}

static void cleanup(void)
{
    if (markersNFT) deleteMarkers(&markersNFT, &markersNFTCount);
    
    // NFT cleanup.
    unloadNFTData();
    ARLOGd("Cleaning up ARToolKit NFT handles.\n");
    ar2DeleteHandle(&ar2Handle);
    kpmDeleteHandle(&kpmHandle);
    arParamLTFree(&gCparamLT);
    
    // OpenGL cleanup.
    arglCleanup(gArglSettings);
    gArglSettings = NULL;
    
    // Camera cleanup.
    arVideoCapStop();
    arVideoClose();
#ifdef _WIN32
    CoUninitialize();
#endif
}

static void Keyboard(unsigned char key, int x, int y)
{
    switch (key) {
        case 0x1B:						// Quit.
        case 'Q':
        case 'q':
            cleanup();
            exit(0);
            break;
        case ' ':
            gDrawRotate = !gDrawRotate;
            break;
        case '?':
        case '/':
            ARLOG("Keys:\n");
            ARLOG(" q or [esc]    Quit demo.\n");
            ARLOG(" ? or /        Show this help.\n");
            ARLOG("\nAdditionally, the ARVideo library supplied the following help text:\n");
            arVideoDispOption();
            break;
        default:
            break;
    }
}


static void mainLoop(void)
{
    static int ms_prev;
    int ms;
    float s_elapsed;
    ARUint8 *image;
    
    int             i, j, k;
    
    // Find out how long since mainLoop() last ran.
    ms = glutGet(GLUT_ELAPSED_TIME);
    s_elapsed = (float)(ms - ms_prev) * 0.01f;
    if (s_elapsed < 0.01f) return; // Don't update more often than 100 Hz.
    ms_prev = ms;
    
    // Update drawing.
    DrawCubeUpdate(s_elapsed);
    
    // Grab a video frame.
    if ((image = arVideoGetImage()) != NULL) {
        gARTImage = image;	// Save the fetched image.
        webcam >> input_img;
        //input_img=imread("/Users/lml/Desktop/webcam.png");
        //imwrite("/Users/lml/Desktop/webcam.jpg", input_img);
        gCallCountMarkerDetect++; // Increment ARToolKit FPS counter.
        
        //char *convImg=(char *)gARTImage;
        //cv::Mat input = cv::Mat(gCparamLT->param.ysize, gCparamLT->param.xsize, CV_8UC1, convImg);
        //imshow("Display Image", input_img);
        
        //int len = strlen(img_name);
        //for (i = 0; i < len; i++) {
        
        //}
        //renderBitmapString(-1.2f, -0.2f , -5.0f, GLUT_BITMAP_HELVETICA_18, image_name);
        
        //3.calculate the
        
        /*
         // Run marker detection on frame
         if (threadHandle) {
         // Perform NFT tracking.
         float            err;
         int              ret;
         int              pageNo;
         
         if( detectedPage == -2 ) {
         trackingInitStart( threadHandle, gARTImage );
         detectedPage = -1;
         }
         if( detectedPage == -1 ) {
         ret = trackingInitGetResult( threadHandle, trackingTrans, &pageNo);
         if( ret == 1 ) {
         if (pageNo >= 0 && pageNo < surfaceSetCount) {
         ARLOGd("Detected page %d.\n", pageNo);
         detectedPage = pageNo;
         ar2SetInitTrans(surfaceSet[detectedPage], trackingTrans);
         } else {
         ARLOGe("Detected bad page %d.\n", pageNo);
         detectedPage = -2;
         }
         } else if( ret < 0 ) {
         ARLOGd("No page detected.\n");
         detectedPage = -2;
         }
         }
         if( detectedPage >= 0 && detectedPage < surfaceSetCount) {
         if( ar2Tracking(ar2Handle, surfaceSet[detectedPage], gARTImage, trackingTrans, &err) < 0 ) {
         ARLOGd("Tracking lost.\n");
         detectedPage = -2;
         } else {
         ARLOGd("Tracked page %d (max %d).\n", detectedPage, surfaceSetCount - 1);
         }
         }
         } else {
         ARLOGe("Error: threadHandle\n");
         detectedPage = -2;
         }
         
         // Update markers.
         for (i = 0; i < markersNFTCount; i++) {
         markersNFT[i].validPrev = markersNFT[i].valid;
         if (markersNFT[i].pageNo >= 0 && markersNFT[i].pageNo == detectedPage) {
         markersNFT[i].valid = TRUE;
         for (j = 0; j < 3; j++) for (k = 0; k < 4; k++) markersNFT[i].trans[j][k] = trackingTrans[j][k];
         }
         else markersNFT[i].valid = FALSE;
         if (markersNFT[i].valid) {
         
         // Filter the pose estimate.
         if (markersNFT[i].ftmi) {
         if (arFilterTransMat(markersNFT[i].ftmi, markersNFT[i].trans, !markersNFT[i].validPrev) < 0) {
         ARLOGe("arFilterTransMat error with marker %d.\n", i);
         }
         }
         
         if (!markersNFT[i].validPrev) {
         // Marker has become visible, tell any dependent objects.
         // --->
         }
         
         // We have a new pose, so set that.
         arglCameraViewRH((const ARdouble (*)[4])markersNFT[i].trans, markersNFT[i].pose.T, VIEW_SCALEFACTOR);
         // Tell any dependent objects about the update.
         // --->
         
         } else {
         
         if (markersNFT[i].validPrev) {
         // Marker has ceased to be visible, tell any dependent objects.
         // --->
         }
         }
         }
         */
        
        // Tell GLUT the display has changed.
        glutPostRedisplay();
    }
}

//
//	This function is called on events when the visibility of the
//	GLUT window changes (including when it first becomes visible).
//
static void Visibility(int visible)
{
    if (visible == GLUT_VISIBLE) {
        glutIdleFunc(mainLoop);
    } else {
        glutIdleFunc(NULL);
    }
}

//
//	This function is called when the
//	GLUT window is resized.
//
static void Reshape(int w, int h)
{
    gWindowW = w;
    gWindowH = h;
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, (GLsizei) w, (GLsizei) h);
    
    // Call through to anyone else who needs to know about window sizing here.
}

//
// This function is called when the window needs redrawing.
//
ARPose pose1={0.7657,0.1866,-0.6155,0,-0.2725,0.9609,-0.0477,0,0.5826,0.2042,0.7866,0,-232.2759,-92.8866,-826.4772,1};
ARPose pose2={0.9450,0.0680,0.3196,0,-0.0673,0.9976,-0.0133,0,-0.3198,-0.0089,0.9474,0,-37.6692,-58.5446,-1029.25,1};
static void Display(void)
{
    int i;
    
    // Select correct buffer for this context.
    glDrawBuffer(GL_BACK);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear the buffers for new frame.
    
    arglPixelBufferDataUpload(gArglSettings, gARTImage);
    arglDispImage(gArglSettings);
    gARTImage = NULL; // Invalidate image data.
    
    // Set up 3D mode.
    glMatrixMode(GL_PROJECTION);
#ifdef ARDOUBLE_IS_FLOAT
    glLoadMatrixf(cameraLens);
#else
    glLoadMatrixd(cameraLens);
#endif
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_DEPTH_TEST);
    
    // Set any initial per-frame GL state you require here.
    // --->
    
    // Lighting and geometry that moves with the camera should be added here.
    // (I.e. should be specified before marker pose transform.)
    // --->
    
    //    for (i = 0; i < markersNFTCount; i++) {
    //
    //        if (markersNFT[i].valid) {
    //
    //#ifdef ARDOUBLE_IS_FLOAT
    //            glLoadMatrixf(markersNFT[i].pose.T);
    //#else
    //            glLoadMatrixd(markersNFT[i].pose.T);
    //#endif
    //            // All lighting and geometry to be drawn relative to the marker goes here.
    //            // --->
    //            DrawCube();
    //        }
    //    }
    
    //    glLoadMatrixd(pose1.T);
    //    pose1.T[0]=pose1.T[0]+0.01;
    //    pose1.T[1]=pose1.T[1]+0.01;
    //    pose1.T[2]=pose1.T[2]+0.01;
    //    pose1.T[4]=pose1.T[4]+0.01;
    //    pose1.T[5]=pose1.T[5]+0.01;
    //    pose1.T[6]=pose1.T[6]+0.01;
    //    pose1.T[8]=pose1.T[8]+0.01;
    //    pose1.T[9]=pose1.T[9]+0.01;
    //    pose1.T[10]=pose1.T[10]+0.01;
    //    DrawCube();
    mutex_targetsList.lock();
    //cout<<"targetsList.size():"<<targetsList.size()<<endl;
    //gluLookAt(0,0,0,0,0,1,0,-1,0);
    for(auto target:targetsList){
        //target->pose.T[13]+=1;
        //target->pose.T[14]+=1;
        glLoadMatrixd(target->pose.T);
//                cout<<target->pose.T[0]<<","<<target->pose.T[1]<<","<<target->pose.T[2]<<target->pose.T[3]<<endl;
//                cout<<target->pose.T[4]<<","<<target->pose.T[5]<<","<<target->pose.T[6]<<target->pose.T[7]<<endl;
//                cout<<target->pose.T[8]<<","<<target->pose.T[9]<<","<<target->pose.T[10]<<target->pose.T[11]<<endl;
//                cout<<target->pose.T[12]<<","<<target->pose.T[13]<<","<<target->pose.T[14]<<","<<target->pose.T[15]<<endl;
        DrawCube();
    }
    mutex_targetsList.unlock();
    //glLoadMatrixd(pose2.T);
    //pose2.T[5]=pose2.T[5]+0.01;
    //DrawCube();
    
    // Set up 2D mode.
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, (GLdouble)gWindowW, 0, (GLdouble)gWindowH, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    
    // Add your own 2D overlays here.
    // --->
    
    glutSwapBuffers();
}
