//System include files
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

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <thread>
#include <mutex>
#include <unistd.h>

//OpenCV required
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/video/tracking.hpp>

//Using ARToolkit library
#include <AR/ar.h>
#include <AR/arMulti.h>
#include <AR/video.h>
#include <AR/gsub_lite.h>
#include <AR/arFilterTransMat.h>
#include <AR2/tracking.h>

//Local code
#include "ARMarkerNFT.h"
#include "trackingSub.h"
#include "commonCvFunctions.h"

//Reference other exsiting implementation of image retrieval
#include "gflags/gflags.h"
#include "vocab_tree/vocab_tree.h"
#include "feature/opencv_libvot_api.h"
#include "utils/io_utils.h"
#include "utils/data_structures.h"

using namespace std;
using namespace cv;
using namespace cvar;

DEFINE_string(output_folder, "", "Output folder for ranked list");

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
static int targetId = 0;                    // Initial target ID
mutex mutex_targetsList;
mutex mutex_detect;
condition_variable cond_detect;
bool canDetect=false;

//libvot config files
const char *db_image_list = "/Users/lml/Desktop/CPP/libvot/build/bin/image_list";
const char *image_db = "/Users/lml/Desktop/CPP/libvot/build/bin/vocab_out/db.out";
const std::string output_folder = FLAGS_output_folder;
std::vector<std::string> db_image_filenames;
vot::VocabTree *tree = new vot::VocabTree();
cv::Mat input_img;
cv::VideoCapture webcam = cv::VideoCapture(0);
cv::SiftDescriptorExtractor cv_sift_detector;

//the target struct
typedef struct _target{
    unsigned int id;
    std::string name;
    bool valid;
    ARPose pose;
    std::vector<cv::Point2f> object_position;
    std::vector<cv::Point> inliners;
} target;
vector<target*> targetsList;

static cv::Mat camera_matrix;
static cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion

// Image acquisition.
static ARUint8		*gARTImage = NULL;
static long			gCallCountMarkerDetect = 0;

// NFT.
static KpmHandle           *kpmHandle = NULL;
// NFT results.
static int detectedPage = -2; // -2 Tracking not inited, -1 tracking inited OK, >= 0 tracking online on page.

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
static void cleanup(void);
static void Keyboard(unsigned char key, int x, int y);
static void Visibility(int visible);
static void Reshape(int w, int h);
static void Display(void);
static void detect(int a, int b, int c);
static void track(cv::Mat capImage,string queryImage);
bool isMatched(Mat &srcImage,Mat &dstImage,vector<KeyPoint> &src_points,vector<KeyPoint> &dst_points,vector<DMatch> &matches);
vector<KeyPoint> getInliners(vector<KeyPoint> src_points,vector<KeyPoint> dst_points,vector<DMatch> &final_matches,vector<cv::Point3f> &src_3D,vector<cv::Point2f> &dst_2D,cv::Mat &H);
bool updateCamPose(vector<cv::Point3f> &src_3D,vector<cv::Point2f> &dst_2D,Mat &rotation_vector,Mat &translation_vector,float trackingTrans[3][4]);
void trackingLost(target *new_target);

// ============================================================================

void detect(int a, int b, int c)
{
    while(1){
        std::unique_lock<std::mutex> lk(mutex_detect);
        while(!canDetect){
            cond_detect.wait(lk);
        }
        if(detectedPage==-2){
            //filter the recognized region
            Mat erase_img=input_img.clone();
            mutex_targetsList.lock();
            for(auto &target:targetsList){
                RotatedRect rectPoint = minAreaRect(target->inliners);
                cv::Point2f fourPoint2f[4];
                rectPoint.points(fourPoint2f);
                for (int j = 0; j <= 3; j++){
                    line(erase_img, fourPoint2f[j], fourPoint2f[(j + 1) % 4], Scalar(0,0,255));
                }
                int minX=640,minY=480,maxX=0,maxY=0;
                for(auto point:fourPoint2f){
                    minX=point.x<minX?point.x:minX;
                    minY=point.y<minY?point.y:minY;
                    maxX=point.x>maxX?point.x:maxX;
                    maxY=point.y>maxY?point.y:maxY;
                }
                int width=maxX-minX;
                int height=maxY-minY;
                if(width<=0||height<=0||minX<=0||minY<=0){
                    trackingLost(target);
                    break;
                }
                Mat imageROI= erase_img(Rect(minX,minY,width,height));
                Mat patch(height,width , CV_8UC3, Scalar(0,255,0));
                Mat mask;
                cvtColor(patch, mask, CV_BGR2GRAY); // 转为灰度图像，摄像头的输入图像
                patch.copyTo(imageROI,mask);
                imwrite("/Users/lml/Desktop/tmp.jpg", erase_img);
                //cout<<target->object_position<<endl;
            }
            mutex_targetsList.unlock();
            
            //1.extract sift descriptor
            std::vector<cv::KeyPoint> cv_keypoints;
            cv::Mat sift_descriptors;
            cv_sift_detector.detect(erase_img, cv_keypoints);
            cv_sift_detector.compute(erase_img, cv_keypoints, sift_descriptors);
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
                bool hasSFlag=false;
                for(auto target:targetsList){
                    //cout<<target->name<<","<<image_name.c_str()<<endl;
                    if(target->name==image_name.c_str()){
                        hasSFlag=true;
                    }
                }
                if(!hasSFlag){
                    std::thread tracker(track,input_img,image_name);
                    tracker.detach();
                    detectedPage=-2;
                    canDetect=false;
                }
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

bool updateCamPose(vector<cv::Point3f> &src_3D,vector<cv::Point2f> &dst_2D,Mat &rotation_vector,Mat &translation_vector,float trackingTrans[3][4]){
    // Solve for pose
    //cv::solvePnPRansac(src_3D, dst_2D, camera_matrix, dist_coeffs, rotation_vector, translation_vector);
//    cv::solvePnP(src_3D, dst_2D, camera_matrix, dist_coeffs, rotation_vector, translation_vector);
//    Rodrigues(rotation_vector,_R_matrix);                   // converts Rotation Vector to Matrix
//    _R_matrix.convertTo(_R_matrix, CV_32FC1);
//    translation_vector.convertTo(translation_vector, CV_32FC1);
    //    cout<<_R_matrix<<endl;
    //    cout<<translation_vector<<endl;
    if(src_3D.size()<4) return false;
    ICPHandleT *icpHandle;
    ICPDataT icpData;
    ICP2DCoordT *sCoord;
    ICP3DCoordT *wCoord;
    ARdouble initMatXw2Xc[3][4];
    ARdouble err;
    int i;
    
    arMalloc(sCoord, ICP2DCoordT, src_3D.size());
    arMalloc(wCoord, ICP3DCoordT, src_3D.size());
    
    for(i=0;i<src_3D.size();i++){
        sCoord[i].x=dst_2D[i].x;
        sCoord[i].y=dst_2D[i].y;
        wCoord[i].x=src_3D[i].x;
        wCoord[i].y=src_3D[i].y;
        wCoord[i].z=src_3D[i].z;
    }
    icpData.num=i;
    icpData.screenCoord=&sCoord[0];
    icpData.worldCoord=&wCoord[0];
    
    if(icpGetInitXw2Xc_from_PlanarData(gCparamLT->param.mat, sCoord, wCoord, (int)src_3D.size(), initMatXw2Xc)<0){
        free(sCoord);
        free(wCoord);
        return false;
    }
    
    if((icpHandle=icpCreateHandle(gCparamLT->param.mat))==NULL){
        free(sCoord);
        free(wCoord);
        return false;
    }
    
    ARdouble camPosed[3][4];
    if(icpPoint(icpHandle, &icpData, initMatXw2Xc, camPosed, &err)<0){
        free(sCoord);
        free(wCoord);
        icpDeleteHandle(&icpHandle);
        return false;
    }
    
    for(int r=0;r<3;r++)for(int c=0;c<4;c++){trackingTrans[r][c]=(float)camPosed[r][c];}
    icpDeleteHandle(&icpHandle);
    free(sCoord);
    free(wCoord);
    if(err>10.0f)return false;
    return true;
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
    delete new_target;
    new_target=NULL;
    mutex_targetsList.unlock();
    {
        std::lock_guard<std::mutex> lk(mutex_detect);
        canDetect = true;
    }
    cond_detect.notify_one();
    detectedPage=-2;
}

void track(cv::Mat capImage,string index){
    sleep(1);
    string queryImage="/Users/lml/Desktop/image.orig/"+index+".jpg";
    //cout<<queryImage<<endl;
    Mat dstImage,prevImage;
    Mat srcImage=imread(queryImage,0); // 数据库中的图像
    cvtColor(capImage, dstImage, CV_BGR2GRAY); // 转为灰度图像，摄像头的输入图像
    prevImage=dstImage.clone(); // 上一帧
    
    vector<KeyPoint> src_points,dst_points;
    vector<DMatch> final_matches,matches;
    
    if(!isMatched(srcImage, dstImage, src_points, dst_points,matches)){
        cout<<"is not matched"<<endl;
        detectedPage=-2;
        {
            std::lock_guard<std::mutex> lk(mutex_detect);
            canDetect = true;
        }
        cond_detect.notify_one();
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
        {
            std::lock_guard<std::mutex> lk(mutex_detect);
            canDetect = true;
        }
        cond_detect.notify_one();
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
    float trackingTrans[3][4];
    if(!updateCamPose(src_3D,dst_2D,rotation_vector,translation_vector,trackingTrans)){
        {
            std::lock_guard<std::mutex> lk(mutex_detect);
            canDetect = true;
        }
        cond_detect.notify_one();
        return;
    }
//    for(int i=0;i<3;i++)
//        for(int j=0;j<4;j++){
//            cout<<trackingTrans[i][j]<<",";
//        }
//        cout<<endl;
//    }
//    return;
    Size size=srcImage.size();
    vector<cv::Point2f> pos_points=calcAffineTransformRect(size, H);
    
    target *new_target = new target();
    new_target->name=index;
    new_target->object_position=pos_points;
    new_target->inliners.clear();
    //    for(auto point:pos_points){
    //        new_target->inliners.push_back(cv::Point(point.x,point.y));
    //    }
    for(auto point:inliners){
        new_target->inliners.push_back(cv::Point(point.pt.x,point.pt.y));
    }
    new_target->id=++targetId;
    //    new_target->pose={0.7657,0.1866,-0.6155,0,-0.2725,0.9609,-0.0477,0,0.5826,0.2042,0.7866,0,-232.2759,-92.8866,-826.4772,1};
    //arglCameraViewRH((const ARdouble (*)[4])trackingTrans, new_target->pose.T, VIEW_SCALEFACTOR);
    new_target->pose={
        trackingTrans[0][0],-trackingTrans[1][0],-trackingTrans[2][0],0,
        trackingTrans[0][1],-trackingTrans[1][1],-trackingTrans[2][1],0,
        trackingTrans[0][2],-trackingTrans[1][2],-trackingTrans[2][2],0,
        trackingTrans[0][3],-trackingTrans[1][3],-trackingTrans[2][3],1
    };

//    cout<<new_target->pose.T[0]<<","<<new_target->pose.T[1]<<","<<new_target->pose.T[2]<<":"<<new_target->pose.T[3]<<endl;
//    cout<<new_target->pose.T[4]<<","<<new_target->pose.T[5]<<","<<new_target->pose.T[6]<<":"<<new_target->pose.T[7]<<endl;
//    cout<<new_target->pose.T[8]<<","<<new_target->pose.T[9]<<","<<new_target->pose.T[10]<<":"<<new_target->pose.T[11]<<endl;
//    cout<<new_target->pose.T[12]<<","<<new_target->pose.T[13]<<","<<new_target->pose.T[14]<<":"<<new_target->pose.T[15]<<endl;

    new_target->valid=true;
    mutex_targetsList.lock();
    targetsList.push_back(new_target);
    mutex_targetsList.unlock();
    {
        std::lock_guard<std::mutex> lk(mutex_detect);
        canDetect = true;
    }
    cond_detect.notify_one();
    detectedPage=-2;
    while(1){
        //continue;
        //cout<<"tracking..."<<endl;
        dstImage=input_img.clone();
        cvtColor(dstImage, dstImage, CV_BGR2GRAY); // 转为灰度图像，摄像头的输入图像
        vector<cv::Point2f> next_corners;
        vector<float> err;
        vector<unsigned char> track_status;
        
        cv::calcOpticalFlowPyrLK(prevImage, dstImage, dst_2D, next_corners, track_status, err);
//        double sumX=0,sumY=0;
//        for(int i=0;i<next_corners.size();i++){
//            sumX+=next_corners[i].x;
//            sumY+=next_corners[i].y;
//        }
//        sumX/=next_corners.size();
//        sumY/=next_corners.size();
//        new_target->pose.T[12]=sumX-gCparamLT->param.xsize/2;
//        new_target->pose.T[13]=gCparamLT->param.ysize/2-sumY;
        //cout<<new_target->pose.T[12]<<","<<new_target->pose.T[13]<<endl;
//        new_target->pose.T[14]=-800;
        Mat outimg;
        inliners.clear();
        new_target->inliners.clear();
        for( size_t i = 0; i < dst_2D.size(); i++ ) {
            inliners.push_back(cv::KeyPoint(dst_2D[i], 1.f));
            new_target->inliners.push_back(cv::Point(dst_2D[i].x,dst_2D[i].y));
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
                trackingLost(new_target);
                return;
            }
            else{
                //cout<<new_target->object_position<<endl;
                vector<cv::Point2f> next_object_position = calcAffineTransformPoints(new_target->object_position, H);
                if(!checkPtInsideImage(prevImage.size(), next_object_position)||!checkRectShape(next_object_position)||checkInsideArea(next_corners, next_object_position, track_status)<6){
                    trackingLost(new_target);
                    return;
                }
                
                new_target->object_position=next_object_position;
                dstImage.copyTo(prevImage);
                dst_2D = next_corners;
                if(!updateCamPose(src_3D,dst_2D,rotation_vector,translation_vector,trackingTrans)){
                    trackingLost(new_target);
                    return;
                }
                new_target->pose={
                    trackingTrans[0][0],-trackingTrans[1][0],-trackingTrans[2][0],0,
                    trackingTrans[0][1],-trackingTrans[1][1],-trackingTrans[2][1],0,
                    trackingTrans[0][2],-trackingTrans[1][2],-trackingTrans[2][2],0,
                    trackingTrans[0][3],-trackingTrans[1][3],-trackingTrans[2][3],1
                };
            }
        }
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
    // ============================================================================
    char glutGamemode[32];
    const char *cparam_name = "Data2/camera_para.dat";
    char vconf[] = "";
    
    glutInit(&argc, argv);
    
#ifdef _WIN32
    CoInitialize(NULL);
#endif
    
    if (!setupCamera(cparam_name, vconf, &gCparamLT)) {
        ARLOGe("main(): Unable to set up AR camera.\n");
        exit(-1);
    }
    
    // Create the OpenGL projection from the calibrated camera parameters.
    arglCameraFrustumRH(&(gCparamLT->param), VIEW_DISTANCE_MIN, VIEW_DISTANCE_MAX, cameraLens);
    
    // Set up GL context(s) for OpenGL to draw into.
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    if (!prefWindowed) {
        if (prefRefresh) sprintf(glutGamemode, "%ix%i:%i@%i", prefWidth, prefHeight, prefDepth, prefRefresh);
        else sprintf(glutGamemode, "%ix%i:%i", prefWidth, prefHeight, prefDepth);
        glutGameModeString(glutGamemode);
        glutEnterGameMode();
    } else {
        glutInitWindowSize(gCparamLT->param.xsize, gCparamLT->param.ysize);
        camera_matrix= (cv::Mat_<double>(3,3) << gCparamLT->param.mat[0][0],gCparamLT->param.mat[0][1],gCparamLT->param.mat[0][2],gCparamLT->param.mat[1][0],gCparamLT->param.mat[1][1],gCparamLT->param.mat[1][2],gCparamLT->param.mat[2][0],gCparamLT->param.mat[2][1],gCparamLT->param.mat[2][2]);
        //cout<<gCparamLT->param.xsize<<endl;
        //cout<<gCparamLT->param.ysize<<endl;
        glutCreateWindow(argv[0]);
    }
    
    // Setup ARgsub_lite library for current OpenGL context.
    if ((gArglSettings = arglSetupForCurrentContext(&(gCparamLT->param), arVideoGetPixelFormat())) == NULL) {
        ARLOGe("main(): arglSetupForCurrentContext() returned error.\n");
        cleanup();
        exit(-1);
    }
    arUtilTimerReset();
    
    // Start the video.
    if (arVideoCapStart() != 0) {
        ARLOGe("setupCamera(): Unable to begin camera data capture.\n");
        return (FALSE);
    }
    
    // Here we start out detect thread.
    //创建线程对象detector,绑定线程函数为detector
    std::thread detector(detect, 1, 2, 3);
    sleep(1);
    {
        std::lock_guard<std::mutex> lk(mutex_detect);
        canDetect = true;
    }
    cond_detect.notify_one();
    //输出t1的线程ID
    std::cout << "ID:" << detector.get_id() << std::endl;
    //等待t1线程函数执行结束
    detector.detach();
    
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

static void cleanup(void)
{
    // NFT cleanup.
    ARLOGd("Cleaning up ARToolKit NFT handles.\n");
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
static void Display(void)
{
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
    
    mutex_targetsList.lock();
    //cout<<"targetsList.size():"<<targetsList.size()<<endl;
    //gluLookAt(0,0,0,0,0,1,0,-1,0);
    for(auto target:targetsList){
        glLoadMatrixd(target->pose.T);
//        cout<<target->pose.T[0]<<","<<target->pose.T[1]<<","<<target->pose.T[2]<<":"<<target->pose.T[3]<<endl;
//        cout<<target->pose.T[4]<<","<<target->pose.T[5]<<","<<target->pose.T[6]<<":"<<target->pose.T[7]<<endl;
//        cout<<target->pose.T[8]<<","<<target->pose.T[9]<<","<<target->pose.T[10]<<":"<<target->pose.T[11]<<endl;
//        cout<<target->pose.T[12]<<","<<target->pose.T[13]<<","<<target->pose.T[14]<<":"<<target->pose.T[15]<<endl;
        DrawCube();
    }
    mutex_targetsList.unlock();
    
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
