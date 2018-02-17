#ifndef __ARMarkerNFT_h__
#define __ARMarkerNFT_h__

#include <AR/ar.h>
#include <AR/arFilterTransMat.h>
#ifndef _MSC_VER
#  include <stdbool.h>
#else
typedef unsigned char bool;
#endif

#ifdef __cplusplus
extern "C" {
#endif
    
    typedef struct {
        ARdouble v[3];
    } ARVec3;
    
    typedef struct {
        ARdouble T[16]; // Position and orientation, column-major order. (position(x,y,z) = {T[12], T[13], T[14]}
    } ARPose;
    
    extern const ARPose ARPoseUnity;
    
    typedef struct _ARMarkerNFT {
        // ARMarker protected
        bool       valid;
        bool       validPrev;
        ARdouble   trans[3][4];
        ARPose     pose;
        ARdouble   marker_width;
        ARdouble   marker_height;
        // ARMarker private
        ARFilterTransMatInfo *ftmi;
        ARdouble   filterCutoffFrequency;
        ARdouble   filterSampleRate;
        // ARMarkerNFT
        int        pageNo;
        char      *datasetPathname;
    } ARMarkerNFT;
    
    void newMarkers(const char *markersConfigDataFilePathC, ARMarkerNFT **markersNFT_out, int *markersNFTCount_out);
    
    void deleteMarkers(ARMarkerNFT **markersNFT_p, int *markersNFTCount_p);
    
#ifdef __cplusplus
}
#endif
#endif // !__ARMarkerNFT_h__
