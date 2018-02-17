#include "ARMarkerNFT.h"

#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#  include <windows.h>
#  define MAXPATHLEN MAX_PATH
#else
#  include <sys/param.h> // MAXPATHLEN
#endif


const ARPose ARPoseUnity = {{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}};

static char *get_buff(char *buf, int n, FILE *fp, int skipblanks)
{
    char *ret;
	size_t l;
    
    do {
        ret = fgets(buf, n, fp);
        if (ret == NULL) return (NULL); // EOF or error.
        
        // Remove NLs and CRs from end of string.
        l = strlen(buf);
        while (l > 0) {
            if (buf[l - 1] != '\n' && buf[l - 1] != '\r') break;
            l--;
            buf[l] = '\0';
        }
    } while (buf[0] == '#' || (skipblanks && buf[0] == '\0')); // Reject comments and blank lines.
    
    return (ret);
}

void newMarkers(const char *markersConfigDataFilePathC, ARMarkerNFT **markersNFT_out, int *markersNFTCount_out)
{
    FILE          *fp;
    char           buf[MAXPATHLEN], buf1[MAXPATHLEN];
    int            tempI;
    ARMarkerNFT   *markersNFT;
    int            markersNFTCount;
    ARdouble       tempF;
    int            i;
    char           markersConfigDataDirC[MAXPATHLEN];
    size_t         markersConfigDataDirCLen;

    if (!markersConfigDataFilePathC || markersConfigDataFilePathC[0] == '\0' || !markersNFT_out || !markersNFTCount_out) return;
        
    // Load the marker data file.
    ARLOGd("Opening marker config. data file from path '%s'.\n", markersConfigDataFilePathC);
    arUtilGetDirectoryNameFromPath(markersConfigDataDirC, markersConfigDataFilePathC, MAXPATHLEN, 1); // 1 = add '/' at end.
    markersConfigDataDirCLen = strlen(markersConfigDataDirC);
    if ((fp = fopen(markersConfigDataFilePathC, "r")) == NULL) {
        ARLOGe("Error: unable to locate marker config data file '%s'.\n", markersConfigDataFilePathC);
        return;
    }
    
    // First line is number of markers to read.
    get_buff(buf, MAXPATHLEN, fp, 1);
    if (sscanf(buf, "%d", &tempI) != 1 ) {
        ARLOGe("Error in marker configuration data file; expected marker count.\n");
        fclose(fp);
        return;
    }
    
    arMallocClear(markersNFT, ARMarkerNFT, tempI);
    markersNFTCount = tempI;
    
    ARLOGd("Reading %d marker configuration(s).\n", markersNFTCount);

    for (i = 0; i < markersNFTCount; i++) {
        
        // Read marker name.
        if (!get_buff(buf, MAXPATHLEN, fp, 1)) {
            ARLOGe("Error in marker configuration data file; expected marker name.\n");
            break;
        }
        
        // Read marker type.
        if (!get_buff(buf1, MAXPATHLEN, fp, 1)) {
            ARLOGe("Error in marker configuration data file; expected marker type.\n");
            break;
        }
        
        // Interpret marker type, and read more data.
        if (strcmp(buf1, "SINGLE") == 0) {
            ARLOGe("Error in marker configuration data file; SINGLE markers not supported in this build.\n");
        } else if (strcmp(buf1, "MULTI") == 0) {
            ARLOGe("Error in marker configuration data file; MULTI markers not supported in this build.\n");
        } else if (strcmp(buf1, "NFT") == 0) {
            markersNFT[i].valid = markersNFT[i].validPrev = FALSE;
            arMalloc(markersNFT[i].datasetPathname, char, markersConfigDataDirCLen + strlen(buf) + 1);
            strcpy(markersNFT[i].datasetPathname, markersConfigDataDirC);
            strcpy(markersNFT[i].datasetPathname + markersConfigDataDirCLen, buf);
            markersNFT[i].pageNo = -1;
        } else {
            ARLOGe("Error in marker configuration data file; unsupported marker type %s.\n", buf1);
        }
        
        // Look for optional tokens. A blank line marks end of options.
        while (get_buff(buf, MAXPATHLEN, fp, 0) && (buf[0] != '\0')) {
            if (strncmp(buf, "FILTER", 6) == 0) {
                markersNFT[i].filterCutoffFrequency = AR_FILTER_TRANS_MAT_CUTOFF_FREQ_DEFAULT;
                markersNFT[i].filterSampleRate = AR_FILTER_TRANS_MAT_SAMPLE_RATE_DEFAULT;
                if (strlen(buf) != 6) {
                    if (sscanf(&buf[6],
#ifdef ARDOUBLE_IS_FLOAT
                               "%f"
#else
                               "%lf"
#endif
                               , &tempF) == 1) markersNFT[i].filterCutoffFrequency = tempF;
                }
                markersNFT[i].ftmi = arFilterTransMatInit(markersNFT[i].filterSampleRate, markersNFT[i].filterCutoffFrequency);
            }
            // Unknown tokens are ignored.
        }
    }
    fclose(fp);
    
    // If not all markers were read, an error occurred.
    if (i < markersNFTCount) {
    
        // Clean up.
        for (; i >= 0; i--) {
            if (markersNFT[i].datasetPathname)  free(markersNFT[i].datasetPathname);
            if (markersNFT[i].ftmi) arFilterTransMatFinal(markersNFT[i].ftmi);
        }
        free(markersNFT);
                
        *markersNFTCount_out = 0;
        *markersNFT_out = NULL;
        return;
    }
    
    *markersNFTCount_out = markersNFTCount;
    *markersNFT_out = markersNFT;
}

void deleteMarkers(ARMarkerNFT **markersNFT_p, int *markersNFTCount_p)
{
    int i;
    
    if (!markersNFT_p || !*markersNFT_p || !*markersNFTCount_p || *markersNFTCount_p < 1) return;
    
    for (i = 0; i < *markersNFTCount_p; i++) {
        if ((*markersNFT_p)[i].datasetPathname) {
            free((*markersNFT_p)[i].datasetPathname);
            (*markersNFT_p)[i].datasetPathname = NULL;
        }
        if ((*markersNFT_p)[i].ftmi) {
            arFilterTransMatFinal((*markersNFT_p)[i].ftmi);
            (*markersNFT_p)[i].ftmi = NULL;
        }
    }
    free(*markersNFT_p);
    *markersNFT_p = NULL;
    *markersNFTCount_p = 0;
}
