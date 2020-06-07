#if !defined(TFE_H)
#define TFE_H 1

#include <errno.h>
#include <string.h>
#define __USE_XOPEN2K
#include <stdlib.h>

#include "tensorflow/c/c_api.h"
#include "opencv/cv.h"

/* not thread safe */
#define TFE_printf_errno(hd) printf("%s: %s\n", hd, strerror(errno))

#define TFE_err(status) (TF_GetCode((status)) != TF_OK)

#define TFE_ptr_err(ptr, status) ((ptr) == NULL || TF_GetCode((status)) != TF_OK)

#define TFE_disable_tf_log() setenv("TF_CPP_MIN_LOG_LEVEL", "3", 1)

#define TFE_printf_status(hd, status)                                               \
    {   if(TF_GetCode(status) != TF_OK) {                                       \
            printf("%s: %s (errno=%s)\n", hd, TF_Message(status), strerror(errno));        \
        } else if(errno != 0) {                                                      \
            printf("%s: %s\n", hd, TF_Message(status));                                 \
        } else {                                                                       \
            printf("%s: Success\n", hd);                                                    \
        }                                                                           \
    }      

TF_Session *
TFE_OD_CreateSession(TF_Graph * graph, TF_Status * status);

TF_Tensor * 
    TFE_TensorFromCVImage(const IplImage * const img, TF_Status * status);

TF_Graph * TFE_ImportGraph(const char * const path, TF_Status * status);

int
TFE_OD_GetOutputs(  TF_Output * in, 
                    size_t inl, 
                    TF_Output * out, 
                    size_t outl, 
                    TF_Graph * graph);

int 
TFE_OD_GetTensors(  TF_Tensor ** in, 
                    size_t inl, 
                    TF_Tensor ** out, 
                    size_t outl, 
                    IplImage * img, 
                    int maxdet);

void 
TFE_OD_DisplayBboxesOnImage(IplImage * img, 
                            float * bbox, 
                            float * scores, 
                            float * classes, 
                            size_t num,
                            int wait);

void
TFE_OD_RunInferenceWithDisplay( TF_Session * sess, 
                                TF_Graph * graph, 
                                TF_Status * status, 
                                const char * imgpath, 
                                int maxDetections);

void
TFE_OD_RunInferenceWithCamera(  TF_Session * sess, 
                                TF_Graph * graph, 
                                TF_Status * status, 
                                int maxDetections);

#endif