#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#define __USE_XOPEN2K
#include <stdlib.h>

#include "tensorflow/c/c_api.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "tfe.h"

#define MAX_DETECTIONS 100

// assuming out of memory for failed tf allocations
#define TFE_EALLOC ENOMEM

#define TFE_set_default_tf_status(s) TF_SetStatus(s, TF_UNKNOWN, "")

#define TFE_set_default_errno() if(errno == 0) errno = TFE_EALLOC

#define TFE_set_errno(val) if(errno == 0) errno = val

#define TFE_FAIL 2

/* not thread safe */
#define TFE_printf_errno(hd) printf("%s: %s\n", hd, strerror(errno))

#define TFE_printf_status(hd, status)                       \
    {   if(TF_GetCode(status) == TF_UNKNOWN) {                \
            TFE_printf_errno(hd);                           \
        } else {                                            \
            printf("%s: %s\n", hd, TF_Message(status));      \
        }                                                   \
    }      

TF_Tensor * 
TFE_TensorFromCVImage(const IplImage * const img)
{
    if(img == NULL) {
        errno = EINVAL;
        return NULL;
    }

    TF_Tensor * tensor;
    int64_t shape[4];

    shape[0] = 1;             
    shape[1] = img->height; 
    shape[2] = img->width;    
    shape[3] = img->nChannels; 

    if((tensor = TF_AllocateTensor(TF_UINT8, shape, 4, img->imageSize)) == NULL) {
        TFE_set_default_errno();
        return NULL;
    }

    memcpy(TF_TensorData(tensor), img->imageData, TF_TensorByteSize(tensor));

    return tensor;
}

TF_Graph * 
TFE_ImportGraph(const char * const path, TF_Status * status)
{
    int gfd;
    off_t gsize;
    char * gbuf = NULL;
    TF_Buffer * gtfbuf = NULL;
    TF_Graph * g = NULL;
    TF_ImportGraphDefOptions * grph_opts = NULL;

    errno = 0;
    TFE_set_default_tf_status(status);

    if((g = TF_NewGraph()) == NULL) {
        TFE_set_default_errno();
        goto end;
    }

    if((gfd = open(path, O_RDONLY)) <= 0) {
        goto end;
    }

    if((gsize = lseek(gfd, 0, SEEK_END)) < 0) {
        goto end;
    }

    if(lseek(gfd, 0, SEEK_SET) < 0) {
        goto end;
    }

    if((gbuf = malloc(gsize)) == NULL) {
        goto end;
    }

    if(read(gfd, gbuf, gsize) < 0) {
        goto end;
    }

    close(gfd);
    gfd = -1;

    if((gtfbuf = TF_NewBuffer()) == NULL) {
        TFE_set_default_errno();
        goto end;
    }

    gtfbuf->data = gbuf;
    gtfbuf->length = gsize;
    // we are gonna handle deallocation of data.
    gtfbuf->data_deallocator = NULL;

    if((grph_opts = TF_NewImportGraphDefOptions()) == NULL) {
        TFE_set_default_errno();
        goto end;
    }

    TF_GraphImportGraphDef(g, gtfbuf, grph_opts, status);

/* cleanup and return if graph */
end:

    if(grph_opts != NULL) TF_DeleteImportGraphDefOptions(grph_opts);
    if(gtfbuf != NULL) TF_DeleteBuffer(gtfbuf);
    if(gbuf != NULL) free(gbuf);
    if(gfd > 0) close(gfd);

    if(errno != 0) {
        if(g != NULL) TF_DeleteGraph(g);
        g = NULL;
    }

    return g;
}

/*
    get outputs for tensorflow object detection API
*/
int
TFE_OD_GetOutputs(TF_Output * in, size_t inl, TF_Output * out, size_t outl, TF_Graph * graph)
{
    if(outl != 4 || inl != 1) {
        errno = EINVAL;
        return -1;
    }

    TF_Operation * tmp;

    if((tmp = TF_GraphOperationByName(graph, "image_tensor")) == NULL) {
        errno = ENOENT;
        return -1;
    }

    in[0].index = 0;
    in[0].oper = tmp;

    if((tmp = TF_GraphOperationByName(graph, "detection_boxes")) == NULL) {
        errno = ENOENT;
        return -1;
    }

    out[0].index = 0;
    out[0].oper = tmp;

    if((tmp = TF_GraphOperationByName(graph, "detection_scores")) == NULL) {
        errno = ENOENT;
        return -1;
    }

    out[1].index = 0;
    out[1].oper = tmp;

    if((tmp = TF_GraphOperationByName(graph, "detection_classes")) == NULL) {
        errno = ENOENT;
        return -1;
    }

    out[2].index = 0;
    out[2].oper = tmp;

    if((tmp = TF_GraphOperationByName(graph, "num_detections")) == NULL) {
        errno = ENOENT;
        return -1;
    }

    out[3].index = 0;
    out[3].oper = tmp;

    return 0;
}

int
TFE_OD_GetTensors(TF_Tensor ** in, size_t inl, TF_Tensor ** out, size_t outl, IplImage * img, int maxDim)
{
    if(inl != 1 || outl != 4) {
        errno = EINVAL;
        return -1;
    }

    if((in[0] = TF_AllocateTensor(
            TF_UINT8, 
            (int64_t[]){1, img->height, img->width, img->nChannels},
            4, img->imageSize)) == NULL) {
        TFE_set_default_errno();
        return -1;
    }

    memcpy(TF_TensorData(in[0]), img->imageData, img->imageSize);

    if((out[0] = TF_AllocateTensor(
            TF_FLOAT,
            (int64_t[]){1, MAX_DETECTIONS, 4},
            3, sizeof(float)*MAX_DETECTIONS*4)) == NULL){
        TFE_set_default_errno();
        return -1;
    }

    if((out[1] = TF_AllocateTensor(
            TF_FLOAT,
            (int64_t[]){1, MAX_DETECTIONS},
            2, sizeof(float)*MAX_DETECTIONS)) == NULL){
        TFE_set_default_errno();
        return -1;
    }

    if((out[2] = TF_AllocateTensor(
            TF_FLOAT,
            (int64_t[]){1, MAX_DETECTIONS},
            2, sizeof(float)*MAX_DETECTIONS)) == NULL){
        TFE_set_default_errno();
        return -1;
    }

    if((out[3] = TF_AllocateTensor(
            TF_FLOAT,
            (int64_t[]){1, 1},
            2, sizeof(float))) == NULL){
        TFE_set_default_errno();
        return -1;
    }

    return 0;
}

void
TFE_DisplayBboxesOnImage(IplImage * img, float * bbox, float * scores, float * classes, size_t num)
{
    const float (* bbox_2d)[4] = (const float (*)[4])bbox;
    size_t i = 0;
    CvFont font;
    CvPoint lp, rp;
    CvScalar color;
    const int textl = 20;
    char text[textl];

    cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.2, 1.2, 0, 2, 0);
    
    for(i = 0; i < num; i++) {
        if(scores[i] < 0.7) {
            continue;
        }

        lp.x = bbox_2d[i][1]*img->width;
        lp.y = bbox_2d[i][0]*img->height;
        rp.x = bbox_2d[i][3]*img->width;
        rp.y = bbox_2d[i][2]*img->height;
        color = cvScalar(0, 0, 0, 0);

        cvRectangle(img, lp, rp, color, 2, 0, 0);
        
        lp.y-=4;
        snprintf(text, textl, "%.2f%% | %d", scores[i]*100, (int)classes[i]);

        cvPutText(img, text, lp, &font, color);
    }

    cvShowImage("Inference result.", img);

    cvWaitKey(-1);
}

int
TFE_OD_Run()
{
    TF_Status * status = NULL;
    TF_Graph * graph = NULL;
    TF_Session * sess = NULL;
    TF_SessionOptions * sess_opt = NULL;
    IplImage * img = NULL;
    TF_Output out_outputs[4], in_outputs[1];
    TF_Tensor * out_values[4], * in_values[1];
    size_t i = 0;

    for(i = 0; i < 4; i++) out_values[i] = NULL;
    for(i = 0; i < 1; i++) in_values[i] = NULL;

    setenv("TF_CPP_MIN_LOG_LEVEL", "3", 1);

    if((status = TF_NewStatus()) == NULL) {
        TFE_set_default_errno();
        TFE_printf_errno("Create status");
        goto end;
    }

    if((graph = TFE_ImportGraph("frozen_inference_graph.pb", status)) == NULL || 
            TF_GetCode(status) != TF_OK) {
        TFE_printf_status("Import graph", status);
        goto end;
    }

    if((sess_opt = TF_NewSessionOptions()) == NULL) {
        TFE_set_default_errno();
        TFE_printf_errno("Create session options");
        goto end;
    }

    if((sess = TF_NewSession(graph, sess_opt, status)) == NULL || 
            TF_GetCode(status) != TF_OK) {
        TFE_set_default_errno();
        TFE_printf_status("Import graph", status);
        goto end;
    }

    TF_DeleteSessionOptions(sess_opt);
    sess_opt = NULL;

    if((img = cvLoadImage("1.jpg", CV_LOAD_IMAGE_COLOR)) == NULL) {
        TFE_set_default_errno();
        TFE_printf_errno("Load image");
        goto end;
    }

    if(TFE_OD_GetOutputs(in_outputs, 1, out_outputs, 4, graph) != 0) {
        TFE_printf_errno("Get outputs");
        goto end;
    }

    if(TFE_OD_GetTensors(in_values, 1, out_values, 4, img, 100) != 0) {
        TFE_printf_errno("Get tensors");
        goto end;
    }

    TF_SessionRun(sess, NULL, 
                in_outputs,  in_values, 1, 
                out_outputs, out_values, 4, 
                NULL, 0, NULL, status);

    if(TF_GetCode(status) != TF_OK) {
        TFE_printf_status("Session run", status);
        TFE_set_errno(EINVAL);
        goto end;
    }

    TFE_DisplayBboxesOnImage(
        img,
        TF_TensorData(out_values[0]),
        TF_TensorData(out_values[1]),
        TF_TensorData(out_values[2]),
        (size_t)((float*)TF_TensorData(out_values[3]))[0]);

    TF_SessionRun(sess, NULL, 
                in_outputs,  in_values, 1, 
                out_outputs, out_values, 4, 
                NULL, 0, NULL, status);

    if(TF_GetCode(status) != TF_OK) {
        TFE_printf_status("Session run", status);
        TFE_set_errno(EINVAL);
        goto end;
    }

end:
    if(graph != NULL) TF_DeleteGraph(graph);
    if(sess_opt != NULL) TF_DeleteSessionOptions(sess_opt);
    if(sess != NULL) TF_DeleteSession(sess, status);
    // might wanna handle status here
    if(status != NULL) TF_DeleteStatus(status);
    if(img != NULL) cvReleaseImage(&img);
    for(i = 0; i < 4; i++)
        if(out_values[i] != NULL) TF_DeleteTensor(out_values[i]);
    for(i = 0; i < 1; i++)
        if(in_values[i] != NULL) TF_DeleteTensor(in_values[i]);

    if(errno == 0) {
        return 0;
    } else {
        return -1;
    }
}
