#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

#include "opencv/highgui.h"
#include "opencv/cv.h"
#include "tfe.h"

/* assuming out of memory for failed tf allocations */
#define TFE_EALLOC ENOMEM

#define BBOX_LABEL_LEN 20

#define TFE_set_ok_status(status) TF_SetStatus(status, TF_OK, "")

#define TFE_set_default_tf_status(s, hdr) TF_SetStatus(s, TF_UNKNOWN, hdr)

#define TFE_set_status_from_errno(status) \
    TFE_set_default_tf_status(status, strerror(errno))

#define TFE_set_default_errno() if(errno == 0) errno = TFE_EALLOC

#define TFE_set_errno(val) if(errno == 0) errno = val

#define TFE_FAIL 2

TF_Tensor * 
TFE_TensorFromCVImage(const IplImage * const img, TF_Status * status)
{
    TF_Tensor * tensor;
    int64_t shape[4];

    tensor = NULL;
    errno = 0;
    TFE_set_ok_status(status);

    if(img == NULL) {
        errno = EINVAL;
        TFE_set_status_from_errno(status);
        return NULL;
    }

    shape[0] = 1;             
    shape[1] = img->height; 
    shape[2] = img->width;    
    shape[3] = img->nChannels; 

    if((tensor = TF_AllocateTensor(TF_UINT8, shape, 4, img->imageSize)) == NULL) {
        TFE_set_default_errno();
        TFE_set_status_from_errno(status);
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
    char * gbuf;
    TF_Buffer * gtfbuf;
    TF_Graph * g;
    TF_ImportGraphDefOptions * grph_opts;

    errno = 0;
    gbuf = NULL;
    gtfbuf = NULL;
    g = NULL;
    grph_opts = NULL;
    TFE_set_ok_status(status);

    if((g = TF_NewGraph()) == NULL) {
        TFE_set_default_errno();
        TFE_set_status_from_errno(status);
        goto end;
    }

    if((gfd = open(path, O_RDONLY)) <= 0) {
        TFE_set_status_from_errno(status);
        goto end;
    }

    if((gsize = lseek(gfd, 0, SEEK_END)) < 0) {
        TFE_set_status_from_errno(status);
        goto end;
    }

    if(lseek(gfd, 0, SEEK_SET) < 0) {
        TFE_set_status_from_errno(status);
        goto end;
    }

    if((gbuf = malloc(gsize)) == NULL) {
        TFE_set_status_from_errno(status);
        goto end;
    }

    if(read(gfd, gbuf, gsize) < 0) {
        TFE_set_status_from_errno(status);
        goto end;
    }

    close(gfd);
    gfd = -1;

    if((gtfbuf = TF_NewBuffer()) == NULL) {
        TFE_set_default_errno();
        TFE_set_status_from_errno(status);
        goto end;
    }

    gtfbuf->data = gbuf;
    gtfbuf->length = gsize;
    /* we are gonna handle deallocation of data. */
    gtfbuf->data_deallocator = NULL;

    if((grph_opts = TF_NewImportGraphDefOptions()) == NULL) {
        TFE_set_default_errno();
        TFE_set_status_from_errno(status);
        goto end;
    }

    TF_GraphImportGraphDef(g, gtfbuf, grph_opts, status);

/* cleanup and return if graph */
end:

    if(grph_opts != NULL) TF_DeleteImportGraphDefOptions(grph_opts);
    if(gtfbuf != NULL) TF_DeleteBuffer(gtfbuf);
    if(gbuf != NULL) free(gbuf);
    if(gfd > 0) close(gfd);

    if(TF_GetCode(status) != TF_OK) {
        if(g != NULL) TF_DeleteGraph(g);
        g = NULL;
    }

    return g;
}

/*
    get outputs for tensorflow object detection API
*/
int
TFE_OD_GetOutputs(
    TF_Output * in, size_t inl, TF_Output * out, size_t outl, TF_Graph * graph)
{
    TF_Operation * tmp;

    if(outl != 4 || inl != 1) {
        errno = EINVAL;
        return -1;
    }

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
TFE_OD_GetTensors(TF_Tensor ** in, size_t inl, TF_Tensor ** out, size_t outl, IplImage * img, int maxdet)
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
            (int64_t[]){1, maxdet, 4},
            3, sizeof(float)*maxdet*4)) == NULL){
        TFE_set_default_errno();
        return -1;
    }

    if((out[1] = TF_AllocateTensor(
            TF_FLOAT,
            (int64_t[]){1, maxdet},
            2, sizeof(float)*maxdet)) == NULL){
        TFE_set_default_errno();
        return -1;
    }

    if((out[2] = TF_AllocateTensor(
            TF_FLOAT,
            (int64_t[]){1, maxdet},
            2, sizeof(float)*maxdet)) == NULL){
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
TFE_OD_DisplayBboxesOnImage(
    IplImage * img, float * bbox, float * scores, float * classes, size_t num, int wait)
{
    const float (* bbox_2d)[4] = (const float (*)[4])bbox;
    size_t i = 0;
    CvFont font;
    CvPoint lp, rp;
    CvScalar color;
    char text[BBOX_LABEL_LEN];

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
        snprintf(text, BBOX_LABEL_LEN, "%.2f%% | %d", scores[i]*100, (int)classes[i]);

        cvPutText(img, text, lp, &font, color);
    }

    cvShowImage("Inference", img);

    if(wait) {
        cvWaitKey(-1);
    } else {
        cvWaitKey(1);
    }
}


TF_Session *
TFE_OD_CreateSession(TF_Graph * graph, TF_Status * status)
{
    TF_Session * session = NULL;
    TF_SessionOptions * sess_opt = NULL;
    
    if((sess_opt = TF_NewSessionOptions()) == NULL) {
        TFE_set_default_tf_status(status, "Creating session options");
        TFE_set_default_errno();
        return NULL;
    }

    if((session = TF_NewSession(graph, sess_opt, status)) == NULL || TF_GetCode(status) != TF_OK) {
        errno = EINVAL;
    }

    TF_DeleteSessionOptions(sess_opt);

    if(TFE_ptr_err(session, status)) return NULL;

    return session;
}

/*
    a lot of variables here such as outputs, tensors could be reused here instead of redefined.
    todo: refractor
*/
void
TFE_OD_RunInferenceWithDisplay(
    TF_Session * sess, TF_Graph * graph, TF_Status * status, const char * imgpath, int maxDetections)
{
    IplImage * img = NULL;
    TF_Output out_outputs[4], in_outputs[1];
    TF_Tensor * out_values[4], * in_values[1];
    size_t i = 0;

    TFE_set_ok_status(status);
    errno = 0;

    for(i = 0; i < 4; i++) out_values[i] = NULL;
    for(i = 0; i < 1; i++) in_values[i] = NULL;

    if((img = cvLoadImage(imgpath, CV_LOAD_IMAGE_COLOR)) == NULL) {
        TFE_set_errno(EINVAL);
        TFE_set_default_tf_status(status, "Couldnt load image");
        goto end;
    }

    if(TFE_OD_GetOutputs(in_outputs, 1, out_outputs, 4, graph) != 0) {
        TFE_set_default_errno();
        TFE_set_default_tf_status(status, "Couldnt get outputs");
        goto end;
    }

    if(TFE_OD_GetTensors(in_values, 1, out_values, 4, img, maxDetections) != 0) {
        TFE_set_default_tf_status(status, "Couldnt get status");
        goto end;
    }

    TF_SessionRun(sess, NULL, 
                in_outputs,  in_values, 1, 
                out_outputs, out_values, 4, 
                NULL, 0, NULL, status);

    if(TFE_err(status)) {
        TFE_set_errno(EINVAL);
        goto end;
    }

    TFE_OD_DisplayBboxesOnImage(
        img,
        TF_TensorData(out_values[0]),
        TF_TensorData(out_values[1]),
        TF_TensorData(out_values[2]),
        (size_t)((float*)TF_TensorData(out_values[3]))[0], 
        1);

    if(TFE_err(status)) {
        TFE_set_errno(EINVAL);
        goto end;
    }

end:
    if(img != NULL) cvReleaseImage(&img);
    for(i = 0; i < 4; i++)
        if(out_values[i] != NULL) TF_DeleteTensor(out_values[i]);
    for(i = 0; i < 1; i++)
        if(in_values[i] != NULL) TF_DeleteTensor(in_values[i]);
}

void
TFE_OD_RunInferenceWithCamera(
    TF_Session * sess, TF_Graph * graph, TF_Status * status, int maxDetections)
{
    IplImage * img, * clone;
    TF_Output out_outputs[4], in_outputs[1];
    TF_Tensor * out_values[4], * in_values[1];
    size_t i;
    int isInit, psize;
    CvCapture * cap;

    TFE_set_ok_status(status);
    errno = 0;
    i = 0;
    isInit = 0;
    img = NULL;
    clone = NULL;
    for(i = 0; i < 4; i++) out_values[i] = NULL;
    for(i = 0; i < 1; i++) in_values[i] = NULL;

    if((cap = cvCreateCameraCapture(0)) == NULL) {
        TFE_set_errno(EINVAL);
        TFE_set_default_tf_status(status, "Couldnt open camera capture");
        goto end;
    }
    
    cvNamedWindow("Inference", 1);

    for(;;) {
        // if(cvGrabFrame(cap) != 1) {
        //     printf("Couldnt grab a frame");
        //     sleep(1);
        //     continue;
        // }

        // img = cvRetrieveFrame(cap, 0);
        // if(img == NULL) {
        //     printf("Couldnt retrieve frame");
        //     sleep(1);
        //     continue;
        // }

        if((img = cvQueryFrame(cap)) == NULL) {
            printf("Couldnt query frame");
            sleep(1);
            continue;
        }

        if(isInit && (psize != img->imageSize)) {
            isInit = 0;
        }

        if(!isInit) {
            for(i = 0; i < 4; i++)
                if(out_values[i] != NULL) TF_DeleteTensor(out_values[i]);
            for(i = 0; i < 1; i++)
                if(in_values[i] != NULL) TF_DeleteTensor(in_values[i]);
            if(TFE_OD_GetOutputs(in_outputs, 1, out_outputs, 4, graph) != 0) {
                TFE_set_default_errno();
                TFE_set_default_tf_status(status, "Couldnt get outputs");
                goto end;
            }
            if(TFE_OD_GetTensors(in_values, 1, out_values, 4, img, maxDetections) != 0) {
                TFE_set_default_tf_status(status, "Couldnt get status");
                goto end;
            }
            isInit = 1;
            psize = img->imageSize;
            if(clone != NULL) {
                cvReleaseImage(&clone);
            }
            clone = cvCreateImage(cvSize(img->width, img->height), img->depth ,3);
        } else {
            memcpy(TF_TensorData(in_values[0]), img->imageData, img->imageSize);
        }

        TF_SessionRun(sess, NULL, 
                in_outputs,  in_values, 1, 
                out_outputs, out_values, 4, 
                NULL, 0, NULL, status);

        if(TFE_err(status)) {
            goto end;
        }

        cvCopyImage(img, clone);

        TFE_OD_DisplayBboxesOnImage(
            clone,
            TF_TensorData(out_values[0]),
            TF_TensorData(out_values[1]),
            TF_TensorData(out_values[2]),
            (size_t)((float*)TF_TensorData(out_values[3]))[0], 
            0);
    } 

    if(TFE_err(status)) {
        TFE_set_errno(EINVAL);
        goto end;
    }

end:
    for(i = 0; i < 4; i++)
        if(out_values[i] != NULL) TF_DeleteTensor(out_values[i]);
    for(i = 0; i < 1; i++)
        if(in_values[i] != NULL) TF_DeleteTensor(in_values[i]);
}
