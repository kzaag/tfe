#include <stdio.h>

#include "tfe.h"
#include<getopt.h>

/*ISO C90*/
void set_args(int, char **, char **, char **);

void
set_args(int argc, char ** argv, char ** graph_path, char ** img_path)
{
    char c;
    size_t tmpl;

    while ((c = getopt(argc, argv, "i:g:")) != -1) {
        tmpl = strlen(optarg);
        if(tmpl <= 0) break;
        switch(c){
        case 'g':
            if((*graph_path = malloc(tmpl+1)) == NULL) break;
            strcpy(*graph_path, optarg);
            break;
        case 'i':
            if((*img_path = malloc(tmpl+1)) == NULL) break;
            strcpy(*img_path, optarg);
            break;
        }
    }
}

int
main(int argc, char ** argv) 
{
    TF_Status * status = NULL;
    TF_Graph * graph = NULL;
    TF_Session * session = NULL;
    char * graph_path = NULL, * img_path = NULL;

    set_args(argc, argv, &graph_path, &img_path);

    if(graph_path == NULL || img_path == NULL) {
        printf("Usage: ./tfe -i [img_path] -g [fronzen inference graph path]\n");
        if(errno == 0) errno = EINVAL;
        goto end;
    }

    if((status = TF_NewStatus()) == NULL) {
        printf("Could allocate status");
        goto end;
    }

    TFE_disable_tf_log();

    graph = TFE_ImportGraph(graph_path, status);
    if(TFE_ptr_err(graph, status)) {
        TFE_printf_status("Create graph", status);
        goto end;
    }

    session = TFE_OD_CreateSession(graph, status);
    if(TFE_ptr_err(graph, status)) {
        TFE_printf_status("Create session", status);
        goto end;
    }

    //TFE_OD_RunInferenceWithDisplay(session, graph, status, img_path, 100);

    TFE_OD_RunInferenceWithCamera(session, graph, status, 100);

    if(TFE_err(status)) {
        TFE_printf_status("Inference", status);
        goto end;
    }

end:
    if(graph_path != NULL) free(graph_path);
    if(img_path != NULL) free(img_path);
    if(session != NULL) TF_DeleteSession(session, status);
    if(graph != NULL) TF_DeleteGraph(graph);
    if(status != NULL) TF_DeleteStatus(status);
    
    if(errno != 0) exit(1);
    exit(0);
}