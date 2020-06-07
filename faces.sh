set -e

cd data

wget http://download.tensorflow.org/models/object_detection/facessd_mobilenet_v2_quantized_320x320_open_image_v4.tar.gz


tar xf facessd_mobilenet_v2_quantized_320x320_open_image_v4.tar.gz

mkdir facessd_mobilenet_v2_quantized_320x320_open_image_v4_out

python3 ${TFPATH}/object_detection/export_inference_graph.py \
    --write_inference_graph=True \
    --output_directory=./facessd_mobilenet_v2_quantized_320x320_open_image_v4_out \
    --trained_checkpoint_prefix=facessd_mobilenet_v2_quantized_320x320_open_image_v4/model.ckpt \
    --pipeline_config_path=facessd_mobilenet_v2_quantized_320x320_open_image_v4/pipeline.config
