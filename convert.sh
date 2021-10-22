#! /bin/sh
paddlex --export_inference --model_dir=output/mobilenetv3/best_model --save_dir=inference_model
hub convert --model_dir inference_model \
              --module_name DriverStatusRecognition \
              --module_version 1.0.0 \
              --output_dir outputs