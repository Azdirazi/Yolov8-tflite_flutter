import 'dart:io';

import 'package:yolo_flutter/yolo.dart';


class ModelSetup{
  static const inModelWidth = 640;
  static const inModelHeight = 640;

  static const numClasses1 = 2;
  static const numClasses2 = 1;
  static const numClasses3 = 1;
  static const numClasses4 = 1;
  static const numClasses5 = 1;
  static const numClasses6 = 1;


  static const double maxImageWidgetHeight = 400;

  final YoloModel model = YoloModel(
    'assets/models/glasses-sunglasses_float16.tflite',
    inModelWidth,
    inModelHeight,
    numClasses1,
  );
  final YoloModel model2 = YoloModel(
    'assets/models/hat_float16.tflite',
    inModelWidth,
    inModelHeight,
    numClasses2,
  );
  final YoloModel model3 = YoloModel(
    'assets/models/cap_float16.tflite',
    inModelWidth,
    inModelHeight,
    numClasses3,
  );
  final YoloModel model4 = YoloModel(
    'assets/models/headband_float16.tflite',
    inModelWidth,
    inModelHeight,
    numClasses4,
  );
  final YoloModel model5 = YoloModel(
    'assets/models/scarf_float16.tflite',
    inModelWidth,
    inModelHeight,
    numClasses5,
  );
  final YoloModel model6 = YoloModel(
    'assets/models/watch_float16.tflite',
    inModelWidth,
    inModelHeight,
    numClasses6,
  );

  File? imageFile;

  double confidenceThreshold = 0.4;
  double iouThreshold = 0.1;
  bool agnosticNMS = false;



  int? imageWidth;
  int? imageHeight;

  // Data lists for various outputs as declared
  List<List<double>>? inferenceOutput, inferenceOutputHat, inferenceOutputCap, inferenceOutputHeadband, inferenceOutputScarf, inferenceOutputWatch;
  List<int> classes = [], classesHat = [], classesCap = [], classesHeadband = [], classesScarf = [], classesWatch = [];
  List<List<double>> bboxes = [], bboxesHat = [], bboxesCap = [], bboxesHeadband = [], bboxesScarf = [], bboxesWatch = [];
  List<double> scores = [], scoresHat = [], scoresCap = [], scoresHeadband = [], scoresScarf = [], scoresWatch = [];

  void initModels() {
    model.init();
    model2.init();
    model3.init();
    model4.init();
    model5.init();
    model6.init();
    // Initialize additional models as needed
  }
}