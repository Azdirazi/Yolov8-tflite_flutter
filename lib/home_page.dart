import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:yolo_flutter/bbox.dart';
import 'package:yolo_flutter/labels.dart';
import 'package:yolo_flutter/yolo.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
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

  List<List<double>>? inferenceOutput,
      inferenceOutputHat,
      inferenceOutputCap,
      inferenceOutputHeadband,
      inferenceOutputScarf,
      inferenceOutputWatch;
  List<int> classes = [],
      classesHat = [],
      classesCap = [],
      classesHeadband = [],
      classesScarf = [],
      classesWatch = [];
  List<List<double>> bboxes = [],
      bboxesHat = [],
      bboxesCap = [],
      bboxesHeadband = [],
      bboxesScarf = [],
      bboxesWatch = [];
  List<double> scores = [],
      scoresHat = [],
      scoresCap = [],
      scoresHeadband = [],
      scoresScarf = [],
      scoresWatch = [];

  int? imageWidth;
  int? imageHeight;

  @override
  void initState() {
    super.initState();
    model.init();
    model2.init();
    model3.init();
    model4.init();
    model5.init();
    model6.init();
  }

  @override
  Widget build(BuildContext context) {
    final bboxesColors = List<Color>.generate(
      numClasses1,
      (_) => Color((Random().nextDouble() * 0xFFFFFF).toInt()).withOpacity(1.0),
    );

    final bboxesColorsHat = List<Color>.generate(
      numClasses2,
      (_) => Color((Random().nextDouble() * 0xFFFFFF).toInt()).withOpacity(1.0),
    );

    final bboxesColorsCap = List<Color>.generate(
      numClasses3,
      (_) => Color((Random().nextDouble() * 0xFFFFFF).toInt()).withOpacity(1.0),
    );

    final bboxesColorsHeadband = List<Color>.generate(
      numClasses4,
      (_) => Color((Random().nextDouble() * 0xFFFFFF).toInt()).withOpacity(1.0),
    );

    final bboxesColorsScarf = List<Color>.generate(
      numClasses5,
      (_) => Color((Random().nextDouble() * 0xFFFFFF).toInt()).withOpacity(1.0),
    );

    final bboxesColorsWatch = List<Color>.generate(
      numClasses6,
      (_) => Color((Random().nextDouble() * 0xFFFFFF).toInt()).withOpacity(1.0),
    );

    final ImagePicker picker = ImagePicker();

    final double displayWidth = MediaQuery.of(context).size.width;

    const textPadding = EdgeInsets.symmetric(horizontal: 16);

    double resizeFactor = 1;

    if (imageWidth != null && imageHeight != null) {
      double k1 = displayWidth / imageWidth!;
      double k2 = maxImageWidgetHeight / imageHeight!;
      resizeFactor = min(k1, k2);
    }

    List<Bbox> bboxesWidgets = [];
    for (int i = 0; i < bboxes.length; i++) {
      final box = bboxes[i];
      final boxClass = classes[i];
      bboxesWidgets.add(
        Bbox(
            box[0] * resizeFactor,
            box[1] * resizeFactor,
            box[2] * resizeFactor,
            box[3] * resizeFactor,
            labels[boxClass],
            scores[i],
            bboxesColors[boxClass]),
      );
    }
    for (int i = 0; i < bboxesHat.length; i++) {
      final box = bboxesHat[i];
      final boxClass = classesHat[i];
      bboxesWidgets.add(
        Bbox(
            box[0] * resizeFactor,
            box[1] * resizeFactor,
            box[2] * resizeFactor,
            box[3] * resizeFactor,
            labels2[boxClass],
            scoresHat[i],
            bboxesColorsHat[boxClass]),
      );
    }
    for (int i = 0; i < bboxesCap.length; i++) {
      final box = bboxesCap[i];
      final boxClass = classesCap[i];
      bboxesWidgets.add(
        Bbox(
            box[0] * resizeFactor,
            box[1] * resizeFactor,
            box[2] * resizeFactor,
            box[3] * resizeFactor,
            labels3[boxClass],
            scoresCap[i],
            bboxesColorsCap[boxClass]),
      );
    }
    for (int i = 0; i < bboxesHeadband.length; i++) {
      final box = bboxesHeadband[i];
      final boxClass = classesHeadband[i];
      bboxesWidgets.add(
        Bbox(
            box[0] * resizeFactor,
            box[1] * resizeFactor,
            box[2] * resizeFactor,
            box[3] * resizeFactor,
            labels4[boxClass],
            scoresHeadband[i],
            bboxesColorsHeadband[boxClass]),
      );
    }
    for (int i = 0; i < bboxesScarf.length; i++) {
      final box = bboxesScarf[i];
      final boxClass = classesScarf[i];
      bboxesWidgets.add(
        Bbox(
            box[0] * resizeFactor,
            box[1] * resizeFactor,
            box[2] * resizeFactor,
            box[3] * resizeFactor,
            labels5[boxClass],
            scoresScarf[i],
            bboxesColorsScarf[boxClass]),
      );
    }
    for (int i = 0; i < bboxesWatch.length; i++) {
      final box = bboxesWatch[i];
      final boxClass = classesWatch[i];
      bboxesWidgets.add(
        Bbox(
            box[0] * resizeFactor,
            box[1] * resizeFactor,
            box[2] * resizeFactor,
            box[3] * resizeFactor,
            labels6[boxClass],
            scoresWatch[i],
            bboxesColorsWatch[boxClass]),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('YOLO')),
      body: ListView(
        children: [
          InkWell(
            onTap: () async {
              final XFile? newImageFile =
                  await picker.pickImage(source: ImageSource.gallery);
              if (newImageFile != null) {
                setState(() {
                  imageFile = File(newImageFile.path);
                });
                final image =
                    img.decodeImage(await newImageFile.readAsBytes())!;
                imageWidth = image.width;
                imageHeight = image.height;
                inferenceOutput = model.infer(image);
                inferenceOutputHat = model2.infer(image);
                inferenceOutputCap = model3.infer(image);
                inferenceOutputHeadband = model4.infer(image);
                inferenceOutputScarf = model5.infer(image);
                inferenceOutputWatch = model6.infer(image);
                updatePostprocess();
              }
            },
            child: SizedBox(
              height: maxImageWidgetHeight,
              child: Center(
                child: Stack(
                  children: [
                    if (imageFile == null)
                      Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const Icon(
                            Icons.file_open_outlined,
                            size: 80,
                          ),
                          Text(
                            'Pick an image',
                            style: Theme.of(context).textTheme.headlineMedium,
                          ),
                        ],
                      )
                    else
                      Image.file(imageFile!),
                    ...bboxesWidgets,
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: 30),
          Padding(
            padding: textPadding,
            child: Row(
              children: [
                Text(
                  'Confidence threshold:',
                  style: Theme.of(context).textTheme.bodyLarge,
                ),
                const SizedBox(width: 8),
                Text(
                  '${(confidenceThreshold * 100).toStringAsFixed(0)}%',
                  style: Theme.of(context)
                      .textTheme
                      .bodyLarge
                      ?.copyWith(fontWeight: FontWeight.bold),
                ),
              ],
            ),
          ),
          const Padding(
            padding: textPadding,
            child: Text(
              'If high, only the clearly recognizable objects will be detected. If low even not clear objects will be detected.',
            ),
          ),
          Slider(
            value: confidenceThreshold,
            min: 0,
            max: 1,
            divisions: 100,
            onChanged: (value) {
              setState(() {
                confidenceThreshold = value;
                updatePostprocess();
              });
            },
          ),
          const SizedBox(height: 8),
          Padding(
            padding: textPadding,
            child: Row(
              children: [
                Text(
                  'IoU threshold',
                  style: Theme.of(context).textTheme.bodyLarge,
                ),
                const SizedBox(width: 8),
                Text(
                  '${(iouThreshold * 100).toStringAsFixed(0)}%',
                  style: Theme.of(context)
                      .textTheme
                      .bodyLarge
                      ?.copyWith(fontWeight: FontWeight.bold),
                ),
              ],
            ),
          ),
          const Padding(
            padding: textPadding,
            child: Text(
              'If high, overlapped objects will be detected. If low, only separated objects will be correctly detected.',
            ),
          ),
          Slider(
            value: iouThreshold,
            min: 0,
            max: 1,
            divisions: 100,
            onChanged: (value) {
              setState(() {
                iouThreshold = value;
                updatePostprocess();
              });
            },
          ),
          SwitchListTile(
            value: agnosticNMS,
            title: Text(
              'Agnostic NMS',
              style: Theme.of(context).textTheme.bodyLarge,
            ),
            subtitle: Text(
              agnosticNMS
                  ? 'Treat all the detections as the same object'
                  : 'Detections with different labels are different objects',
            ),
            onChanged: (value) {
              setState(() {
                agnosticNMS = value;
                updatePostprocess();
              });
            },
          ),
        ],
      ),
    );
  }

  void updatePostprocess() {
    if (inferenceOutput == null || inferenceOutput!.isEmpty) {
      debugPrint("Inference output is empty or null");
      return;
    }

    // Hasil dari model glasses-sunglasses
    final (newClasses1, newBboxes1, newScores1) = model.postprocess(
      inferenceOutput!,
      imageWidth!,
      imageHeight!,
      confidenceThreshold: confidenceThreshold,
      iouThreshold: iouThreshold,
      agnostic: agnosticNMS,
    );

    // Hasil dari model hat
    final (newClasses2, newBboxes2, newScores2) = model2.postprocess(
      inferenceOutputHat!,
      imageWidth!,
      imageHeight!,
      confidenceThreshold: confidenceThreshold,
      iouThreshold: iouThreshold,
      agnostic: agnosticNMS,
    );

    // Hasil dari model cap
    final (newClasses3, newBboxes3, newScores3) = model3.postprocess(
      inferenceOutputCap!,
      imageWidth!,
      imageHeight!,
      confidenceThreshold: confidenceThreshold,
      iouThreshold: iouThreshold,
      agnostic: agnosticNMS,
    );

    // Hasil dari model headband
    final (newClasses4, newBboxes4, newScores4) = model4.postprocess(
      inferenceOutputHeadband!,
      imageWidth!,
      imageHeight!,
      confidenceThreshold: confidenceThreshold,
      iouThreshold: iouThreshold,
      agnostic: agnosticNMS,
    );

    // Hasil dari model scarf
    final (newClasses5, newBboxes5, newScores5) = model5.postprocess(
      inferenceOutputScarf!,
      imageWidth!,
      imageHeight!,
      confidenceThreshold: confidenceThreshold,
      iouThreshold: iouThreshold,
      agnostic: agnosticNMS,
    );
    // Hasil dari model Watch
    final (newClasses6, newBboxes6, newScores6) = model6.postprocess(
      inferenceOutputWatch!,
      imageWidth!,
      imageHeight!,
      confidenceThreshold: confidenceThreshold,
      iouThreshold: iouThreshold,
      agnostic: agnosticNMS,
    );

    debugPrint('Detected from model 1: ${newScores1} classes');
    debugPrint('Detected from model 2: ${newScores2} classes hat');
    debugPrint('Detected from model 3: ${newScores3} classes Cap');
    debugPrint('Detected from model 4: ${newScores4} classes headband');
    debugPrint('Detected from model 5: ${newScores5} classes headband');
    debugPrint('Detected from model 6: ${newScores6} classes headband');

    // Gabungkan hasil dari kedua model
    setState(() {
      classes = newClasses1;
      bboxes = newBboxes1;
      scores = newScores1;
      classesHat = newClasses2;
      bboxesHat = newBboxes2;
      scoresHat = newScores2;
      classesCap = newClasses3;
      bboxesCap = newBboxes3;
      scoresCap = newScores3;
      classesHeadband = newClasses4;
      bboxesHeadband = newBboxes4;
      scoresHeadband = newScores4;
      classesScarf = newClasses5;
      bboxesScarf = newBboxes5;
      scoresScarf = newScores5;
      classesWatch = newClasses6;
      bboxesWatch = newBboxes6;
      scoresWatch = newScores6;
    });
  }
}
