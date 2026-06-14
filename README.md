# BillboardVision
This project focuses on evaluating and improving human detection models for real-world surveillance scenarios.

The project has three main objectives:

1. Model Benchmarking
Develop a robust evaluation pipeline to assess various object detection models on video data. The current implementation includes several one-stage detectors, primarily YOLO-based models. Future work will extend the benchmark to include two-stage detectors and evaluate their performance under identical conditions.

2. Domain Adaptation for High-Angle Surveillance
Improve the performance of human detection models for high-angle camera views, such as cameras installed on billboards and other elevated structures. A particular focus is enhancing detection robustness across diverse clothing styles, including various forms of veiled clothing that may be underrepresented in existing training datasets.

3. Billboard Audience Detection and Analytics
Develop a system that not only counts people in a scene but also identifies individuals who are looking at a specific target, such as an advertising billboard. The system will estimate audience size and store relevant analytics data in a database for further analysis.

The repository also contains videos captured from high-angle viewpoints, extracted video frames, corresponding annotation files, and benchmark results obtained using pretrained models. Fine-tuned models and updated evaluation results will be added as the project progresses.
